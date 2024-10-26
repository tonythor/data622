"""
Data preparation functions to load raw IMDB data, merge it, and add numeric columns for analysis.
Includes cross-platform support, memory-efficient chunking, and robust error handling.

More on the dataset: https://developer.imdb.com/non-commercial-datasets/
"""

# Standard library imports
import os
import gc
import gzip
import shutil
from pathlib import Path
from typing import Optional, Generator, List, Tuple
from functools import partial

# Third-party imports
import numpy as np
import pandas as pd
import requests
from loguru import logger
import multiprocessing as mp

# Constants
INITIAL_JOIN = "merged_clean.parquet"
IMDB_AUGMENTED = "augmented.parquet"
CHUNK_SIZE = 250_000  # reduce to 100_000 for older macs
dictionary_pagesize_limit = 2097152  # 2MB instead of 1048576 for older macs

def download_imdb_data(data_dir: str) -> None:
    """
    Download IMDB datasets using requests (cross-platform compatible).
    
    Args:
        data_dir (str): Target directory for downloaded files
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    imdb_urls = [
        "https://datasets.imdbws.com/name.basics.tsv.gz",
        "https://datasets.imdbws.com/title.akas.tsv.gz",
        "https://datasets.imdbws.com/title.basics.tsv.gz",
        "https://datasets.imdbws.com/title.crew.tsv.gz",
        "https://datasets.imdbws.com/title.episode.tsv.gz",
        "https://datasets.imdbws.com/title.principals.tsv.gz",
        "https://datasets.imdbws.com/title.ratings.tsv.gz",
    ]
    
    for url in imdb_urls:
        file_name = url.split("/")[-1]
        gz_path = data_path / file_name
        tsv_path = data_path / file_name.replace('.gz', '')
        
        logger.info(f"Downloading {file_name}...")
        
        try:
            # Download with progress tracking
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(gz_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
            
            # Extract gz file
            logger.info(f"Extracting {file_name}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tsv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove gz file after extraction
            gz_path.unlink()
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            raise


def process_dataframe_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single chunk of the DataFrame.
    """
    # Convert runtimeMinutes to numeric, replacing invalid values with NaN
    if 'runtimeMinutes' in chunk.columns:
        chunk['runtimeMinutes'] = pd.to_numeric(chunk['runtimeMinutes'], errors='coerce')
    
    return chunk
def read_tsv_in_chunks(file_path: Path) -> Generator[pd.DataFrame, None, None]:
    """
    Read TSV file in chunks to manage memory usage.
    """
    for chunk in pd.read_csv(file_path, sep='\t', na_values='\\N', 
                           chunksize=CHUNK_SIZE, low_memory=False):
        yield process_dataframe_chunk(chunk)

def initial_load_and_merge(data_dir: str) -> None:
    """
    Load and merge IMDB datasets with optimized processing and memory management.
    """
    data_path = Path(data_dir)
    
    try:
        # Read basics with optimized dtypes
        dtypes = {
            'tconst': 'category',
            'titleType': 'category',
            'primaryTitle': 'string',
            'originalTitle': 'string',
            'isAdult': 'int8',
            'startYear': 'float32',
            'endYear': 'float32',
            'runtimeMinutes': 'string',
            'genres': 'string'
        }
        
        logger.info("Loading basics...")
        basics_df = pd.read_csv(
            data_path / "title.basics.tsv",
            sep='\t',
            na_values='\\N',
            dtype=dtypes,
            usecols=['tconst', 'primaryTitle', 'runtimeMinutes', 'genres']  # Only load needed columns
        )
        basics_df['runtimeMinutes'] = pd.to_numeric(basics_df['runtimeMinutes'], errors='coerce')
        
        logger.info("Loading ratings...")
        # Load ratings all at once since it's typically small
        ratings_df = pd.read_csv(
            data_path / "title.ratings.tsv",
            sep='\t',
            na_values='\\N',
            dtype={'tconst': 'category', 'averageRating': 'float32', 'numVotes': 'int32'}
        )
        
        logger.info("Merging ratings with basics...")
        # Merge ratings and basics
        merged_df = pd.merge(ratings_df, basics_df, on='tconst')
        del ratings_df, basics_df
        gc.collect()
        
        logger.info("Loading names...")
        # Load only required columns from names
        names_df = pd.read_csv(
            data_path / "name.basics.tsv",
            sep='\t',
            na_values='\\N',
            usecols=['nconst', 'primaryName'],
            dtype={'nconst': 'category', 'primaryName': 'string'}
        )
        
        logger.info("Processing principals...")
        # Process principals in chunks with optimized merging
        principals_list = []
        principals_dtype = {
            'tconst': 'category',
            'ordering': 'int8',
            'nconst': 'category',
            'category': 'category'
        }
        
        # Read principals in chunks and process each chunk
        for chunk in pd.read_csv(
            data_path / "title.principals.tsv",
            sep='\t',
            na_values='\\N',
            dtype=principals_dtype,
            usecols=['tconst', 'nconst'],  # Only load needed columns
            chunksize=CHUNK_SIZE
        ):
            # Merge with names efficiently
            chunk_merged = pd.merge(chunk, names_df, on='nconst')
            principals_list.append(chunk_merged)
            
        # Combine all principal chunks
        principals_merged = pd.concat(principals_list, ignore_index=True)
        del principals_list, names_df
        gc.collect()
        
        logger.info("Calculating actor metrics...")
        # Calculate actor counts and names more efficiently
        actor_metrics = principals_merged.groupby('tconst').agg({
            'nconst': 'count',
            'primaryName': lambda x: ', '.join(x.dropna())
        }).reset_index()
        
        actor_metrics.columns = ['tconst', 'numActors', 'actorNames']
        del principals_merged
        gc.collect()
        
        logger.info("Performing final merge...")
        # Final merge with optimized memory usage
        final_df = merged_df.merge(actor_metrics, on='tconst', how='left')
        del merged_df, actor_metrics
        gc.collect()
        
        logger.info("Saving to parquet...")
        # Save with optimized compression
        output_path = data_path / INITIAL_JOIN
        final_df.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
            version="2.0",
            use_dictionary=True,
            dictionary_pagesize_limit=dictionary_pagesize_limit
        )
        logger.info(f"Merged dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise

def load_imdb(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load the prebuilt IMDB file with error handling.
    """
    try:
        df = pd.read_parquet(Path(data_dir) / IMDB_AUGMENTED)
        logger.info(f"Loaded augmented dataset from: {IMDB_AUGMENTED}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: {IMDB_AUGMENTED} not found in {data_dir}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None
    

def parallel_actor_experience(actor_chunk: pd.Series, experience_dict: dict) -> List[Tuple[int, int]]:
    """
    Process actor experience calculations in parallel.
    """
    results = []
    for actor_names in actor_chunk:
        if pd.isna(actor_names) or actor_names == '':
            results.append((0, 1))
            continue
            
        actor_list = actor_names.split(', ')
        experienced_count = sum(1 for actor in actor_list 
                              if experience_dict.get(actor, 0) > 10)
        
        avg_count = sum(experience_dict.get(actor, 0) for actor in actor_list) / len(actor_list)
        likert_score = (
            5 if avg_count >= 30 else
            4 if avg_count >= 20 else
            3 if avg_count >= 10 else
            2 if avg_count >= 5 else
            1
        )
        results.append((experienced_count, likert_score))
    return results

def extract_actor_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized actor table extraction with batch processing.
    """
    try:
        # Pre-allocate memory for better performance
        df['actorNames'] = df['actorNames'].fillna('')
        
        # Process in batches for memory efficiency
        batch_size = 100_000
        actor_tables = []
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            # Keep original column name 'actorNames'
            actor_batch = (
                batch[['tconst', 'actorNames']]
                .assign(actorNames=lambda x: x['actorNames'].str.split(', '))
                .explode('actorNames')
            )
            actor_tables.append(actor_batch)
        
        # Combine results efficiently
        actor_table = pd.concat(actor_tables, ignore_index=True)
        return actor_table[actor_table['actorNames'].str.len() > 0]
        
    except Exception as e:
        logger.error(f"Error extracting actor table: {e}")
        raise

def add_experienced_actor_features(df: pd.DataFrame, actor_table: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized version with parallel processing for actor experience calculation.
    """
    # Create actor experience dictionary directly from value_counts()
    experience_dict = actor_table['actorNames'].value_counts().to_dict()
    
    num_cores = max(2, mp.cpu_count() - 1)  # at least two! 
    chunk_size = len(df) // num_cores
    actor_chunks = np.array_split(df['actorNames'], num_cores)
    
    # Process in parallel
    with mp.Pool(num_cores) as pool:
        results = pool.map(
            partial(parallel_actor_experience, experience_dict=experience_dict), 
            actor_chunks
        )
    
    # Flatten results
    all_results = [item for sublist in results for item in sublist]
    
    # Add results to DataFrame efficiently
    df['experienced_actor_count'] = [r[0] for r in all_results]
    df['experienced_actors_likert'] = [r[1] for r in all_results]
    
    return df

def add_rating_bins_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized rating bins and feature addition with vectorized operations.
    """
    # Use numpy for faster binning
    bins = np.array([-float('inf'), 1, 2, 3, 4, 5, 6, 7, 8, 9, float('inf')])
    df['rating_bin'] = np.digitize(df['averageRating'], bins) - 1
    
    if 'genres' in df.columns:
        # Vectorized genre processing
        genre_expanded = df['genres'].str.get_dummies(sep=',')
        
        # No longer using sparse matrices - they cause issues with parquet
        df = pd.concat([df, genre_expanded], axis=1)
    
    relevant_columns = [
        'tconst', 'primaryTitle', 'runtimeMinutes', 'numVotes', 
        'rating_bin', 'numActors', 'actorNames'
    ] + (list(genre_expanded.columns) if 'genres' in df.columns else [])
    
    # Optimize memory usage
    df = df[relevant_columns].copy()
    df = df.astype({col: 'category' for col in df.select_dtypes(['object']).columns})
    
    return df.dropna(subset=['rating_bin', 'runtimeMinutes'])
def augment_add_columns(data_dir: str) -> None:
    """
    Optimized main augmentation function with better memory management and parallel processing.
    """
    try:
        # Load merged dataset with memory optimization
        df = pd.read_parquet(
            Path(data_dir) / INITIAL_JOIN,
            columns=['tconst', 'primaryTitle', 'runtimeMinutes', 'numVotes', 
                    'averageRating', 'genres', 'numActors', 'actorNames']
        )
        
        # Process in stages with memory cleanup
        logger.info("Adding rating bins and features...")
        df = add_rating_bins_and_features(df)
        
        logger.info("Extracting actor table...")
        actor_table = extract_actor_table(df)
        
        logger.info("Adding actor experience features...")
        df = add_experienced_actor_features(df, actor_table)
        del actor_table  # Free memory
        gc.collect() 
        
        # Save augmented dataset with optimization
        output_path = Path(data_dir) / IMDB_AUGMENTED
        df.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            compression="zstd",
            version="2.4",
            use_dictionary=True,
            dictionary_pagesize_limit=dictionary_pagesize_limit
        )
        logger.info(f"Augmented dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error augmenting dataset: {e}")
        raise