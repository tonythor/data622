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
import base64
import multiprocessing as mp
from collections import Counter
from datetime import timedelta
from functools import partial
from io import BytesIO
from pathlib import Path
from time import time
from typing import Generator, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
from IPython.display import HTML
from loguru import logger
import joblib  # Useful for saving/loading large models
from imblearn.over_sampling import SMOTE

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV



# Constants
INITIAL_JOIN = "merged_clean.parquet"
IMDB_AUGMENTED = "augmented.parquet"
CHUNK_SIZE = 250_000  # reduce to 100_000 for older macs
DICTIONARY_PAGESIZE_LIMIT = 2097152  # 2MB instead of 1048576 for older macs
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

            # total_size = int(response.headers.get('content-length', 0))
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
    start_time = time()

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
            usecols=['tconst', 'primaryTitle', 'runtimeMinutes', 'genres'] 
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

        actor_metrics.columns = ['tconst', 'num_actors', 'actor_names']
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
            version="2.4",
            use_dictionary=True,
            dictionary_pagesize_limit=DICTIONARY_PAGESIZE_LIMIT
        )
        logger.info(f"Merged dataset saved to: {output_path}")
        end_time = time()
        elapsed_time = timedelta(seconds=end_time - start_time)
        logger.info(f"\nTotal initial_load_and_merge execution time: {elapsed_time} \n")

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
        df['actor_names'] = df['actor_names'].fillna('')

        # Process in batches for memory efficiency
        batch_size = 100_000
        actor_tables = []

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]

            # Keep original column name 'actor_names'
            actor_batch = (
                batch[['tconst', 'actor_names']]
                .assign(actor_names=lambda x: x['actor_names'].str.split(', '))
                .explode('actor_names')
            )
            actor_tables.append(actor_batch)

        # Combine results efficiently
        actor_table = pd.concat(actor_tables, ignore_index=True)
        return actor_table[actor_table['actor_names'].str.len() > 0]
    except Exception as e:
        logger.error(f"Error extracting actor table: {e}")
        raise

def add_experienced_actor_features(df: pd.DataFrame, actor_table: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized version with parallel processing for actor experience calculation.
    """
    # Create actor experience dictionary directly from value_counts()
    experience_dict = actor_table['actor_names'].value_counts().to_dict()

    num_cores = max(2, mp.cpu_count() - 1)  # at least two! 
    chunk_size = len(df) // num_cores
    actor_chunks = np.array_split(df['actor_names'], num_cores)

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
        'rating_bin', 'num_actors', 'actor_names'
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
        start_time = time()
        # Load merged dataset with memory optimization
        df = pd.read_parquet(
            Path(data_dir) / INITIAL_JOIN,
            columns=['tconst', 'primaryTitle', 'runtimeMinutes', 'numVotes',
                    'averageRating', 'genres', 'num_actors', 'actor_names']
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
            dictionary_pagesize_limit=DICTIONARY_PAGESIZE_LIMIT
        )
        logger.info(f"Augmented dataset saved to: {output_path}")
        end_time = time()
        elapsed_time = timedelta(seconds=end_time - start_time)
        logger.info(f"\nTotal augment_add_columns execution time: {elapsed_time} \n")
    except Exception as e:
        logger.error(f"Error augmenting dataset: {e}")
        raise


def train_and_evaluate_rf(df, target='rating_bin', test_size=0.2, n_estimators=100,
                          random_state=42, cache_path='rf_model.joblib', rerun=False, show_output=True):
    """
    Train and evaluate a Random Forest model on the given IMDB dataset with caching option.
    
    Parameters:
    -----------
    df                 : pandas DataFrame          : Input DataFrame containing features and target
    target             : str, default='rating_bin' : Name of the target column
    test_size          : float, default=0.2        : Proportion of test split
    n_estimators       : int, default=100          : Number of trees in the forest
    random_state       : int, default=42           : Random state for reproducibility
    cache_path         : str, default='rf_model.joblib' : Path to cache the trained model
    rerun              : bool, default=False       : If False, load cached model if available

    Returns:
    --------
    rf_model           : Trained RandomForestRegressor model
    feature_importance : DataFrame with feature importance
    metrics            : Tuple with R2 score and RMSE
    predictions        : Tuple of (y_test, y_pred) for summary reporting
    """

    # Check if a cached model exists and rerun is False
    if os.path.exists(cache_path) and not rerun:
        print(f"Loading cached model from {cache_path}...")
        rf_model = joblib.load(cache_path)
        
        # Load data for predictions and metrics
        X = df.drop(target, axis=1)
        y = df[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        if show_output:
            print("Model loaded successfully with the following performance:")
            print(f"R2 Score: {r2:.3f}")
            print(f"Root Mean Squared Error: {rmse:.3f}")
        
        return rf_model, feature_importance, (r2, rmse), (y_test, y_pred)

    # If no cached model or rerun is True, train a new model
    if show_output: print("Training a new Random Forest model...")
    start_time = time()

    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train the Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    # Save the model to the cache path
    joblib.dump(rf_model, cache_path)
    if show_output: print(f"Model trained and saved to {cache_path}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Make predictions and calculate metrics
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if show_output: 
        print("Model Performance:")
        print(f"R2 Score: {r2:.3f}")
        print(f"Root Mean Squared Error: {rmse:.3f}")

    # Calculate elapsed time
    end_time = time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    if show_output: print(f"\nTotal model train and execution time: {elapsed_time} \n")

    return rf_model, feature_importance, (r2, rmse), (y_test, y_pred)

def train_and_evaluate_svm(df, target='rating_bin', test_size=0.2, kernel='rbf', 
                           C=1.0, epsilon=0.1, random_state=42, 
                           cache_path='svm_model.joblib', rerun=False, show_output=True):
    """
    Train and evaluate an SVM model on the given dataset with caching option.

    Parameters:
    -----------
    df                 : pandas DataFrame          : Input DataFrame containing features and target
    target             : str, default='rating_bin' : Name of the target column
    test_size          : float, default=0.2        : Proportion of test split
    kernel             : str, default='rbf'        : Kernel type for SVM (e.g., 'linear', 'rbf')
    C                  : float, default=1.0        : Regularization parameter
    epsilon            : float, default=0.1        : Epsilon in the epsilon-SVR model
    random_state       : int, default=42           : Random state for reproducibility
    cache_path         : str, default='svm_model.joblib' : Path to cache the trained model
    rerun              : bool, default=False       : If False, load cached model if available

    Returns:
    --------
    svm_model          : Trained SVR model
    metrics            : Tuple with R2 score and RMSE
    predictions        : Tuple of (y_test, y_pred) for summary reporting
    """

    # Check if a cached model exists and rerun is False
    if os.path.exists(cache_path) and not rerun:
        print(f"Loading cached SVM model from {cache_path}...")
        svm_model = joblib.load(cache_path)
        
        # Load data for predictions and metrics
        X = df.drop(target, axis=1)
        y = df[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_pred = svm_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if show_output:
            print("SVM Model loaded successfully with the following performance:")
            print(f"R2 Score: {r2:.3f}")
            print(f"Root Mean Squared Error: {rmse:.3f}")
        
        return svm_model, (r2, rmse), (y_test, y_pred)

    # If no cached model or rerun is True, train a new model
    if show_output: print("Training a new SVM model...")
    start_time = time()

    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train the SVM model
    svm_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svm_model.fit(X_train, y_train)

    # Save the model to the cache path
    joblib.dump(svm_model, cache_path)
    if show_output: print(f"SVM model trained and saved to {cache_path}")

    # Make predictions and calculate metrics
    y_pred = svm_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if show_output:
        print("SVM Model Performance:")
        print(f"R2 Score: {r2:.3f}")
        print(f"Root Mean Squared Error: {rmse:.3f}")
 
    # Calculate elapsed time
    end_time = time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    if show_output: print(f"\nTotal SVM model train and execution time: {elapsed_time} \n")

    return svm_model, (r2, rmse), (y_test, y_pred)

def train_and_evaluate_svm_optimized(
    df, target='rating_bin', test_size=0.2, random_state=42, 
    cache_path='svm_model_opt.joblib', rerun=False, sample_size=50000,
    n_iter=5, scoring='r2', cv=5, verbose=2, show_output=True):
    """
    Optimized version of SVM training with hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    -----------
    df : pandas DataFrame : Input DataFrame containing features and target.
    target : str : Name of target column (default='rating_bin').
    test_size : float : Proportion for test split (default=0.2).
    random_state : int : Random state for reproducibility (default=42).
    cache_path : str : Where to save/load model (default='svm_model_opt.joblib').
    rerun : bool : Whether to force retraining (default=False).
    sample_size : int : Total sample size to use (default=50000).
    n_iter : int : Number of parameter settings sampled in RandomizedSearchCV (default=5).
    scoring : str : Scoring metric for evaluation (default='r2').
    cv : int : Number of cross-validation folds (default=5).
    verbose : int : Verbosity level for RandomizedSearchCV (default=2).
    
    Returns:
    --------
    Trained SVM model, metrics (R2, RMSE), and test data (y_test, y_pred).
    """
    # Check cache
    if os.path.exists(cache_path) and not rerun:
        print(f"Loading cached optimized SVM model from {cache_path}...")
        svm_model = joblib.load(cache_path)
        
        # Prepare test data for evaluation
        df_stratified = df.groupby(target, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), sample_size // 10), random_state=random_state)
        )
        X = df_stratified.drop(target, axis=1)
        y = df_stratified[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        _, X_test, _, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        y_pred = svm_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if show_output:
            print("SVM Model loaded successfully with the following performance:")
            print(f"R2 Score: {r2:.3f}")
            print(f"Root Mean Squared Error: {rmse:.3f}")
        
        return svm_model, (r2, rmse), (y_test, y_pred)

    if show_output: print("Training new optimized SVM model...")
    start_time = time()
    
    # Stratified sampling
    df_stratified = df.groupby(target, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), sample_size // 10), random_state=random_state)
    )
    
    # Feature scaling
    X = df_stratified.drop(target, axis=1)
    y = df_stratified[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    
    # RandomizedSearchCV
    print("Performing hyperparameter optimization...")
    svr = SVR()
    random_search = RandomizedSearchCV(
        estimator=svr,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    # Best model
    svm_model = random_search.best_estimator_
    print("Best hyperparameters found:", random_search.best_params_)
    
    # Save the model
    joblib.dump(svm_model, cache_path)
    if show_output: print(f"Model saved to {cache_path}")
    
    # Predictions and metrics
    y_pred = svm_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if show_output:
        print("Model Performance:")
        print(f"R2 Score: {r2:.3f}")
        print(f"Root Mean Squared Error: {rmse:.3f}")
    
    end_time = time()
    if show_output: print(f"Total execution time: {timedelta(seconds=end_time - start_time)}")
    
    return svm_model, (r2, rmse), (y_test, y_pred)

def generate_summary_report(y_test, y_pred, report_name="CHANGE ME"):
    # Calculate metrics
    y_pred_rounded = np.round(y_pred)
    exact_matches = (y_pred_rounded == y_test).mean()
    close_predictions = (abs(y_pred_rounded - y_test) <= 1).mean()
    differences = Counter(y_pred_rounded - y_test)
    major_errors = sum(v for k, v in differences.items() if abs(k) >= 3) / len(y_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Create summary as a single f-string
    summary = f"""
{'='*80}
Results for: {report_name}
{'='*80}

Quick Summary:
- Gets the exact rating {exact_matches:.0%} of the time
- Within one rating point {close_predictions:.0%} of the time
- Makes big mistakes only {major_errors:.1%} of the time

Technical Details:
- R² Score: {r2:.3f}
- RMSE: {rmse:.3f}

What This Means:
If you're trying to decide if a movie will be good (7-8), great (8-9),
or excellent (9-10), this model will get it right or be off by just
one rating point {close_predictions:.0%} of the time.
{'='*80}"""
    
    print(summary)

# def generate_summary_report_3(y_test, y_pred):
#     """
#     Generate a friendly, human-readable summary of the model's performance.
#     This is for assignment 3 only. 

#     Parameters:
#     -----------
#     y_test : array : Ground truth target values
#     y_pred : array : Predicted target values

#     Returns:
#     --------
#     str : A formatted summary report of the model's performance
#     """
#     # Round predictions for comparison
#     y_pred_rounded = np.round(y_pred)

#     # Calculate key metrics
#     exact_matches = (y_pred_rounded == y_test).mean()
#     close_predictions = (abs(y_pred_rounded - y_test) <= 1).mean()
#     differences = Counter(y_pred_rounded - y_test)

#     # Calculate additional insights
#     major_errors = sum(v for k, v in differences.items() if abs(k) >= 3) / len(y_test)
#     zero_bin_percentage = (differences.get(0, 0) / len(y_test)) * 100
#     low_bin_percentage = (differences.get(-1, 0) / len(y_test)) * 100
#     high_bin_percentage = (differences.get(1, 0) / len(y_test)) * 100
#     close_bin_percentage = low_bin_percentage + high_bin_percentage

#     # Build the summary report
#     summary = (
#         f"**Accuracy Measures:**\n"
#         f"- {exact_matches:.2%} exact matches (got the rating bin exactly right)\n"
#         f"- {close_predictions:.2%} within 1 bin (either exact or just one bin off)\n\n"
#         f"**Distribution of Errors:**\n"
#         f"- {zero_bin_percentage:.1f}% perfect predictions (0 bins off)\n"
#         f"- {close_bin_percentage:.1f}% off by just 1 bin "
#         f"({low_bin_percentage:.1f}% low + {high_bin_percentage:.1f}% high)\n"
#         f"- Only {100 - close_predictions * 100:.1f}% off by more than 1 bin\n"
#         f"- Major mistakes (off by 3 or more bins): {major_errors:.1%}\n\n"
#         f"**Putting It in Perspective:**\n"
#         f"If you're trying to predict if a movie is 'good' (7–8), 'great' (8–9), \n"
#         f"or 'excellent' (9–10), you'll be within the right range {close_predictions:.2%} of the time.\n"
#         f"Major errors are pretty rare, happening less than {major_errors * 100:.1f}% of the time."
#     )

#     print(summary)



def plot_importance(importance_df, top_n=10):
    """
    Create horizontal feature importance plot and return base64 encoded image for Quarto.
    
    Parameters:
    -----------
    importance_df : pandas DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default=10
        Number of top features to show
        
    Returns:
    --------
    HTML : IPython.display.HTML
        HTML object containing the base64 encoded image
    """
    # Take top N features but reverse order for bottom-to-top display
    plot_df = importance_df.head(top_n).iloc[::-1]
    
    # Create horizontal bar plot
    fig = px.bar(
        plot_df,
        y='feature',
        x='importance',
        text=plot_df['importance'].round(3),
        title='Top Feature Importances',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        orientation='h'
    )
    
    # Customize layout
    fig.update_layout(
        width=800,
        height=500,
        showlegend=False,
        title_x=0.5,
        margin=dict(t=50, l=200, r=50, b=50),
        yaxis={'dtick': 1}
    )
    
    # Customize bars
    fig.update_traces(
        marker_color='#2563eb',
        textposition='outside',
        texttemplate='%{text:.3f}'
    )
    
    # Convert plotly figure to PNG image
    img_bytes = pio.to_image(fig, format="png")
    
    # Encode the image to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Create HTML for the image
    img_html = f'<img src="data:image/png;base64,{img_base64}" alt="Feature Importance Plot" />'
    
    return HTML(img_html)


def prebuild():
    """
    Call this only if you've not prebuilt your data directory already.
    """
    data_dir="622data_nogit/imdb"
    download_imdb_data(data_dir)
    initial_load_and_merge(data_dir)
    augment_add_columns(data_dir)

def histogram_likert(df):
    """
    Generates a histogram for the 'experienced_actors_likert' column 
    with a minimal theme and customized titles.

    Parameters:
    -----------------
    df : pandas DataFrame : The DataFrame containing the column to plot.
    """
    # Create the Histogram
    fig = px.histogram(
        df,
        x='experienced_actors_likert', 
        title='Likert Score Distribution',
        width=800,  # Set the width of the plot
        height=500,  # Set the height of the plot
        template="simple_white"
    )
    
    # Center the title
    fig.update_layout(
        title_x=0.5,
        xaxis_title='Experienced Actors Likert Scores',
        yaxis_title='Count'
    )
                     
    fig.show()
def train_and_evaluate_basic_tree(df, target='rating_bin', 
                                test_size=0.2, max_depth=10,
                                random_state=42, cache_path='basic_tree.joblib',
                                rerun=False, show_output=True):
    """
    Train and evaluate a basic decision tree using simple features.
    
    Parameters:
    -----------
    df : pandas DataFrame : Input DataFrame containing features and target
    target : str : Name of target column (default='rating_bin')
    test_size : float : Proportion of dataset to use for testing
    max_depth : int : Maximum depth of the decision tree
    random_state : int : Random seed for reproducibility
    cache_path : str : Path to cache the trained model
    rerun : bool : Whether to retrain model even if cached version exists

    Returns:
    --------
    tree_model : Trained DecisionTreeRegressor
    metrics : Tuple with (R2 score, RMSE)
    predictions : Tuple of (y_test, y_pred)
    """
    # Check if cached model exists
    if os.path.exists(cache_path) and not rerun:
        if show_output: print(f"Loading cached basic decision tree from {cache_path}...")
        tree_model = joblib.load(cache_path)
        
        # Basic features only
        basic_features = ['runtimeMinutes', 'num_actors', 'Action', 'Comedy', 
                         'Drama', 'Documentary', 'Romance']
        X = df[basic_features]
        y = df[target]
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        y_pred = tree_model.predict(X_test)
        
    else:
        if show_output: print("Training new basic decision tree...")
        # Select basic features
        basic_features = ['runtimeMinutes', 'num_actors', 'Action', 'Comedy', 
                         'Drama', 'Documentary', 'Romance']
        X = df[basic_features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        tree_model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
        tree_model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(tree_model, cache_path)
        y_pred = tree_model.predict(X_test)
        
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if show_output:   
        print("Basic Decision Tree Performance:")
        print(f"R2 Score: {r2:.3f}")
        print(f"Root Mean Squared Error: {rmse:.3f}")
    
    return tree_model, (r2, rmse), (y_test, y_pred)

def train_and_evaluate_complex_tree(df, target='rating_bin',
                                  test_size=0.2, max_depth=10,
                                  random_state=42, cache_path='complex_tree.joblib',
                                  rerun=False, show_output=True):
    """
    Train and evaluate a more complex decision tree using advanced features.
    
    Parameters:
    -----------
    df : pandas DataFrame : Input DataFrame containing features and target
    target : str : Name of target column (default='rating_bin')
    test_size : float : Proportion of dataset to use for testing
    max_depth : int : Maximum depth of the decision tree
    random_state : int : Random seed for reproducibility
    cache_path : str : Path to cache the trained model
    rerun : bool : Whether to retrain model even if cached version exists

    Returns:
    --------
    tree_model : Trained DecisionTreeRegressor
    metrics : Tuple with (R2 score, RMSE)
    predictions : Tuple of (y_test, y_pred)
    """
    # Check if cached model exists
    if os.path.exists(cache_path) and not rerun:
        print(f"Loading cached complex decision tree from {cache_path}...")
        tree_model = joblib.load(cache_path)
        
        # Complex features
        complex_features = ['runtimeMinutes', 'num_actors', 'experienced_actor_count',
                          'experienced_actors_likert', 'Action', 'Comedy', 'Drama',
                          'Documentary', 'Romance', 'Horror', 'Thriller']
        X = df[complex_features]
        y = df[target]
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        y_pred = tree_model.predict(X_test)
        
    else:
        if show_output: print("Training new complex decision tree...")
        # Select complex features
        complex_features = ['runtimeMinutes', 'num_actors', 'experienced_actor_count',
                          'experienced_actors_likert', 'Action', 'Comedy', 'Drama',
                          'Documentary', 'Romance', 'Horror', 'Thriller']
        X = df[complex_features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        tree_model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
        tree_model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(tree_model, cache_path)
        y_pred = tree_model.predict(X_test)
        
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if show_output:
        print("Complex Decision Tree Performance:")
        print(f"R2 Score: {r2:.3f}")
        print(f"Root Mean Squared Error: {rmse:.3f}")
    
    return tree_model, (r2, rmse), (y_test, y_pred)
