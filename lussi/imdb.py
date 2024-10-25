import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

def build_imdb(data_dir: str, output_format: str = "parquet") -> None:
    """
    Build and merge IMDB datasets, including actor names, and save to Parquet or Arrow format.
    
    Args:
        data_dir (str): Path to the directory containing IMDB TSV files.
        output_format (str): Format to save the merged dataset (parquet or arrow). Default is parquet.
    """
    # Paths to the TSV files
    ratings_path = Path(data_dir) / "title.ratings.tsv"
    basics_path = Path(data_dir) / "title.basics.tsv"
    principals_path = Path(data_dir) / "title.principals.tsv"
    names_path = Path(data_dir) / "name.basics.tsv"
    
    # Load datasets
    ratings = pd.read_csv(ratings_path, sep='\t', na_values='\\N')
    basics = pd.read_csv(basics_path, sep='\t', na_values='\\N', low_memory=False)
    principals = pd.read_csv(principals_path, sep='\t', na_values='\\N', low_memory=False)
    names = pd.read_csv(names_path, sep='\t', na_values='\\N')

    # Convert runtimeMinutes to numeric, handling non-numeric values
    basics['runtimeMinutes'] = pd.to_numeric(basics['runtimeMinutes'], errors='coerce')

    # Merge ratings and basics on 'tconst'
    df = pd.merge(ratings, basics, on='tconst')

    # Add a binary 'HighRating' column
    df['HighRating'] = (df['averageRating'] >= 6.0).astype(int)

    # Merge principals with names to get actor names
    principals_with_names = pd.merge(principals, names[['nconst', 'primaryName']], on='nconst')

    # Group actor names by movie (tconst)
    actor_names = principals_with_names.groupby('tconst')['primaryName'].apply(lambda x: ', '.join(x)).reset_index()
    actor_names.columns = ['tconst', 'actorNames']

    # Merge actor names with the main DataFrame
    df = pd.merge(df, actor_names, on='tconst', how='left')

    # Select relevant columns
    df = df[['tconst', 'primaryTitle', 'genres', 'runtimeMinutes', 'numVotes', 'HighRating', 'numActors', 'actorNames']]
    df = df.dropna()

    # Save the final DataFrame to Parquet or Arrow
    output_path = Path(data_dir) / f"imdb_merged.{output_format}"
    
    if output_format == "parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd", version="2.0")
    elif output_format == "arrow":
        df.to_feather(output_path)
    else:
        raise ValueError("Unsupported format. Choose 'parquet' or 'arrow'.")

    print(f"IMDB merged dataset with actor names saved to: {output_path}")

    build_imdb("622data_nogit/imdb", "parquet")


def load_imdb(data_dir: str) -> pd.DataFrame:
    """
    Load the IMDB dataset from the data directory and return a Pandas 2.x DataFrame
    with Arrow-backed data types, if available.

    Args:
        data_dir (str): Path to the directory containing the IMDB Parquet or Arrow file.

    Returns:
        pd.DataFrame: Loaded DataFrame with Arrow-backed data types if possible.
    """
    # Define the path to the saved Parquet/Arrow file
    parquet_path = Path(data_dir) / "imdb_merged.parquet"

    # Load the DataFrame (with Arrow-backed data types if Pandas 2.x is installed)
    df = pd.read_parquet(parquet_path)

    # Confirm we're using Pandas 2.x
    pandas_version = pd.__version__
    print(f"Pandas version: {pandas_version}")

    # Check if Arrow-backed data types are being used
    print("\nDataFrame info:")
    df.info(verbose=True, memory_usage="deep")

    # Return the loaded DataFrame
    return df

    # df = load_imdb("622data_nogit/imdb")
    # print(df.head())
