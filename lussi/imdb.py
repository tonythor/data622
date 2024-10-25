import pandas as pd
from pathlib import Path

def build_imdb(data_dir: str, output_format: str = "parquet") -> pd.DataFrame:
    """
    Build and merge IMDB datasets, including actor names and expanded genres,
    then return a DataFrame with rating bins for multi-class classification.
    """
    # Paths to TSV files
    ratings_path = Path(data_dir) / "title.ratings.tsv"
    basics_path = Path(data_dir) / "title.basics.tsv"
    principals_path = Path(data_dir) / "title.principals.tsv"
    names_path = Path(data_dir) / "name.basics.tsv"

    # Load datasets
    ratings = pd.read_csv(ratings_path, sep='\t', na_values='\\N')
    basics = pd.read_csv(basics_path, sep='\t', na_values='\\N', low_memory=False)
    principals = pd.read_csv(principals_path, sep='\t', na_values='\\N', low_memory=False)
    names = pd.read_csv(names_path, sep='\t', na_values='\\N')

    # Convert runtimeMinutes to numeric
    basics['runtimeMinutes'] = pd.to_numeric(basics['runtimeMinutes'], errors='coerce')

    # Merge ratings and basics on 'tconst'
    df = pd.merge(ratings, basics, on='tconst')

    # Calculate the number of actors for each movie (group by 'tconst')
    num_actors = principals.groupby('tconst')['nconst'].count().reset_index()
    num_actors.columns = ['tconst', 'numActors']

    # Merge numActors with the main DataFrame
    df = pd.merge(df, num_actors, on='tconst', how='left')

    # Create rating bins (10 bins: 0-1, 1-2, ..., 9-10)
    df['rating_bin'] = pd.cut(df['averageRating'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=False, include_lowest=True)

    # Merge principals with names for actor information
    principals_with_names = pd.merge(principals, names[['nconst', 'primaryName']], on='nconst')
    actor_names = principals_with_names.groupby('tconst')['primaryName'].apply(lambda x: ', '.join(x.dropna())).reset_index()
    actor_names.columns = ['tconst', 'actorNames']

    # Merge actor names into the main DataFrame
    df = pd.merge(df, actor_names, on='tconst', how='left')

    # One-hot encode genres
    genre_expanded = df['genres'].str.get_dummies(sep=',')
    df = pd.concat([df, genre_expanded], axis=1)

    # Select relevant columns
    df = df[['tconst', 'primaryTitle', 'runtimeMinutes', 'numVotes', 'rating_bin', 'numActors', 'actorNames'] + list(genre_expanded.columns)]

    # Drop any remaining missing values
    df = df.dropna()

    # Save the final DataFrame to Parquet or Arrow
    output_path = Path(data_dir) / f"imdb_merged.{output_format}"
    if output_format == "parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd", version="2.0")
    elif output_format == "arrow":
        df.to_feather(output_path)

    print(f"IMDB merged dataset saved to: {output_path}")


	# Example usage: Build the dataset
	#df = build_imdb("622data_nogit/imdb", "parquet")
	#print(df.head())

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
