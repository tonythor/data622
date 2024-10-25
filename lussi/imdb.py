import pandas as pd
from pathlib import Path

def build_imdb(data_dir: str, output_format: str = "parquet") -> None:
    """
    Build and merge IMDB datasets, saving the result to an optimized file format.
    
    Args:
        data_dir (str): Path to the directory containing IMDB TSV files.
        output_format (str): Format to save the merged dataset (parquet or arrow). Default is parquet.
    """
    # Define paths to all relevant files
    ratings_path = Path(data_dir) / "title.ratings.tsv"
    basics_path = Path(data_dir) / "title.basics.tsv"
    principals_path = Path(data_dir) / "title.principals.tsv"
    
    # Load datasets with appropriate settings
    ratings = pd.read_csv(ratings_path, sep='\t', na_values='\\N')
    basics = pd.read_csv(basics_path, sep='\t', na_values='\\N', low_memory=False)
    principals = pd.read_csv(principals_path, sep='\t', na_values='\\N', low_memory=False)

    # Convert 'runtimeMinutes' to numeric, coercing errors to NaN
    basics['runtimeMinutes'] = pd.to_numeric(basics['runtimeMinutes'], errors='coerce')

    # Merge datasets
    df = pd.merge(ratings, basics, on='tconst')

    # Add a binary 'HighRating' column
    df['HighRating'] = (df['averageRating'] >= 6.0).astype(int)

    # Calculate the number of actors per title
    actors = principals.groupby('tconst')['nconst'].count().reset_index()
    actors.columns = ['tconst', 'numActors']

    # Merge actor data with the main DataFrame
    df = pd.merge(df, actors, on='tconst', how='left')

    # Select and clean relevant columns
    df = df[['tconst', 'primaryTitle', 'genres', 'runtimeMinutes', 'numVotes', 'HighRating', 'numActors']]
    df = df.dropna()

    # Define output path and save the DataFrame
    output_path = Path(data_dir) / f"imdb_merged.{output_format}"

    if output_format == "parquet":
        df.to_parquet(output_path, index=False)
    elif output_format == "arrow":
        df.to_feather(output_path)
    else:
        raise ValueError("Unsupported format. Choose 'parquet' or 'arrow'.")

    print(f"IMDB merged dataset saved to: {output_path}")

# Example usage
# build_imdb("622data_nogit/imdb", "parquet")