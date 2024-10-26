"""
A tidy/glimpse like function but for python. It works for pandas dataframe.

"""

def glimpse(df):
    """
    Display DataFrame info similar to R's glimpse function
    """
    # Basic DataFrame info
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumn preview:")
    print("-" * 80)
    
    # For each column, show type and first few values
    for col in df.columns:
        # Get sample values but handle potential errors
        try:
            sample = df[col].head(5).tolist()
            # Truncate long strings
            sample = [str(x)[:50] + '...' if isinstance(x, str) and len(str(x)) > 50 else str(x) for x in sample]
            sample_str = ', '.join(sample)
        except:
            sample_str = "Error getting samples"
            
        # Print column info
        print(f"{col:<20} <{df[col].dtype}> {sample_str}")

