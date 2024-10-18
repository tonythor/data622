import os
import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup

# Define the file name at the top of the file
ZIP_FILENAME = 'combined_salaries.csv'

# Base URL for the salary data
BASE_URL = "https://www.ziprecruiter.com/Salaries/What-Is-the-Average-"

def extract_salary_table(url, job_title):
    """
    Extract the salary table from a given ZipRecruiter URL.
    
    Parameters:
    url (str): The URL to scrape the salary data from.
    job_title (str): The job title for which the salary data is being scraped.
    
    Returns:
    pd.DataFrame: DataFrame containing the salary data for the given job title.
    
    Raises:
    Exception: If no salary table is found on the page.
    """
    # Set up Selenium WebDriver
    driver = webdriver.Chrome()  # Make sure ChromeDriver is installed and accessible
    driver.get(url)
    
    # Allow time for the page to load
    time.sleep(5)
    
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table')

    if not table:
        driver.quit()
        raise Exception(f"No table found on the page: {url}")
    
    # Extract table headers and rows
    headers = [header.text.strip() for header in table.find_all('th')]
    rows = [
        [col.text.strip() for col in row.find_all('td')]
        for row in table.find_all('tr')[1:]  # Skip header row
    ]

    # Create DataFrame and add a 'Job Title' column
    df = pd.DataFrame(rows, columns=headers)
    df['Job Title'] = job_title
    
    # Close the WebDriver
    driver.quit()
    return df

def add_state_abbreviation(df):
    """
    Add a 'State Abbreviation' column by merging with state abbreviations data.
    
    Parameters:
    df (pd.DataFrame): The salary DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with state abbreviations added.
    """
    # Dictionary of state names and abbreviations
    state_abbreviations = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 
        'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 
        'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    
    # Convert the dictionary to a DataFrame
    state_df = pd.DataFrame(list(state_abbreviations.items()), columns=['State', 'Abbreviation'])
    
    # Merge the state abbreviations with the salary data
    df = df.merge(state_df, how='left', on='State')
    
    return df

def add_salary_tier_column(df):
    """
    Add a 'Salary Tier' column to the DataFrame based on the Annual Salary.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with salary data to add the tier column.
    
    Returns:
    pd.DataFrame: The DataFrame with the added 'Salary Tier' column.
    """
    # Define salary tiers based on annual salary
    salary_bins = [0, 60000, 80000, 100000, 110000, 120000, 140000, 160000, float('inf')]
    salary_labels = ['<60K', '60K-80K', '80K-100K', '100K-110K', '110K-120K', '120K-140K', '140K-160K', '>160K']

    # Create a new column 'Salary Tier' based on the bins
    df['Salary Tier'] = pd.cut(df['Annual Salary'], bins=salary_bins, labels=salary_labels, right=False)
    
    return df

def clean_salary_data(df):
    """
    Clean the salary columns by removing non-numeric characters 
    and converting them to integers (for salaries and pay) or floats (for hourly wage).
    
    Parameters:
    df (pd.DataFrame): The DataFrame with salary data to be cleaned.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # Clean Annual Salary, Monthly Pay, Weekly Pay (remove commas, dollar signs, and convert to int)
    salary_columns = ['Annual Salary', 'Monthly Pay', 'Weekly Pay']
    
    for col in salary_columns:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True).astype(int)
    
    # Clean Hourly Wage (remove dollar signs, convert to float)
    df['Hourly Wage'] = df['Hourly Wage'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    
    return df

def load_zip(data_dir: str):
    """
    Load the CSV file from the specified data directory if it exists; otherwise, raise an error.
    
    Parameters:
    data_dir (str): The directory where the data files are stored.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    
    Raises:
    FileNotFoundError: If the file does not exist.
    """
    zip_filename = os.path.join(data_dir, ZIP_FILENAME)
    if os.path.exists(zip_filename):
        # print(f"Loading data from {zip_filename}")
        return pd.read_csv(zip_filename)
    else:
        raise FileNotFoundError(f"The file '{zip_filename}' does not exist.")

def process_job_urls(job_urls):
    """
    Process a list of job URLs to extract salary data for multiple job titles.
    
    Parameters:
    job_urls (list): List of tuples containing (URL suffix, job title).
    
    Returns:
    pd.DataFrame: Combined DataFrame of all extracted data.
    """
    dfs = []
    for url_suffix, job_title in job_urls:
        full_url = BASE_URL + url_suffix
        try:
            df = extract_salary_table(full_url, job_title)
            dfs.append(df)
            print(f"Successfully extracted data for {job_title}")
        except Exception as e:
            print(f"Failed to extract data for {job_title}: {e}")
    
    if dfs:
        # Concatenate all DataFrames into one combined DataFrame
        return pd.concat(dfs, ignore_index=True)
    else:
        print("No data was extracted.")

def build_zip(data_dir: str):
    """
    Build the salary data file by downloading data from the provided job URLs,
    cleaning it, adding the state abbreviations and salary tiers, 
    and saving it to a CSV file in the specified data directory.
    
    Parameters:
    data_dir (str): The directory where the data files should be saved.
    """
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    zip_filename = os.path.join(data_dir, 'combined_salaries.csv')

    # List of job URL suffixes and corresponding job titles
    job_urls = [
        ("DATA-Scientist-Salary-by-State", "Data Scientist"),
        ("Data-Engineer-Salary-by-State", "Data Engineer"),
        ("Data-Analyst-Salary-by-State", "Data Analyst"),
        ("Machine-Learning-Engineer-Salary-by-State", "Machine Learning Engineer"),
        ("Quantitative-Analyst-Salary-by-State", "Quantitative Analyst"),
        ("BIG-DATA-Engineer-Salary-by-State", "Big Data Engineer"),
        ("Statistician-Salary-by-State", "Statistician")
    ]
    
    # Process the job URLs and extract salary data
    df_combined = process_job_urls(job_urls)
    
    # Clean the salary data and add the new columns
    if not df_combined.empty:
        # Clean the salary columns (Annual Salary, Monthly Pay, Weekly Pay, Hourly Wage)
        df_combined = clean_salary_data(df_combined)

        # Add the state abbreviation column
        df_combined = add_state_abbreviation(df_combined)
        
        # Add the salary tier column
        df_combined = add_salary_tier_column(df_combined)
        
        # Save the DataFrame to CSV
        df_combined.to_csv(zip_filename, index=False)
        print(f"Combined data with state abbreviations and salary tiers saved to {zip_filename}")
    else:
        print("No data to save.")
