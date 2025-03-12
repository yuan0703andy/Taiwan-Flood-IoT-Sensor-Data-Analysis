import os
import zipfile
import pandas as pd
import glob
from datetime import datetime
import re
import concurrent.futures
import time
# Define the absolute paths to ensure correct file handling
base_path = os.path.normpath(os.path.join('..', 'dataset', 'rain_station'))

print(f"Base data path: {base_path}")

# Load the station reference data
rain_station_path = os.path.join(base_path, 'rain_station.csv')
print(f"Looking for rain_station.csv at: {rain_station_path}")

if os.path.exists(rain_station_path):
    print(f"rain_station.csv found at {rain_station_path}")
    rain_station_df = pd.read_csv(rain_station_path)
else:
    print(f"Error: rain_station.csv not found at {rain_station_path}")
    raise FileNotFoundError(f"Could not find rain_station.csv at {rain_station_path}")

# Function to process data for a specific year
def process_year_data(year):
    if int(year) >= 2019 and int(year) <= 2023:
        return process_2019_2023_data(year)
    elif int(year) >= 2024 and int(year) <= 2025:
        return process_2024_2025_data(year)
    else:
        raise ValueError(f"Year {year} is not supported")

def extract_date_from_filename(filename):
    # 移除 "rain_" 前綴
    date_part = filename.replace('rain_', '').replace('.zip', '')
    
    # 判斷日期部分的長度
    if len(date_part) == 7:  # 例如 2024011 (YYYY-M-D)
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:])
    elif len(date_part) == 8:  # 例如 20240110 (YYYY-MM-DD)
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:])
    else:
        # 其他可能的格式
        match = re.search(r'(\d{4})(\d{1,2})(\d{1,2})', date_part)
        if not match:
            raise ValueError(f"Cannot parse date from filename: {filename}")
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
    
    return datetime(year, month, day)

# Function to process 2019-2023 format data
def process_2019_2023_data(year):
    # Find all month folders for the year
    year_path = os.path.join(base_path, year)
    month_folders = [f for f in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, f))]
    
    all_data = []
    
    for month_folder in month_folders:
        month_path = os.path.join(year_path, month_folder)
        zip_files = glob.glob(os.path.join(month_path, '*.zip'))
        
        for zip_file in zip_files:
            file_name = os.path.basename(zip_file)
            try:
                date_obj = extract_date_from_filename(file_name)
                # 繼續處理...
            except ValueError as e:
                print(f"Error with file {file_name}: {e}")
                continue
                
            # Process the zip file
            with zipfile.ZipFile(zip_file, 'r') as z:
                for csv_file in z.namelist():
                    if csv_file.endswith('.csv'):
                        with z.open(csv_file) as f:
                            try:
                                # Read CSV
                                df = pd.read_csv(f, encoding='utf-8')
                                
                                # Keep only required columns
                                if 'station_id' in df.columns and 'HOUR_24' in df.columns:
                                    df_filtered = df[['station_id', 'HOUR_24']]
                                    
                                    # Skip if empty or all NAs
                                    if df_filtered.empty or df_filtered.isna().all().all():
                                        continue
                                    
                                    # Rename columns to match
                                    df_filtered = df_filtered.rename(columns={
                                        'station_id': 'StationId',
                                        'HOUR_24': 'Past24hr'
                                    })
                                    
                                    # Add datetime column
                                    df_filtered['datetime'] = date_obj
                                    
                                    # Merge with station information to get CountyName and TownName
                                    df_with_location = pd.merge(
                                        df_filtered,
                                        rain_station_df[['StationId', 'CountyName', 'TownName']],
                                        on='StationId',
                                        how='left'
                                    )
                                    
                                    # Skip if result is empty
                                    if not df_with_location.empty:
                                        all_data.append(df_with_location)
                            except Exception as e:
                                print(f"Error processing {csv_file} in {zip_file}: {str(e)}")
    
    # Combine all data into a single DataFrame
    if not all_data:
        return pd.DataFrame()
    
    # Filter out empty DataFrames before concatenation
    non_empty_data = [df for df in all_data if not df.empty and not df.isna().all().all()]
    
    if not non_empty_data:
        return pd.DataFrame()
        
    # Ensure all DataFrames have consistent dtypes before concatenation
    for i in range(len(non_empty_data)):
        for col in non_empty_data[i].columns:
            if col == 'StationId':
                non_empty_data[i][col] = non_empty_data[i][col].astype(str)
            elif col == 'Past24hr':
                non_empty_data[i][col] = non_empty_data[i][col].astype(float)
    
    year_df = pd.concat(non_empty_data, ignore_index=True)
    
    # Handle duplicate StationId for the same day by taking mean
    year_df = year_df.groupby(['StationId', 'datetime', 'CountyName', 'TownName'], as_index=False)['Past24hr'].mean()
    
    # Replace negative values with NA
    year_df['Past24hr'] = year_df['Past24hr'].apply(lambda x: pd.NA if x < 0 else x)
    
    # Sort by datetime
    year_df = year_df.sort_values(by='datetime')
    
    return year_df

# Function to process 2024-2025 format data
def process_2024_2025_data(year):
    # Find all month folders for the year
    year_path = os.path.join(base_path, year)
    month_folders = [f for f in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, f))]
    
    all_data = []
    
    for month_folder in month_folders:
        month_path = os.path.join(year_path, month_folder)
        zip_files = glob.glob(os.path.join(month_path, '*.zip'))
        
        for zip_file in zip_files:
            file_name = os.path.basename(zip_file)
            try:
                date_obj = extract_date_from_filename(file_name)
                # 繼續處理...
            except ValueError as e:
                print(f"Error with file {file_name}: {e}")
                continue
                
            # Process the zip file
            with zipfile.ZipFile(zip_file, 'r') as z:
                for csv_file in z.namelist():
                    if csv_file.endswith('.csv'):
                        with z.open(csv_file) as f:
                            try:
                                # Read CSV
                                df = pd.read_csv(f, encoding='utf-8')
                                
                                # Keep only required columns
                                required_columns = ['StationId', 'StationName', 'CountyName', 'TownName', 'Past24hr']
                                if all(col in df.columns for col in required_columns):
                                    df_filtered = df[required_columns].copy()
                                    
                                    # Skip if empty or all NAs
                                    if df_filtered.empty or df_filtered.isna().all().all():
                                        continue
                                    
                                    # Add datetime column
                                    df_filtered.loc[:, 'datetime'] = date_obj
                                    
                                    all_data.append(df_filtered)
                            except Exception as e:
                                print(f"Error processing {csv_file} in {zip_file}: {str(e)}")

    # Combine all data into a single DataFrame
    if not all_data:
        return pd.DataFrame()
    
    # Filter out empty DataFrames before concatenation
    non_empty_data = [df for df in all_data if not df.empty and not df.isna().all().all()]
    
    if not non_empty_data:
        return pd.DataFrame()
    
    # Ensure all DataFrames have consistent dtypes before concatenation
    for i in range(len(non_empty_data)):
        for col in non_empty_data[i].columns:
            if col == 'StationId':
                non_empty_data[i][col] = non_empty_data[i][col].astype(str)
            elif col == 'StationName' or col == 'CountyName' or col == 'TownName':
                non_empty_data[i][col] = non_empty_data[i][col].astype(str)
            elif col == 'Past24hr':
                non_empty_data[i][col] = non_empty_data[i][col].astype(float)
    
    year_df = pd.concat(non_empty_data, ignore_index=True)
    
    # Handle duplicate StationId for the same day by taking mean
    year_df = year_df.groupby(['StationId', 'StationName', 'datetime', 'CountyName', 'TownName'], as_index=False)['Past24hr'].mean()
    
    # Replace negative values with NA
    year_df['Past24hr'] = year_df['Past24hr'].apply(lambda x: pd.NA if x < 0 else x)
    
    # Sort by datetime
    year_df = year_df.sort_values(by='datetime')
    
    return year_df

# Function to process a single year and save the result
def process_and_save_year(year):
    start_time = time.time()
    print(f"Started processing year {year}...")
    try:
        year_path = os.path.join(base_path, year)
        if not os.path.exists(year_path):
            print(f"Directory for year {year} not found at {year_path}")
            return None
            
        df = process_year_data(year)
        if not df.empty:
            # Make sure the data is sorted by datetime
            df = df.sort_values(by='datetime')
            
            print(f"Year {year} processed successfully with {len(df)} records.")
            
            # Save each year's dataframe immediately after processing
            output_path = os.path.join(str(base_path), f'rain_station_data_{year}_merged.csv')
            print(f"Saving data for {year} to: {output_path}")
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            # Verify the file was saved
            if os.path.exists(output_path):
                elapsed_time = time.time() - start_time
                print(f"Successfully saved processed data for {year} to {output_path} in {elapsed_time:.2f} seconds")
                return year, df
            else:
                print(f"Warning: Failed to save file for {year} to {output_path}")
                return None
        else:
            print(f"No data found for year {year}.")
            return None
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")
        return None

# Main function to process all years in parallel
def main():
    years = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']
    annual_dataframes = {}
    
    # Use ThreadPoolExecutor for parallel processing
    # Since most of the time is spent on I/O operations (reading files),
    # threads are more efficient than processes
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all years for processing
        future_to_year = {executor.submit(process_and_save_year, year): year for year in years}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_year):
            year = future_to_year[future]
            result = future.result()
            if result:
                processed_year, df = result
                annual_dataframes[processed_year] = df
    
    
if __name__ == "__main__":
    main()