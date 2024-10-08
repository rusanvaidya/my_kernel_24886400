def pop_target(df, target_col):
    """
    Extract target variable from the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    target_col : str
        Name of the target variable

    Returns
    -------
    pd.Dataframe
        subsetted pandas dataframe containing all features
    pd.Dataframe
        subsetted pandas dataframe containing target
    """
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    return df_copy, target

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)

def load_sets(path='../data/processed/'):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  , allow_pickle=True) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' , allow_pickle=True) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test

def subset_x_y(target, features, start_index:int, end_index:int):
    """Keep only the rows for X and y (optional) sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    features : int
        Index of the starting observation
    features : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """

    return features[start_index:end_index], target[start_index:end_index]

def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(df_copy) / 5)

    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(df_copy))

    return X_train, y_train, X_val, y_val, X_test, 

def data_cleaning(df, num_columns=None, cat_columns=None, 
                              num_impute_strategy='mean', cat_impute_strategy='most_frequent', 
                              special_column_transformations=None):
    """
    Generalized function to clean data, handle missing values, and standardize specific columns.

    Parameters
    ----------
    df: pd.DataFrame 
        The input dataframe to clean.
    num_columns: list
        List of numerical columns to impute. If None, all numerical columns will be considered.
    cat_columns: list
        List of categorical columns to impute. If None, all categorical columns will be considered.
    num_impute_strategy: str
        Strategy to impute numerical columns. Default is 'mean'.
    cat_impute_strategy: str
        Strategy to impute categorical columns. Default is 'most_frequent'.
    special_column_transformations: dict
        Dictionary where key is column name and value is the transformation function.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Handling missing values for numerical columns
    if num_columns is None:
        num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    numerical_imputer = SimpleImputer(strategy=num_impute_strategy)
    df[num_columns] = numerical_imputer.fit_transform(df[num_columns])
    
    # Handling missing values for categorical columns
    if cat_columns is None:
        cat_columns = df.select_dtypes(include=['object']).columns
    
    categorical_imputer = SimpleImputer(strategy=cat_impute_strategy)
    df[cat_columns] = categorical_imputer.fit_transform(df[cat_columns])
    
    # Applying any special transformations passed in the special_column_transformations dictionary
    if special_column_transformations:
        for col, func in special_column_transformations.items():
            df[col] = df[col].apply(lambda x: func(x) if pd.notna(x) else np.nan)

    # Final fallback for any remaining missing values
    remaining_num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    remaining_cat_columns = df.select_dtypes(include=['object']).columns
    
    df[remaining_num_columns] = numerical_imputer.fit_transform(df[remaining_num_columns])
    df[remaining_cat_columns] = categorical_imputer.fit_transform(df[remaining_cat_columns])
    
    return df

def get_data_from_gdrive(url, file_path):
    """
    Downloads a CSV file directly from Google Drive if it doesn't exist locally,
    loads it into a DataFrame, and removes the CSV file after loading it.

    Parameters:
    url (str): Google Drive URL of the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the contents of the CSV file.
    """
    import gdown
    import os
    import pandas as pd

    # Extract file ID from the URL
    file_id = url.split('/')[5]
    
    # Use the file ID as part of the filename to avoid filename argument
    filename = f"{file_path}/{file_id}.csv"
    
    # Check if the file already exists locally
    if os.path.exists(filename):
        print(f"{filename} already exists. Reading the file from the local directory.")
        try:
            df = pd.read_csv(filename)
            return df
        except pd.errors.EmptyDataError:
            print("Failed to load: the CSV file is empty or corrupt.")
            return None
        except pd.errors.ParserError:
            print("Failed to load: the CSV file is improperly formatted.")
            return None
    else:
        print(f"{filename} does not exist. Downloading the file from Google Drive.")
        
        # Google Drive URL for direct download
        file_url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download the file using gdown
        try:
            output = gdown.download(file_url, filename, quiet=True)
        except Exception as e:
            print(f"Failed to download the file. Error: {e}")
            return None
        
        # Load the downloaded file into a DataFrame
        if output:
            try:
                df = pd.read_csv(output)
                print('Data fetch completed!')
                return df
            except pd.errors.EmptyDataError:
                print("Failed to load: the CSV file is empty or corrupt.")
                return None
            except pd.errors.ParserError:
                print("Failed to load: the CSV file is improperly formatted.")
                return None
        else:
            print("File download failed.")
            return None


def merge_dataframes_on_common_columns(df_list, how='left'):
    """
    This function merges a list of DataFrames on their common column names.
    
    Args:
    df_list (list): List of DataFrames to merge.
    
    Returns:
    pd.DataFrame: A single merged DataFrame.
    """
    from functools import reduce
    import pandas as pd

    # Function to find common columns between two dataframes
    def merge_two_dfs_on_common_columns(df1, df2):
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        print(common_cols)
        if not common_cols:
            raise ValueError("No common columns to merge on.")
        return pd.merge(df1, df2, on=common_cols, how=how)
    
    # Reduce the list of dataframes by merging them one by one
    merged_df = reduce(merge_two_dfs_on_common_columns, df_list)
    
    return merged_df

def clean_data(data):
    """
    General function to clean data.
    Removes duplicates, handles missing values, and resets index.
    """
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle missing values (customize as needed)
    data = data.dropna()
    
    # Reset index
    data = data.reset_index(drop=True)
    
    return data


def remove_outliers(df, column):
    """
    Removes outliers from a specific column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column from which to remove outliers.
    
    Returns:
    pd.DataFrame: A DataFrame with outliers removed from the specified column.
    """
    
    Q1 = df[column].quantile(0.25)  # 25th percentile (lower quartile)
    Q3 = df[column].quantile(0.75)  # 75th percentile (upper quartile)
    IQR = Q3 - Q1  # Interquartile Range
    
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
    
    # Filter the dataframe to include only non-outliers
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_cleaned
