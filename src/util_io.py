import os
import pandas as pd

def _read_csv(path):
    """
    reads csv file into dataframe while preserving index order
    """
    if not os.path.exists(path):                                        # sanity check that file exists
        raise FileNotFoundError(f"File not found --   {path}")
    
    df = pd.read_csv(path, index_col=0)
    return df

def load_frames(data_dir = "data", imp = False):                                     # data_dir is path to folder with csv files
    """
    loads X_train, y_train, X_test from data_dir
    returns (X_train, y_train, X_test)
    -- indices are preserved and aligned
    """
    if imp:
        X_train = _read_csv(os.path.join(data_dir, "X_train_imp.csv"))
        X_test  = _read_csv(os.path.join(data_dir, "X_test_imp.csv"))
    else:
        X_train = _read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test  = _read_csv(os.path.join(data_dir, "X_test.csv"))

    y_train_df = _read_csv(os.path.join(data_dir, "y_train.csv"))
    y_train = y_train_df.iloc[:, 0]                                     # y_train has a single data column so use it as Series type
    y_train = y_train.astype("int8")                                    # convert to int8 to save memory and avoid conversions inside ML libraries

    # print(f"Loaded -- X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}")
    return X_train, y_train, X_test

 
# if __name__ == "__main__":
#     X_train, y_train, X_test = load_frames(data_dir="data")