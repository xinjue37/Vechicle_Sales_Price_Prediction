import warnings
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump, load
from category_encoders import TargetEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import FeatureHasher

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# 1. Function & Class for One-hot encoding method
## Get top categories in training data
def get_top_categories(df, cat_cols, max_categories=10, file_path='data/top_categories.json'):
    ### Save top_categories_dict to JSON file
    def save_top_categories(top_categories_dict, file_path='data/top_categories.json'):
        with open(file_path, 'w') as f:
            json.dump(top_categories_dict, f)

    ### Load top_categories_dict from JSON file
    def load_top_categories(file_path='data/top_categories.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
        
    if os.path.exists(file_path):
        top_categories_dict = load_top_categories(file_path)
    else:
        top_categories_dict = {}
        for col in cat_cols:
            value_counts = df[col].value_counts()
            num_unique_values = len(value_counts)

            num_category_retain = max_categories - 1 if num_unique_values > max_categories else num_unique_values - 1
            top_categories = value_counts.head(num_category_retain).index.tolist()

            top_categories_dict[col] = top_categories
        save_top_categories(top_categories_dict, file_path) # Save the dictionary to JSON file
    return top_categories_dict

## Function to map categories using the provided top categories
def apply_top_categories(df, top_categories_dict):
    for col, top_categories in top_categories_dict.items():
        df[col] = df[col].fillna('Missing')  # Fix NaN
        df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')

    # Verify the changes
    for col in top_categories_dict.keys():
        if df[col].nunique() > len(top_categories_dict[col]) + 1:
            raise Exception(f"Column '{col}' has too many unique values after applying top categories.")
    return df

## Function to get OneHotEncoder
def get_one_hot_encoder(X_train=None, cat_cols=None):
    if os.path.exists('data/onehot_encoder.joblib'):
        encoder = load('data/onehot_encoder.joblib')
    elif X_train is None or cat_cols is None:
        raise ValueError("X_train and cat_cols must be provided if data/onehot_encoder.joblib does not exist")
    else:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoder.fit(X_train[cat_cols])
        dump(encoder, 'data/onehot_encoder.joblib')  # Save to file

    return encoder

## Function to transform data based on OneHotEncoder
def transform_one_hot_encoder(X_data, cat_cols):
    encoder = get_one_hot_encoder()
    X_data_encoded = pd.DataFrame(
        encoder.transform(X_data[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_data.index
    )
    return X_data_encoded

## Class for One Hot Encoding Preprocessing Inference Pipeline
class OneHotInferencePipeline:
    def __init__(self, top_categories_path='data/top_categories.json', 
                 encoder_path='data/onehot_encoder.joblib',
                 num_cols=None, cat_cols=None):
        """
        Initialize the pipeline with pre-saved artifacts and column lists.
        
        Args:
            top_categories_path (str): Path to JSON file with top categories.
            encoder_path (str): Path to saved OneHotEncoder.
            num_cols (list): Numerical columns used during training.
            cat_cols (list): Categorical columns used during training.
        """
        # Load top categories mapping
        with open(top_categories_path, 'r') as f:
            self.top_categories_dict = json.load(f)
        
        # Load pre-fitted OneHotEncoder
        self.encoder = load(encoder_path)
        
        # Store column lists
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        # Precompute expected feature names after sanitization
        # 1. Get original column order (numerical first, then one-hot encoded)
        original_features = self.num_cols + self.encoder.get_feature_names_out(self.cat_cols).tolist()
        # 2. Apply sanitization to match training
        self.feature_names = [self._sanitize(col) for col in original_features]

    def _sanitize(self, col_name):
        """Replicate the column name sanitization from training."""
        return str(col_name).replace('[', '_').replace(']', '_').replace('<', '_')

    def transform(self, raw_data):
        """
        Process raw data through top-category mapping, one-hot encoding, and feature alignment.
        
        Args:
            raw_data (pd.DataFrame): Raw input data with original columns.
            
        Returns:
            pd.DataFrame: Processed data ready for model prediction.
        """
        # Validate input structure
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)
        
        # Drop the columns that are not needed (only if they exist in the DataFrame)
        with open('data/COL_TO_REMOVE.json', 'r') as file:
            COL_TO_REMOVE = json.load(file)

        cols_to_drop = [col for col in COL_TO_REMOVE if col in raw_data.columns]
        raw_data = raw_data.drop(columns=cols_to_drop)

        # Process the date information
        if 'Sold_Date' in raw_data.columns:
            raw_data['Sold_Date'] = pd.to_datetime(raw_data['Sold_Date'], errors='coerce').copy()
            # Extract datetime features
            raw_data['Sold_Year'] = raw_data['Sold_Date'].dt.year
            raw_data['Sold_Month'] = raw_data['Sold_Date'].dt.month
            raw_data['Sold_Day'] = raw_data['Sold_Date'].dt.day
            raw_data['Sold_DayOfWeek'] = raw_data['Sold_Date'].dt.dayofweek
            # Drop original Sold_Date
            raw_data.drop('Sold_Date', axis=1, inplace=True)
            
        # Ensure required columns exist
        missing_cols = set(self.cat_cols + self.num_cols) - set(raw_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        processed = raw_data.copy()
        
        # Apply top-category mapping to categorical columns
        processed = apply_top_categories(processed, self.top_categories_dict)
        
        # One-hot encode categorical features
        cat_encoded = transform_one_hot_encoder(processed, self.cat_cols)
        
        # Combine with numerical features
        combined = pd.concat([processed[self.num_cols], cat_encoded], axis=1)
        
        # Sanitize column names (critical for XGBoost compatibility)
        combined.columns = [self._sanitize(col) for col in combined.columns]
        
        # Enforce column order seen during training
        return combined.reindex(columns=self.feature_names)

# 2. Class for Target Encoding Preprocessing Inference Pipeline
class Target_encoding_InferencePipeline:
    def __init__(self, target_encoder, cat_cols, num_cols):
        """
        Initialize the inference pipeline with fitted components and column info.
        
        Args:
            target_encoder (TargetEncoder): Pre-fitted TargetEncoder.
            cat_cols (list): Categorical columns used during training.
            num_cols (list): Numerical columns used during training.
        """
        self.target_encoder = target_encoder
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        # Store the expected feature order (cat_cols followed by num_cols)
        self.feature_names = self.cat_cols + self.num_cols

    def transform(self, raw_data):
        """
        Process raw data by encoding categorical features and combining with numerical features.
        
        Args:
            raw_data (pd.DataFrame): Raw input data to process.
            
        Returns:
            pd.DataFrame: Processed data with encoded features.
        """
        # Ensure input is a DataFrame to handle column selection
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)
        
        # Drop the columns that are not needed (only if they exist in the DataFrame)
        with open('data/COL_TO_REMOVE.json', 'r') as file:
            COL_TO_REMOVE = json.load(file)
        cols_to_drop = [col for col in COL_TO_REMOVE if col in raw_data.columns]
        raw_data = raw_data.drop(columns=cols_to_drop)

        # Process the date information
        if 'Sold_Date' in raw_data.columns:
            raw_data['Sold_Date'] = pd.to_datetime(raw_data['Sold_Date'], errors='coerce').copy()
            # Extract datetime features
            raw_data['Sold_Year'] = raw_data['Sold_Date'].dt.year
            raw_data['Sold_Month'] = raw_data['Sold_Date'].dt.month
            raw_data['Sold_Day'] = raw_data['Sold_Date'].dt.day
            raw_data['Sold_DayOfWeek'] = raw_data['Sold_Date'].dt.dayofweek
            # Drop original Sold_Date
            raw_data.drop('Sold_Date', axis=1, inplace=True)

        # Extract and encode categorical columns
        raw_data_cat = raw_data[self.cat_cols]
        raw_data_cat_encoded = self.target_encoder.transform(raw_data_cat)
        
        # Extract numerical columns
        raw_data_num = raw_data[self.num_cols]
        
        # Concatenate encoded categoricals and numericals
        processed_data = pd.concat([raw_data_cat_encoded, raw_data_num], axis=1)
        
        # Ensure columns are in the correct order (matches training data)
        processed_data = processed_data.reindex(columns=self.feature_names)
        
        return processed_data
    
# 3. Class for Feature Hashing Preprocessing Inference Pipeline
class Feature_Hashing_InferencePipeline:
    def __init__(self, hasher, cat_cols, num_cols):
        """
        Initialize the pipeline with a configured FeatureHasher and column info.
        
        Args:
            hasher (FeatureHasher): Pre-configured FeatureHasher (with `n_features` and `input_type`).
            cat_cols (list): Categorical columns used during training.
            num_cols (list): Numerical columns used during training.
        """
        self.hasher = hasher
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        # Store expected output feature names (hashed columns followed by numerical columns)
        self.feature_names = (
            [f"hashed_{i}" for i in range(self.hasher.n_features)] 
            + self.num_cols
        )

    def transform(self, raw_data):
        """
        Process raw data by hashing categorical features and combining with numerical features.
        
        Args:
            raw_data (pd.DataFrame): Raw input data to process.
            
        Returns:
            pd.DataFrame: Processed data with hashed features and numerical columns.
        """
        # Ensure input is a DataFrame
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)
        
        # Drop the columns that are not needed
        with open('data/COL_TO_REMOVE.json', 'r') as file:
            COL_TO_REMOVE = json.load(file)
        cols_to_drop = [col for col in COL_TO_REMOVE if col in raw_data.columns]
        raw_data = raw_data.drop(columns=cols_to_drop)

        # Process the date information
        if 'Sold_Date' in raw_data.columns:
            raw_data['Sold_Date'] = pd.to_datetime(raw_data['Sold_Date'], errors='coerce').copy()
            # Extract datetime features
            raw_data['Sold_Year'] = raw_data['Sold_Date'].dt.year
            raw_data['Sold_Month'] = raw_data['Sold_Date'].dt.month
            raw_data['Sold_Day'] = raw_data['Sold_Date'].dt.day
            raw_data['Sold_DayOfWeek'] = raw_data['Sold_Date'].dt.dayofweek
            # Drop original Sold_Date
            raw_data.drop('Sold_Date', axis=1, inplace=True)

        # Process categorical columns
        raw_data_cat = raw_data[self.cat_cols].astype(str)
        hashed_features = self.hasher.transform(raw_data_cat.values)
        hashed_df = pd.DataFrame(
            hashed_features.toarray(),
            columns=[f"hashed_{i}" for i in range(self.hasher.n_features)]
        )

        # Process numerical columns and reset index for alignment
        raw_data_num = raw_data[self.num_cols].reset_index(drop=True)

        # Combine features and enforce column order
        processed_data = pd.concat([hashed_df, raw_data_num], axis=1)
        processed_data = processed_data[self.feature_names]  # Ensure column order matches training

        return processed_data
    
# 4. Function for Regression Model building and training
## Function to tune and save the model
def tune_and_save_model(name, model, params, X_train, y_train, X_test, n_iter=10, save_dir="models"):
    print(f"> > > Tuning {name}...")
    search = RandomizedSearchCV(estimator=model,
                                param_distributions=params,
                                n_iter=n_iter,
                                cv=3,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1,
                                random_state=0)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    dump(best_model, os.path.join(save_dir, f"{name}_best_model.pkl"))

    # Predict on test
    y_pred = best_model.predict(X_test)
    return best_model, y_pred


# Compute the evaluation metrics 
def evaluate_model(y_test, y_pred, category_to_slice, top_k=5):
    category_metrics = {}  # Initialize a dictionary to store metrics by category
    
    # Loop through each unique 'make_category' in X_test
    for make_category in category_to_slice.value_counts()[:top_k].index.tolist() + ["Overall"]:
        # Filter the data based on 'Make' category
        if make_category == "Overall":
            y_test_slice = y_test
            y_pred_slice = y_pred
        else:
            masked_idx   = category_to_slice == make_category
            y_test_slice = y_test[masked_idx]
            y_pred_slice = y_pred[masked_idx]
        
        # Calculate the evaluation metrics
        r2 = r2_score(y_test_slice, y_pred_slice)
        rmse = np.sqrt(mean_squared_error(y_test_slice, y_pred_slice))
        mae = mean_absolute_error(y_test_slice, y_pred_slice)
        
        # Store metrics for each category (Make)
        category_metrics[make_category] = {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }
    
    return category_metrics  # Return the metrics for each category





