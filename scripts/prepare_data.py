"""
Prepare data for the telecom churn prediction project.

This script splits the preprocessed data files into features (X) and target (y) files,
and also creates training, validation, and holdout sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os

# Define the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data', 'processed')
models_dir = os.path.join(base_dir, 'models')

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed data...")
train_data = pd.read_csv(os.path.join(data_dir, 'preprocessed_cell2celltrain.csv'))
holdout_data = pd.read_csv(os.path.join(data_dir, 'preprocessed_cell2cellholdout.csv'))

# Check if the holdout data has the target variable
has_holdout_target = 'Churn' in holdout_data.columns
print(f"Holdout data has target variable: {has_holdout_target}")

# Convert Churn to binary (Yes/No to 1/0)
if 'Churn' in train_data.columns:
    train_data['Churn'] = train_data['Churn'].map({'Yes': 1, 'No': 0})
    
if has_holdout_target and 'Churn' in holdout_data.columns:
    holdout_data['Churn'] = holdout_data['Churn'].map({'Yes': 1, 'No': 0})

# Split training data into features and target
print("Splitting data into features and target...")
X_train_val = train_data.drop('Churn', axis=1)
y_train_val = train_data['Churn']

# Split holdout data into features and target if available
if has_holdout_target:
    X_holdout = holdout_data.drop('Churn', axis=1)
    y_holdout = holdout_data['Churn']
else:
    X_holdout = holdout_data
    y_holdout = None

# Identify categorical columns
categorical_cols = X_train_val.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Handle categorical variables
print("Encoding categorical variables...")

# For binary categorical variables (Yes/No), use simple mapping
binary_cols = []
for col in categorical_cols:
    unique_vals = X_train_val[col].unique()
    if len(unique_vals) == 2 and set(unique_vals) == {'Yes', 'No'}:
        binary_cols.append(col)
        X_train_val[col] = X_train_val[col].map({'Yes': 1, 'No': 0})
        X_holdout[col] = X_holdout[col].map({'Yes': 1, 'No': 0})

# Remove binary columns from categorical columns list
categorical_cols = [col for col in categorical_cols if col not in binary_cols]

# For other categorical variables, use one-hot encoding
if categorical_cols:
    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform the training data
    encoded_cats_train = encoder.fit_transform(X_train_val[categorical_cols])
    
    # Transform the holdout data
    encoded_cats_holdout = encoder.transform(X_holdout[categorical_cols])
    
    # Get the feature names
    feature_names = encoder.get_feature_names_out(categorical_cols)
    
    # Create DataFrames with the encoded variables
    encoded_df_train = pd.DataFrame(encoded_cats_train, columns=feature_names, index=X_train_val.index)
    encoded_df_holdout = pd.DataFrame(encoded_cats_holdout, columns=feature_names, index=X_holdout.index)
    
    # Drop the original categorical columns and add the encoded ones
    X_train_val = X_train_val.drop(categorical_cols, axis=1).join(encoded_df_train)
    X_holdout = X_holdout.drop(categorical_cols, axis=1).join(encoded_df_holdout)
    
    # Save the encoder for future use
    import joblib
    joblib.dump(encoder, os.path.join(models_dir, 'categorical_encoder.joblib'))
    print(f"Categorical encoder saved to: {os.path.join(models_dir, 'categorical_encoder.joblib')}")

# Split training data into training and validation sets
print("Splitting training data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# Save the data
print("Saving data files...")
X_train.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
X_val.to_csv(os.path.join(data_dir, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(data_dir, 'y_val.csv'), index=False)
X_holdout.to_csv(os.path.join(data_dir, 'X_holdout.csv'), index=False)

if y_holdout is not None:
    y_holdout.to_csv(os.path.join(data_dir, 'y_holdout.csv'), index=False)
else:
    # Create an empty y_holdout file
    pd.Series(dtype='int64').to_csv(os.path.join(data_dir, 'y_holdout.csv'), index=False)

print("Data preparation complete!")
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Holdout set: {X_holdout.shape}")