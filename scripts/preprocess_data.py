#!/usr/bin/env python
"""
Preprocessing Script for Telecom Customer Churn Data

This script performs data preprocessing steps on the telecom customer data:
1. Loads the raw data
2. Handles missing values
3. Converts non-numeric values to numeric
4. Saves the preprocessed data to files

Usage:
    python preprocess_data.py

Output:
    Preprocessed training and holdout datasets in the processed data directory
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def main():
    """Main function to preprocess the telecom churn data."""
    print("Starting data preprocessing...")
    
    # Define file paths - use absolute paths to avoid issues
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    processed_data_dir = os.path.join(base_dir, "data", "processed")
    train_file = "cell2celltrain.csv"
    holdout_file = "cell2cellholdout.csv"
    
    # Ensure processed directory exists
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Load raw data
    print(f"Loading raw data from {raw_data_dir}...")
    train_data = pd.read_csv(os.path.join(raw_data_dir, train_file))
    holdout_data = pd.read_csv(os.path.join(raw_data_dir, holdout_file))
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Holdout data shape: {holdout_data.shape}")
    
    # Categorize features
    id_columns = ['CustomerID']
    target_column = 'Churn'
    
    # Categorical columns (non-binary, non-numeric)
    categorical_columns = [
        'ServiceArea', 'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus'
    ]
    
    # Binary columns (yes/no or similar)
    binary_columns = [
        'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner',
        'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers',
        'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard',
        'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle',
        'MadeCallToRetentionTeam'
    ]
    
    # Numerical columns (continuous)
    numerical_continuous_columns = [
        'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
        'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
        'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
        'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls',
        'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls',
        'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls',
        'CallForwardingCalls', 'CallWaitingCalls', 'AgeHH1', 'AgeHH2'
    ]
    
    # Numerical columns (discrete/integer)
    numerical_discrete_columns = [
        'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'Handsets',
        'HandsetModels', 'CurrentEquipmentDays', 'RetentionCalls',
        'RetentionOffersAccepted', 'ReferralsMadeBySubscriber',
        'IncomeGroup', 'AdjustmentsToCreditRating', 'HandsetPrice'
    ]
    
    # All numerical columns
    numerical_columns = numerical_continuous_columns + numerical_discrete_columns
    
    # Check for missing values
    print("Checking for missing values...")
    missing_train = train_data.isnull().sum()
    missing_holdout = holdout_data.isnull().sum()
    
    print(f"Training data missing values: {missing_train.sum()}")
    print(f"Holdout data missing values: {missing_holdout.sum()}")
    
    # Check for non-numeric values in numerical columns
    print("Checking for non-numeric values in numerical columns...")
    for col in numerical_columns:
        non_numeric_mask = pd.to_numeric(train_data[col], errors='coerce').isna() & ~train_data[col].isna()
        non_numeric_count = non_numeric_mask.sum()
        if non_numeric_count > 0:
            print(f"Column {col} has {non_numeric_count} non-numeric values")
            print(f"Example values: {train_data.loc[non_numeric_mask, col].unique()[:5]}")
    
    # Preprocess data
    print("Preprocessing data...")
    train_data_processed = preprocess_data(train_data, numerical_columns, categorical_columns, binary_columns)
    holdout_data_processed = preprocess_data(holdout_data, numerical_columns, categorical_columns, binary_columns)
    
    # Verify no missing values remain
    train_missing = train_data_processed.isnull().sum().sum()
    holdout_missing = holdout_data_processed.isnull().sum().sum()
    
    print(f"Missing values after preprocessing (train): {train_missing}")
    print(f"Missing values after preprocessing (holdout): {holdout_missing}")
    
    if holdout_missing > 0:
        print("Warning: There are still missing values in the holdout data.")
        print("Columns with missing values:")
        missing_cols = holdout_data_processed.columns[holdout_data_processed.isnull().any()].tolist()
        for col in missing_cols:
            missing_count = holdout_data_processed[col].isnull().sum()
            print(f"  {col}: {missing_count} missing values")
        
        print("Applying additional imputation to holdout data...")
        # Apply more aggressive imputation to holdout data
        for col in missing_cols:
            if col in numerical_columns:
                # Use mean from training data for numerical columns
                mean_value = train_data_processed[col].mean()
                holdout_data_processed[col] = holdout_data_processed[col].fillna(mean_value)
            else:
                # Use mode from training data for categorical columns
                mode_value = train_data_processed[col].mode()[0]
                holdout_data_processed[col] = holdout_data_processed[col].fillna(mode_value)
        
        # Verify again
        holdout_missing_after = holdout_data_processed.isnull().sum().sum()
        print(f"Missing values after additional preprocessing (holdout): {holdout_missing_after}")
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {processed_data_dir}...")
    train_data_processed.to_csv(os.path.join(processed_data_dir, f"preprocessed_{train_file}"), index=False)
    holdout_data_processed.to_csv(os.path.join(processed_data_dir, f"preprocessed_{holdout_file}"), index=False)
    
    print("Preprocessing complete!")

def preprocess_data(data, numerical_columns, categorical_columns, binary_columns):
    """
    Preprocess the data by handling missing values and non-numeric data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to preprocess
    numerical_columns : list
        List of numerical column names
    categorical_columns : list
        List of categorical column names
    binary_columns : list
        List of binary column names
        
    Returns:
    --------
    pandas.DataFrame
        The preprocessed data
    """
    # Create a copy of the data
    processed_data = data.copy()
    
    # Convert non-numeric values to NaN in numerical columns
    for col in numerical_columns:
        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
    
    # Impute numerical features with median
    num_imputer = SimpleImputer(strategy='median')
    processed_data[numerical_columns] = num_imputer.fit_transform(processed_data[numerical_columns])
    
    # Impute categorical features with most frequent value
    for col in categorical_columns + binary_columns:
        if processed_data[col].isnull().sum() > 0:
            most_frequent = processed_data[col].mode()[0]
            # Fix the inplace warning by using a different approach
            processed_data[col] = processed_data[col].fillna(most_frequent)
    
    return processed_data

if __name__ == "__main__":
    main()