"""
Utility functions for the telecom churn prediction project.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def align_features(train_df, test_df):
    """
    Align features between training and test/holdout datasets.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe with the reference feature set
    test_df : pd.DataFrame
        Test/holdout dataframe to be aligned with training features
        
    Returns:
    --------
    pd.DataFrame
        Test dataframe with aligned features (same columns as train_df)
    """
    # Get the list of features from both dataframes
    train_features = set(train_df.columns)
    test_features = set(test_df.columns)
    
    # Find features in train but not in test
    missing_in_test = train_features - test_features
    
    # Find features in test but not in train
    extra_in_test = test_features - train_features
    
    # Log the differences
    if missing_in_test:
        logger.warning(f"Features in training but missing in test: {missing_in_test}")
        logger.warning("These will be filled with zeros in the test set")
    
    if extra_in_test:
        logger.warning(f"Features in test but not in training: {extra_in_test}")
        logger.warning("These will be dropped from the test set")
    
    # Create a copy of the test dataframe
    aligned_test_df = test_df.copy()
    
    # Add missing columns with zeros
    for col in missing_in_test:
        aligned_test_df[col] = 0
    
    # Drop extra columns
    aligned_test_df = aligned_test_df.drop(columns=list(extra_in_test), errors='ignore')
    
    # Ensure the same column order as the training set
    aligned_test_df = aligned_test_df[train_df.columns]
    
    return aligned_test_df