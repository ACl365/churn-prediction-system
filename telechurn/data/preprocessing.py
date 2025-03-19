"""
TeleChurn Predictor - Data Preprocessing Module

This module handles the preprocessing of telecom customer data, including:
- Loading raw data from various sources
- Cleaning and handling missing values
- Basic transformations and encoding
- Creating a processed dataset ready for feature engineering

Author: Alexander Clarke
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelecomDataPreprocessor:
    """
    A class for preprocessing telecom customer data.
    
    This class provides methods to load, clean, transform, and prepare
    telecom customer data for machine learning models.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TelecomDataPreprocessor with optional configuration.
        
        Args:
            config (Dict, optional): Configuration parameters for preprocessing.
                Can include thresholds for outlier detection, strategies for
                missing value imputation, etc.
        """
        self.config = config or {}
        
        # Default configurations
        self.default_config = {
            'missing_threshold': 0.7,  # Drop columns with >70% missing values
            'correlation_threshold': 0.95,  # Threshold for high correlation
            'categorical_encoding': 'one-hot',  # Default encoding strategy
            'numerical_scaling': 'standard',  # Default scaling strategy
            'outlier_treatment': 'clip',  # Default outlier treatment
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
            'target_column': 'Churn',  # Target column name
            'id_column': 'CustomerID',  # ID column name
        }
        
        # Update default config with user-provided config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize column type dictionaries
        self.initialize_column_types()
        
        # Initialize transformers
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        logger.info("TelecomDataPreprocessor initialized with configuration")
    
    def initialize_column_types(self):
        """
        Initialize column type dictionaries based on domain knowledge of the telecom dataset.
        """
        # Columns to exclude from modeling (ID, target, etc.)
        self.exclude_columns = [self.config['id_column'], self.config['target_column']]
        
        # Categorical columns (non-binary, non-numeric)
        self.categorical_columns = [
            'ServiceArea', 'CreditRating', 'PrizmCode', 'Occupation', 'MaritalStatus'
        ]
        
        # Binary columns (yes/no or similar)
        self.binary_columns = [
            'ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner',
            'RVOwner', 'Homeownership', 'BuysViaMailOrder', 'RespondsToMailOffers',
            'OptOutMailings', 'NonUSTravel', 'OwnsComputer', 'HasCreditCard',
            'NewCellphoneUser', 'NotNewCellphoneUser', 'OwnsMotorcycle',
            'MadeCallToRetentionTeam'
        ]
        
        # Numerical columns (continuous)
        self.numerical_continuous_columns = [
            'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
            'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
            'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
            'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls',
            'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls',
            'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls',
            'CallForwardingCalls', 'CallWaitingCalls', 'AgeHH1', 'AgeHH2'
        ]
        
        # Numerical columns (discrete/integer)
        self.numerical_discrete_columns = [
            'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'Handsets',
            'HandsetModels', 'CurrentEquipmentDays', 'RetentionCalls',
            'RetentionOffersAccepted', 'ReferralsMadeBySubscriber',
            'IncomeGroup', 'AdjustmentsToCreditRating', 'HandsetPrice'
        ]
        
        # All numerical columns
        self.numerical_columns = self.numerical_continuous_columns + self.numerical_discrete_columns
        
        logger.info(f"Column types initialized: "
                   f"{len(self.numerical_columns)} numerical, "
                   f"{len(self.categorical_columns)} categorical, "
                   f"{len(self.binary_columns)} binary")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the data file.
            
        Returns:
            pd.DataFrame: Loaded data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                data = pd.read_excel(file_path)
            elif file_extension == '.json':
                data = pd.read_json(file_path)
            elif file_extension == '.parquet':
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Successfully loaded data from {file_path} with shape {data.shape}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def load_telecom_data(self, train_path: str, holdout_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load telecom churn data from train and optional holdout files.
        
        Args:
            train_path (str): Path to the training data file.
            holdout_path (str, optional): Path to the holdout/test data file.
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Tuple containing training data and optional holdout data.
        """
        train_data = self.load_data(train_path)
        logger.info(f"Loaded training data with {train_data.shape[0]} rows and {train_data.shape[1]} columns")
        
        holdout_data = None
        if holdout_path:
            holdout_data = self.load_data(holdout_path)
            logger.info(f"Loaded holdout data with {holdout_data.shape[0]} rows and {holdout_data.shape[1]} columns")
            
            # Check if columns match between train and holdout
            train_cols = set(train_data.columns)
            holdout_cols = set(holdout_data.columns)
            
            if train_cols != holdout_cols:
                missing_in_holdout = train_cols - holdout_cols
                missing_in_train = holdout_cols - train_cols
                
                if missing_in_holdout:
                    logger.warning(f"Columns in train but not in holdout: {missing_in_holdout}")
                
                if missing_in_train:
                    logger.warning(f"Columns in holdout but not in train: {missing_in_train}")
        
        return train_data, holdout_data
    
    def analyze_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Args:
            data (pd.DataFrame): The dataset to analyze.
            
        Returns:
            pd.DataFrame: A summary of missing values.
        """
        # Calculate missing values
        missing = data.isnull().sum()
        missing_percent = missing / len(data) * 100
        
        # Create summary DataFrame
        missing_summary = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Missing Percentage': missing_percent.values
        })
        
        # Sort by missing percentage
        missing_summary = missing_summary.sort_values('Missing Percentage', ascending=False)
        
        logger.info(f"Missing value analysis complete. "
                   f"{missing_summary['Missing Values'].sum()} total missing values found.")
        
        return missing_summary
    
    def handle_missing_values(self, data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): The dataset with missing values.
            is_train (bool): Whether this is training data (to fit imputers).
            
        Returns:
            pd.DataFrame: The dataset with handled missing values.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Analyze missing values
        missing_summary = self.analyze_missing_values(df)
        
        # Drop columns with too many missing values
        columns_to_drop = missing_summary[
            missing_summary['Missing Percentage'] > self.config['missing_threshold'] * 100
        ]['Column'].tolist()
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Dropped {len(columns_to_drop)} columns with >{self.config['missing_threshold']*100}% missing values: {columns_to_drop}")
        
        # Handle missing values for numerical columns
        for column in self.numerical_columns:
            if column not in df.columns:
                continue
                
            if df[column].isnull().sum() > 0:
                if is_train:
                    # Fit imputer on training data
                    imputer = SimpleImputer(strategy='median')
                    df[column] = imputer.fit_transform(df[[column]])
                    self.imputers[column] = imputer
                    logger.info(f"Fitted median imputer for '{column}'")
                else:
                    # Use pre-fitted imputer
                    if column in self.imputers:
                        df[column] = self.imputers[column].transform(df[[column]])
                        logger.info(f"Applied pre-fitted imputer to '{column}'")
                    else:
                        # Fallback if no imputer was fitted
                        df[column] = df[column].fillna(df[column].median())
                        logger.warning(f"No pre-fitted imputer found for '{column}', using median")
        
        # Handle missing values for categorical columns
        for column in self.categorical_columns:
            if column not in df.columns:
                continue
                
            if df[column].isnull().sum() > 0:
                if is_train:
                    # Fit imputer on training data
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[column] = imputer.fit_transform(df[[column]])
                    self.imputers[column] = imputer
                    logger.info(f"Fitted mode imputer for '{column}'")
                else:
                    # Use pre-fitted imputer
                    if column in self.imputers:
                        df[column] = self.imputers[column].transform(df[[column]])
                        logger.info(f"Applied pre-fitted imputer to '{column}'")
                    else:
                        # Fallback if no imputer was fitted
                        df[column] = df[column].fillna(df[column].mode()[0])
                        logger.warning(f"No pre-fitted imputer found for '{column}', using mode")
        
        # Handle missing values for binary columns
        for column in self.binary_columns:
            if column not in df.columns:
                continue
                
            if df[column].isnull().sum() > 0:
                if is_train:
                    # Fit imputer on training data
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[column] = imputer.fit_transform(df[[column]])
                    self.imputers[column] = imputer
                    logger.info(f"Fitted mode imputer for '{column}'")
                else:
                    # Use pre-fitted imputer
                    if column in self.imputers:
                        df[column] = self.imputers[column].transform(df[[column]])
                        logger.info(f"Applied pre-fitted imputer to '{column}'")
                    else:
                        # Fallback if no imputer was fitted
                        df[column] = df[column].fillna(df[column].mode()[0])
                        logger.warning(f"No pre-fitted imputer found for '{column}', using mode")
        
        return df
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            data (pd.DataFrame): The dataset with potential outliers.
            
        Returns:
            pd.DataFrame: The dataset with handled outliers.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        if self.config['outlier_treatment'] == 'none':
            return df
        
        for column in self.numerical_columns:
            if column not in df.columns:
                continue
                
            # Calculate z-scores
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = z_scores > self.config['outlier_threshold']
            
            if outliers.sum() == 0:
                continue
                
            logger.info(f"Found {outliers.sum()} outliers in '{column}'")
            
            if self.config['outlier_treatment'] == 'clip':
                # Clip outliers to the threshold
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Clipped outliers in '{column}' to range [{lower_bound}, {upper_bound}]")
                
            elif self.config['outlier_treatment'] == 'remove':
                # Remove rows with outliers
                df = df[~outliers]
                logger.info(f"Removed {outliers.sum()} rows with outliers in '{column}'")
                
            elif self.config['outlier_treatment'] == 'impute':
                # Replace outliers with median
                median_value = df[column].median()
                df.loc[outliers, column] = median_value
                logger.info(f"Replaced outliers in '{column}' with median: {median_value}")
        
        return df
    
    def encode_categorical_features(self, data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            data (pd.DataFrame): The dataset with categorical features.
            is_train (bool): Whether this is training data (to fit encoders).
            
        Returns:
            pd.DataFrame: The dataset with encoded categorical features.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        encoding_strategy = self.config['categorical_encoding']
        
        # Process binary columns (convert to 0/1)
        for column in self.binary_columns:
            if column not in df.columns:
                continue
                
            if is_train:
                # Map Yes/No to 1/0
                if df[column].dtype == 'object':
                    # Create a mapping dictionary
                    unique_values = df[column].dropna().unique()
                    
                    # Try to identify positive and negative values
                    positive_values = [val for val in unique_values if val.lower() in ['yes', 'y', 'true', 't', '1']]
                    negative_values = [val for val in unique_values if val.lower() in ['no', 'n', 'false', 'f', '0']]
                    
                    if positive_values and negative_values:
                        mapping = {val: 1 for val in positive_values}
                        mapping.update({val: 0 for val in negative_values})
                        
                        # Handle any other values as NaN
                        for val in unique_values:
                            if val not in mapping:
                                mapping[val] = np.nan
                        
                        df[column] = df[column].map(mapping)
                        self.encoders[column] = mapping
                        logger.info(f"Mapped binary column '{column}' using {mapping}")
                    else:
                        # If we can't identify positive/negative, use label encoding
                        le = LabelEncoder()
                        df[column] = le.fit_transform(df[column].astype(str))
                        self.encoders[column] = le
                        logger.info(f"Applied label encoding to binary column '{column}'")
            else:
                # Use pre-fitted encoder
                if column in self.encoders:
                    if isinstance(self.encoders[column], dict):
                        df[column] = df[column].map(self.encoders[column])
                    else:
                        # Handle unknown values by mapping to the most common class
                        try:
                            df[column] = self.encoders[column].transform(df[column].astype(str))
                        except ValueError:
                            # Handle unknown categories
                            logger.warning(f"Unknown categories in '{column}', using most frequent encoding")
                            most_common_class = self.encoders[column].transform([df[column].mode()[0].astype(str)])[0]
                            df[column] = df[column].apply(
                                lambda x: self.encoders[column].transform([str(x)])[0] 
                                if str(x) in self.encoders[column].classes_ 
                                else most_common_class
                            )
        
        # Process categorical columns
        for column in self.categorical_columns:
            if column not in df.columns:
                continue
                
            if encoding_strategy == 'one-hot':
                if is_train:
                    # Fit one-hot encoder
                    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = ohe.fit_transform(df[[column]])
                    
                    # Create new column names
                    encoded_cols = [f"{column}_{cat}" for cat in ohe.categories_[0]]
                    
                    # Add encoded columns to dataframe
                    for i, col in enumerate(encoded_cols):
                        df[col] = encoded[:, i]
                    
                    # Store encoder
                    self.encoders[column] = (ohe, encoded_cols)
                    
                    # Drop original column
                    df = df.drop(columns=[column])
                    logger.info(f"Applied one-hot encoding to '{column}', created {len(encoded_cols)} new columns")
                else:
                    # Use pre-fitted encoder
                    if column in self.encoders:
                        ohe, encoded_cols = self.encoders[column]
                        encoded = ohe.transform(df[[column]])
                        
                        # Add encoded columns to dataframe
                        for i, col in enumerate(encoded_cols):
                            df[col] = encoded[:, i]
                        
                        # Drop original column
                        df = df.drop(columns=[column])
                        logger.info(f"Applied pre-fitted one-hot encoding to '{column}'")
                    else:
                        logger.warning(f"No pre-fitted encoder found for '{column}', skipping")
                        
            elif encoding_strategy == 'label':
                if is_train:
                    # Fit label encoder
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column].astype(str))
                    self.encoders[column] = le
                    logger.info(f"Applied label encoding to '{column}'")
                else:
                    # Use pre-fitted encoder
                    if column in self.encoders:
                        # Handle unknown values by mapping to the most common class
                        try:
                            df[column] = self.encoders[column].transform(df[column].astype(str))
                        except ValueError:
                            # Handle unknown categories
                            logger.warning(f"Unknown categories in '{column}', using most frequent encoding")
                            most_common_class = self.encoders[column].transform([df[column].mode()[0].astype(str)])[0]
                            df[column] = df[column].apply(
                                lambda x: self.encoders[column].transform([str(x)])[0] 
                                if str(x) in self.encoders[column].classes_ 
                                else most_common_class
                            )
                    else:
                        logger.warning(f"No pre-fitted encoder found for '{column}', skipping")
        
        return df
    
    def scale_numerical_features(self, data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            data (pd.DataFrame): The dataset with numerical features.
            is_train (bool): Whether this is training data (to fit scalers).
            
        Returns:
            pd.DataFrame: The dataset with scaled numerical features.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        scaling_strategy = self.config['numerical_scaling']
        
        # Skip scaling if none
        if scaling_strategy == 'none':
            return df
        
        # Get numerical columns that exist in the dataframe
        columns_to_scale = [col for col in self.numerical_columns if col in df.columns]
        
        if not columns_to_scale:
            return df
        
        if scaling_strategy == 'standard':
            # Standardization (z-score normalization)
            if is_train:
                # Fit scaler on training data
                scaler = StandardScaler()
                df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                self.scalers['numerical'] = scaler
                logger.info(f"Fitted standard scaler to {len(columns_to_scale)} columns")
            else:
                # Use pre-fitted scaler
                if 'numerical' in self.scalers:
                    df[columns_to_scale] = self.scalers['numerical'].transform(df[columns_to_scale])
                    logger.info(f"Applied pre-fitted standard scaler to {len(columns_to_scale)} columns")
                else:
                    logger.warning("No pre-fitted scaler found, skipping scaling")
            
        elif scaling_strategy == 'minmax':
            # Min-max scaling
            from sklearn.preprocessing import MinMaxScaler
            
            if is_train:
                # Fit scaler on training data
                scaler = MinMaxScaler()
                df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                self.scalers['numerical'] = scaler
                logger.info(f"Fitted min-max scaler to {len(columns_to_scale)} columns")
            else:
                # Use pre-fitted scaler
                if 'numerical' in self.scalers:
                    df[columns_to_scale] = self.scalers['numerical'].transform(df[columns_to_scale])
                    logger.info(f"Applied pre-fitted min-max scaler to {len(columns_to_scale)} columns")
                else:
                    logger.warning("No pre-fitted scaler found, skipping scaling")
            
        elif scaling_strategy == 'robust':
            # Robust scaling (using median and IQR)
            from sklearn.preprocessing import RobustScaler
            
            if is_train:
                # Fit scaler on training data
                scaler = RobustScaler()
                df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                self.scalers['numerical'] = scaler
                logger.info(f"Fitted robust scaler to {len(columns_to_scale)} columns")
            else:
                # Use pre-fitted scaler
                if 'numerical' in self.scalers:
                    df[columns_to_scale] = self.scalers['numerical'].transform(df[columns_to_scale])
                    logger.info(f"Applied pre-fitted robust scaler to {len(columns_to_scale)} columns")
                else:
                    logger.warning("No pre-fitted scaler found, skipping scaling")
        
        return df
    
    def create_feature_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature groups and aggregations that might be useful.
        
        Args:
            data (pd.DataFrame): The dataset.
            
        Returns:
            pd.DataFrame: The dataset with additional feature groups.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Create call-related feature groups
        if all(col in df.columns for col in ['DroppedCalls', 'BlockedCalls']):
            df['TotalCallIssues'] = df['DroppedCalls'] + df['BlockedCalls']
            logger.info("Created 'TotalCallIssues' feature")
        
        if all(col in df.columns for col in ['OutboundCalls', 'InboundCalls']):
            df['TotalCalls'] = df['OutboundCalls'] + df['InboundCalls']
            logger.info("Created 'TotalCalls' feature")
            
            if 'TotalCalls' in df.columns and df['TotalCalls'].sum() > 0:
                df['OutboundCallRatio'] = df['OutboundCalls'] / df['TotalCalls'].replace(0, 1)
                logger.info("Created 'OutboundCallRatio' feature")
        
        # Create customer service related features
        if all(col in df.columns for col in ['CustomerCareCalls', 'RetentionCalls']):
            df['TotalServiceCalls'] = df['CustomerCareCalls'] + df['RetentionCalls']
            logger.info("Created 'TotalServiceCalls' feature")
        
        # Create handset-related features
        if all(col in df.columns for col in ['Handsets', 'HandsetModels']):
            df['HandsetsPerModel'] = df['Handsets'] / df['HandsetModels'].replace(0, 1)
            logger.info("Created 'HandsetsPerModel' feature")
        
        # Create revenue-related features
        if all(col in df.columns for col in ['MonthlyRevenue', 'MonthlyMinutes']):
            df['RevenuePerMinute'] = df['MonthlyRevenue'] / df['MonthlyMinutes'].replace(0, 1)
            logger.info("Created 'RevenuePerMinute' feature")
        
        return df
    
    def preprocess(self, data: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Preprocess the dataset with all preprocessing steps.
        
        Args:
            data (pd.DataFrame): The raw dataset.
            is_train (bool): Whether this is training data (to fit transformers).
                
        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        logger.info("Starting preprocessing pipeline")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        df = self.handle_missing_values(df, is_train)
        
        # Handle outliers (only for training data or if explicitly configured)
        if is_train or self.config.get('apply_outlier_treatment_to_test', False):
            df = self.handle_outliers(df)
        
        # Create feature groups
        df = self.create_feature_groups(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, is_train)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, is_train)
        
        # Prepare target variable if it exists and this is training data
        target_column = self.config['target_column']
        if target_column in df.columns:
            if df[target_column].dtype == 'object':
                # Convert Yes/No to 1/0
                target_mapping = {'Yes': 1, 'No': 0}
                df[target_column] = df[target_column].map(target_mapping)
                logger.info(f"Mapped target column '{target_column}' using {target_mapping}")
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df.shape}")
        
        return df
    
    def process_and_save(self, train_path: str, output_train_path: str, 
                         holdout_path: Optional[str] = None, 
                         output_holdout_path: Optional[str] = None) -> None:
        """
        Load, preprocess, and save telecom churn data.
        
        Args:
            train_path (str): Path to the training data file.
            output_train_path (str): Path to save the processed training data.
            holdout_path (str, optional): Path to the holdout/test data file.
            output_holdout_path (str, optional): Path to save the processed holdout data.
        """
        # Load data
        train_data, holdout_data = self.load_telecom_data(train_path, holdout_path)
        
        # Preprocess training data
        processed_train = self.preprocess(train_data, is_train=True)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
        
        # Save processed training data
        file_extension = os.path.splitext(output_train_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                processed_train.to_csv(output_train_path, index=False)
            elif file_extension == '.parquet':
                processed_train.to_parquet(output_train_path, index=False)
            else:
                processed_train.to_csv(output_train_path, index=False)
                logger.warning(f"Unsupported output format: {file_extension}. Saved as CSV instead.")
            
            logger.info(f"Successfully saved processed training data to {output_train_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed training data to {output_train_path}: {str(e)}")
            raise
        
        # Process and save holdout data if provided
        if holdout_data is not None and output_holdout_path:
            # Preprocess holdout data using transformers fitted on training data
            processed_holdout = self.preprocess(holdout_data, is_train=False)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_holdout_path), exist_ok=True)
            
            # Save processed holdout data
            file_extension = os.path.splitext(output_holdout_path)[1].lower()
            
            try:
                if file_extension == '.csv':
                    processed_holdout.to_csv(output_holdout_path, index=False)
                elif file_extension == '.parquet':
                    processed_holdout.to_parquet(output_holdout_path, index=False)
                else:
                    processed_holdout.to_csv(output_holdout_path, index=False)
                    logger.warning(f"Unsupported output format: {file_extension}. Saved as CSV instead.")
                
                logger.info(f"Successfully saved processed holdout data to {output_holdout_path}")
                
            except Exception as e:
                logger.error(f"Error saving processed holdout data to {output_holdout_path}: {str(e)}")
                raise


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'target_column': 'Churn',
        'id_column': 'CustomerID',
        'missing_threshold': 0.7,
        'correlation_threshold': 0.95,
        'categorical_encoding': 'one-hot',
        'numerical_scaling': 'standard',
        'outlier_treatment': 'clip',
        'outlier_threshold': 3.0,
    }
    
    # Initialize preprocessor
    preprocessor = TelecomDataPreprocessor(config)
    
    # Process and save data
    preprocessor.process_and_save(
        train_path="C:/Users/alex5/Documents/Projects/telecom_churn/Raw Data/cell2celltrain.csv",
        output_train_path="C:/Users/alex5/Documents/Projects/telecom_churn/project/data/processed/train_processed.csv",
        holdout_path="C:/Users/alex5/Documents/Projects/telecom_churn/Raw Data/cell2cellholdout.csv",
        output_holdout_path="C:/Users/alex5/Documents/Projects/telecom_churn/project/data/processed/holdout_processed.csv"
    )