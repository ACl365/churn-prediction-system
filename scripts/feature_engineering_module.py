"""
Feature Engineering Module for Telecom Customer Churn Prediction

This module contains functions for creating advanced engineered features
from telecom customer data to improve churn prediction accuracy.

Features are organized into categories:
- Customer behavioral features
- Usage pattern features
- Change over time features
- Ratio and aggregation features
- Customer profile features

Each feature is documented with its business meaning and predictive relevance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA


class FeatureEngineer:
    """
    Class for engineering features from telecom customer data for churn prediction.
    """
    
    def __init__(self, remove_correlated=True, correlation_threshold=0.85):
        """
        Initialize the FeatureEngineer.
        
        Parameters:
        -----------
        remove_correlated : bool, default=True
            Whether to remove highly correlated features
        correlation_threshold : float, default=0.85
            Threshold above which to consider features as highly correlated
        """
        self.remove_correlated = remove_correlated
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()
        self.one_hot_encoder = None
        self.selected_features = None
        
    def fit_transform(self, df, target_col='Churn'):
        """
        Create engineered features and transform the input dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with raw telecom customer data
        target_col : str, default='Churn'
            Name of the target column for churn prediction
            
        Returns:
        --------
        pandas.DataFrame
            Transformed dataframe with engineered features
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Convert target to binary if it's not already
        if target_col in data.columns and data[target_col].dtype == 'object':
            data[target_col] = data[target_col].map({'Yes': 1, 'No': 0})
        
        # Generate all feature categories
        data = self._create_behavioral_features(data)
        data = self._create_usage_pattern_features(data)
        data = self._create_change_features(data)
        data = self._create_ratio_features(data)
        data = self._create_profile_features(data)
        
        # Handle categorical features
        data = self._handle_categorical_features(data)
        
        # Scale numerical features
        data = self._scale_numerical_features(data, exclude=[target_col] if target_col in data.columns else [])
        
        # Remove highly correlated features if specified
        if self.remove_correlated and target_col in data.columns:
            data = self._remove_correlated_features(data, exclude=[target_col])
        
        # Select the most important features
        if target_col in data.columns:
            data = self._select_important_features(data, target_col)
            
        return data
    
    def transform(self, df):
        """
        Transform new data using the fitted feature engineering pipeline.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe to transform
            
        Returns:
        --------
        pandas.DataFrame
            Transformed dataframe with engineered features
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Apply the same feature engineering steps as in fit_transform
        # but without fitting again
        data = self._create_behavioral_features(data)
        data = self._create_usage_pattern_features(data)
        data = self._create_change_features(data)
        data = self._create_ratio_features(data)
        data = self._create_profile_features(data)
        
        # Apply pre-fitted transformations
        if self.one_hot_encoder is not None:
            cat_features = [col for col in data.columns if data[col].dtype == 'object']
            if cat_features:
                cat_data = pd.DataFrame(
                    self.one_hot_encoder.transform(data[cat_features]),
                    columns=self.one_hot_encoder.get_feature_names_out(cat_features),
                    index=data.index
                )
                data = data.drop(cat_features, axis=1)
                data = pd.concat([data, cat_data], axis=1)
        
        # Filter to only include selected features if they've been determined
        if self.selected_features is not None:
            available_features = [f for f in self.selected_features if f in data.columns]
            data = data[available_features]
            
        return data
    
    def _create_behavioral_features(self, df):
        """
        Create features that capture customer behavior patterns.
        
        Business Meaning: These features capture how customers interact with the service, 
        which can be strong indicators of satisfaction or frustration.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added behavioral features
        """
        data = df.copy()
        
        # Customer service interaction intensity
        # Business meaning: Higher service interaction often indicates frustration or problems
        if 'CustomerCareCalls' in data.columns:
            data['CustomerCareCallsPerMonth'] = data['CustomerCareCalls'] / data['MonthsInService'].clip(lower=1)
        
        # Problem indicator features
        # Business meaning: Technical issues are a major driver of churn
        if all(col in data.columns for col in ['DroppedCalls', 'BlockedCalls', 'UnansweredCalls']):
            data['TotalProblemCalls'] = data['DroppedCalls'] + data['BlockedCalls'] + data['UnansweredCalls']
            data['ProblemCallsPerMonth'] = data['TotalProblemCalls'] / data['MonthsInService'].clip(lower=1)
        
        # Service utilization features
        # Business meaning: How actively the customer is using the service
        if all(col in data.columns for col in ['OutboundCalls', 'InboundCalls']):
            data['TotalCalls'] = data['OutboundCalls'] + data['InboundCalls']
            data['CallsPerMonth'] = data['TotalCalls'] / data['MonthsInService'].clip(lower=1)
        
        # Feature usage breadth
        # Business meaning: Customers using more features are typically more "sticky"
        feature_columns = ['ThreewayCalls', 'CallForwardingCalls', 'CallWaitingCalls']
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        if feature_columns:
            # Count how many special features the customer uses (has > 0 usage)
            data['SpecialFeaturesUsed'] = (data[feature_columns] > 0).sum(axis=1)
        
        # Retention history
        # Business meaning: Past retention activity suggests higher risk
        if all(col in data.columns for col in ['RetentionCalls', 'RetentionOffersAccepted']):
            data['RetentionCallsPerMonth'] = data['RetentionCalls'] / data['MonthsInService'].clip(lower=1)
            data['RetentionSuccessRate'] = data['RetentionOffersAccepted'] / data['RetentionCalls'].replace(0, 1)
        
        return data
    
    def _create_usage_pattern_features(self, df):
        """
        Create features that capture the customer's usage patterns.
        
        Business Meaning: How customers use the service reveals their 
        dependency on it and satisfaction with it.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added usage pattern features
        """
        data = df.copy()
        
        # Time-of-day usage patterns
        # Business meaning: Different usage patterns may indicate different customer segments
        if all(col in data.columns for col in ['PeakCallsInOut', 'OffPeakCallsInOut']):
            total_calls = data['PeakCallsInOut'] + data['OffPeakCallsInOut']
            data['PeakCallsRatio'] = data['PeakCallsInOut'] / total_calls.replace(0, 1)
        
        # Usage intensity relative to cost
        # Business meaning: Value perception is critical to retention
        if all(col in data.columns for col in ['MonthlyMinutes', 'MonthlyRevenue']):
            data['RevenuePerMinute'] = data['MonthlyRevenue'] / data['MonthlyMinutes'].replace(0, 1)
        
        # Additional service usage
        # Business meaning: Usage of premium services indicates investment in the relationship
        if all(col in data.columns for col in ['DirectorAssistedCalls', 'RoamingCalls']):
            data['PremiumServiceUsage'] = data['DirectorAssistedCalls'] > 0
            data['RoamingServiceUsage'] = data['RoamingCalls'] > 0
        
        # Overage behavior
        # Business meaning: Frequent overages may indicate plan mismatch and bill shock
        if 'OverageMinutes' in data.columns:
            data['HasOverages'] = data['OverageMinutes'] > 0
        
        return data
    
    def _create_change_features(self, df):
        """
        Create features that capture changes in customer behavior over time.
        
        Business Meaning: Sudden changes in usage patterns are strong 
        predictors of potential churn.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added change features
        """
        data = df.copy()
        
        # Normalize change metrics to handle extreme values
        # Business meaning: Direction and magnitude of usage change indicates satisfaction trends
        if 'PercChangeMinutes' in data.columns:
            data['NormalizedChangeMinutes'] = data['PercChangeMinutes'].clip(lower=-100, upper=100) / 100
        
        if 'PercChangeRevenues' in data.columns:
            data['NormalizedChangeRevenues'] = data['PercChangeRevenues'].clip(lower=-100, upper=100) / 100
        
        # Combined change indicator
        # Business meaning: Aligns minute and revenue changes to detect consistency
        if all(col in data.columns for col in ['PercChangeMinutes', 'PercChangeRevenues']):
            # Detect when both metrics are moving in the same direction (both positive or both negative)
            data['ConsistentChangeDirection'] = (
                (data['PercChangeMinutes'] > 0) & (data['PercChangeRevenues'] > 0)
            ) | (
                (data['PercChangeMinutes'] < 0) & (data['PercChangeRevenues'] < 0)
            )
            
            # Detect large negative changes in either metric (potential warning sign)
            data['LargeNegativeChange'] = (data['PercChangeMinutes'] < -20) | (data['PercChangeRevenues'] < -20)
        
        # Customer tenure and lifecycle features
        # Business meaning: Churn risk varies significantly by tenure
        if 'MonthsInService' in data.columns:
            # Create tenure buckets for segmentation
            data['TenureBucket'] = pd.cut(
                data['MonthsInService'], 
                bins=[0, 3, 6, 12, 24, float('inf')],
                labels=['0-3 months', '3-6 months', '6-12 months', '1-2 years', '2+ years']
            )
            
            # New customers are often at higher risk
            data['IsNewCustomer'] = data['MonthsInService'] <= 3
        
        return data
    
    def _create_ratio_features(self, df):
        """
        Create ratio and aggregation features that capture relationships between variables.
        
        Business Meaning: Relational features often have stronger predictive power
        than absolute values for churn prediction.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added ratio features
        """
        data = df.copy()
        
        # Value and cost perception indicators
        # Business meaning: Value perception is a key driver of satisfaction
        if all(col in data.columns for col in ['MonthlyRevenue', 'MonthlyMinutes']):
            # Calculate average revenue per minute
            data['ARPM'] = data['MonthlyRevenue'] / data['MonthlyMinutes'].replace(0, 1)
        
        # Service quality indicators
        # Business meaning: Quality issues are leading churn drivers
        if all(col in data.columns for col in ['DroppedCalls', 'BlockedCalls', 'TotalCalls']):
            # Calculate problem call ratio from total calls
            problem_calls = data['DroppedCalls'] + data['BlockedCalls']
            if 'TotalCalls' in data.columns:
                data['ProblemCallRatio'] = problem_calls / data['TotalCalls'].replace(0, 1)
        
        # Customer service satisfaction proxies
        # Business meaning: Resolution efficiency indicates service quality
        if all(col in data.columns for col in ['CustomerCareCalls', 'RetentionCalls']):
            # Ratio of retention interactions to customer service interactions
            data['RetentionToServiceRatio'] = data['RetentionCalls'] / data['CustomerCareCalls'].replace(0, 1)
        
        # Equipment and technology indicators
        # Business meaning: Device satisfaction impacts overall experience
        if all(col in data.columns for col in ['CurrentEquipmentDays', 'MonthsInService']):
            # Calculate what percentage of the relationship the customer has had their current device
            service_days = data['MonthsInService'] * 30  # Approximate days
            data['EquipmentLifeRatio'] = data['CurrentEquipmentDays'] / service_days.replace(0, 1)
        
        # Plan optimization indicators
        # Business meaning: Plan fit is critical to value perception
        if all(col in data.columns for col in ['TotalRecurringCharge', 'MonthlyRevenue']):
            # Calculate what percentage of bill is from recurring plans vs. overages/extra services
            data['RecurringRevenueRatio'] = data['TotalRecurringCharge'] / data['MonthlyRevenue'].replace(0, 1)
        
        return data
    
    def _create_profile_features(self, df):
        """
        Create customer profile features based on demographics and other characteristics.
        
        Business Meaning: Different customer segments have different churn patterns
        and triggers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added profile features
        """
        data = df.copy()
        
        # Customer sophistication indicators
        # Business meaning: Tech-savvy customers may have different expectations
        tech_columns = ['OwnsComputer', 'HandsetWebCapable']
        tech_columns = [col for col in tech_columns if col in data.columns]
        
        if tech_columns:
            # Create tech-savvy score (1 point for each tech indicator)
            for col in tech_columns:
                if data[col].dtype == 'object':
                    data[f'{col}_Binary'] = (data[col] == 'Yes').astype(int)
                else:
                    data[f'{col}_Binary'] = data[col]
            
            tech_binary_cols = [f'{col}_Binary' for col in tech_columns]
            data['TechSavvyScore'] = data[tech_binary_cols].sum(axis=1)
            
            # Clean up temporary columns
            data = data.drop(tech_binary_cols, axis=1)
        
        # Household complexity indicators
        # Business meaning: Household composition affects service needs
        if 'ChildrenInHH' in data.columns:
            data['HasChildren'] = (data['ChildrenInHH'] == 'Yes').astype(int)
        
        if all(col in data.columns for col in ['AgeHH1', 'AgeHH2']):
            # Identify single vs. multi-person households
            data['IsMultiPersonHH'] = (data['AgeHH2'] > 0).astype(int)
            
            # Calculate average household age
            data['AvgHHAge'] = data.apply(
                lambda x: (x['AgeHH1'] + x['AgeHH2']) / 2 if x['AgeHH2'] > 0 else x['AgeHH1'],
                axis=1
            )
            
            # Create age-based segments
            data['AgeSegment'] = pd.cut(
                data['AvgHHAge'],
                bins=[0, 25, 35, 50, 65, float('inf')],
                labels=['18-25', '26-35', '36-50', '51-65', '65+']
            )
        
        # Customer engagement and loyalty indicators
        # Business meaning: Engaged customers are typically more loyal
        if 'ReferralsMadeBySubscriber' in data.columns:
            data['HasMadeReferrals'] = (data['ReferralsMadeBySubscriber'] > 0).astype(int)
            data['HighReferrer'] = (data['ReferralsMadeBySubscriber'] >= 2).astype(int)
        
        # Credit and financial indicators
        # Business meaning: Financial stability correlates with churn patterns
        if 'CreditRating' in data.columns:
            # Map credit ratings to numeric scale if needed
            credit_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
            if all(val in credit_mapping for val in data['CreditRating'].unique()):
                data['CreditScore'] = data['CreditRating'].map(credit_mapping)
        
        if 'IncomeGroup' in data.columns:
            # Create income tier indicators
            data['HighIncome'] = (data['IncomeGroup'] >= 7).astype(int)
            data['LowIncome'] = (data['IncomeGroup'] <= 3).astype(int)
        
        return data
    
    def _handle_categorical_features(self, df):
        """
        Apply encoding to categorical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with encoded categorical features
        """
        data = df.copy()
        
        # Identify categorical columns
        cat_features = [col for col in data.columns if data[col].dtype == 'object']
        
        if cat_features:
            # Initialize and fit the encoder if not already done
            if self.one_hot_encoder is None:
                self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.one_hot_encoder.fit(data[cat_features])
            
            # Transform the categorical features
            cat_data = pd.DataFrame(
                self.one_hot_encoder.transform(data[cat_features]),
                columns=self.one_hot_encoder.get_feature_names_out(cat_features),
                index=data.index
            )
            
            # Drop original categorical columns and concatenate encoded features
            data = data.drop(cat_features, axis=1)
            data = pd.concat([data, cat_data], axis=1)
        
        return data
    
    def _scale_numerical_features(self, df, exclude=None):
        """
        Scale numerical features using StandardScaler.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        exclude : list, default=None
            List of columns to exclude from scaling
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with scaled numerical features
        """
        data = df.copy()
        exclude = exclude or []
        
        # Identify numerical columns to scale (excluding specified columns)
        num_features = [
            col for col in data.columns 
            if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
        ]
        
        if num_features:
            # Fit the scaler on numerical features if not already done
            self.scaler.fit(data[num_features])
            
            # Transform the numerical features
            data[num_features] = self.scaler.transform(data[num_features])
        
        return data
    
    def _remove_correlated_features(self, df, exclude=None):
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        exclude : list, default=None
            List of columns to exclude from correlation analysis
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with uncorrelated features
        """
        data = df.copy()
        exclude = exclude or []
        
        # Calculate correlation matrix for numerical features
        corr_features = [
            col for col in data.columns 
            if col not in exclude and pd.api.types.is_numeric_dtype(data[col])
        ]
        
        if len(corr_features) > 1:
            corr_matrix = data[corr_features].corr().abs()
            
            # Create a mask for the upper triangle
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            to_drop = [
                column for column in upper_tri.columns 
                if any(upper_tri[column] > self.correlation_threshold)
            ]
            
            # Drop highly correlated features
            if to_drop:
                data = data.drop(to_drop, axis=1)
                print(f"Removed {len(to_drop)} highly correlated features.")
        
        return data
    
    def _select_important_features(self, df, target_col, k=None):
        """
        Select the most important features based on their relationship with the target.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_col : str
            Name of the target column
        k : int, default=None
            Number of top features to select. If None, uses a heuristic
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with selected important features
        """
        data = df.copy()
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Set k if not specified (heuristic: square root of number of features)
        if k is None:
            k = min(int(np.sqrt(X.shape[1])), X.shape[1])
        
        # Select numerical features only
        num_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        
        if num_features:
            # Use SelectKBest with F-classification to select top features
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X[num_features], y)
            
            # Get selected features
            selected_mask = selector.get_support()
            selected_num_features = [num_features[i] for i in range(len(num_features)) if selected_mask[i]]
            
            # Store all selected features (include non-numerical features)
            self.selected_features = selected_num_features + [
                col for col in X.columns if col not in num_features
            ]
            
            # Return dataframe with selected features and target
            selected_features = self.selected_features + [target_col]
            return data[selected_features]
        
        return data


# Utility functions for feature importance and selection

def get_feature_importances(model, feature_names):
    """
    Extract feature importances from a trained model.
    
    Parameters:
    -----------
    model : trained model object
        Must have a feature_importances_ attribute (like tree-based models)
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature names and their importance scores,
        sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model doesn't have feature_importances_ attribute")
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance


def select_features_from_model(X, y, model, threshold='mean'):
    """
    Select features based on importance weights from a trained model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature dataframe
    y : pandas.Series
        Target variable
    model : estimator object
        Estimator with fit method and feature_importances_ attribute
    threshold : str or float, default='mean'
        Threshold value to use for feature selection
        
    Returns:
    --------
    list
        List of selected feature names
    """
    from sklearn.feature_selection import SelectFromModel
    
    # Create selector and fit it
    selector = SelectFromModel(model, threshold=threshold)
    selector.fit(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    return selected_features


def calculate_iv(data, feature, target):
    """
    Calculate Information Value (IV) for a feature.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    feature : str
        Feature column name
    target : str
        Target column name
        
    Returns:
    --------
    float
        Information Value of the feature
    """
    # Create bins for numerical features
    if pd.api.types.is_numeric_dtype(data[feature]):
        x = pd.qcut(data[feature], q=10, duplicates='drop')
    else:
        x = data[feature].astype('str')
    
    # Calculate WOE and IV
    df = pd.DataFrame({
        'x': x,
        'y': data[target]
    })
    
    # Count events and non-events
    df_grouped = df.groupby('x')['y'].agg(['sum', 'count'])
    df_grouped.columns = ['events', 'total']
    df_grouped['non_events'] = df_grouped['total'] - df_grouped['events']
    
    # Calculate percentages
    N_events = df_grouped['events'].sum()
    N_non_events = df_grouped['non_events'].sum()
    
    df_grouped['event_pct'] = df_grouped['events'] / N_events
    df_grouped['non_event_pct'] = df_grouped['non_events'] / N_non_events
    
    # Calculate WOE and IV
    df_grouped['woe'] = np.log(df_grouped['event_pct'] / df_grouped['non_event_pct'])
    df_grouped['iv'] = (df_grouped['event_pct'] - df_grouped['non_event_pct']) * df_grouped['woe']
    
    return df_grouped['iv'].sum()


def calculate_all_ivs(data, features, target):
    """
    Calculate IV for multiple features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    features : list
        List of feature column names
    target : str
        Target column name
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features and their IVs, sorted by IV
    """
    iv_values = []
    
    for feature in features:
        try:
            iv = calculate_iv(data, feature, target)
            iv_values.append((feature, iv))
        except:
            continue
    
    iv_df = pd.DataFrame(iv_values, columns=['Feature', 'IV'])
    return iv_df.sort_values('IV', ascending=False)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load the data
    df = pd.read_csv('data/cell2celltrain.csv')
    
    # Initialize the feature engineer
    feature_eng = FeatureEngineer()
    
    # Generate engineered features
    df_featured = feature_eng.fit_transform(df)
    
    # Print shape to confirm feature generation
    print(f"Original shape: {df.shape}")
    print(f"Engineered shape: {df_featured.shape}")
