#!/usr/bin/env python
"""
Feature Engineering Module for Telecom Customer Churn Prediction

This module provides classes and functions for creating advanced features
from telecom customer data to predict churn. It includes:

1. FeatureEngineer class for creating behavioral, usage pattern, change, ratio, and profile features
2. Functions for feature selection and evaluation
3. Utility functions for feature importance analysis

The features are designed based on domain knowledge of telecom customer behavior
and are documented with their business meaning.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    A class for engineering features from telecom customer data.
    
    This class creates advanced features from raw telecom data, including:
    - Behavioral features (usage patterns, customer service interactions)
    - Usage pattern features (peak vs. off-peak, revenue per minute)
    - Change features (changes in usage over time)
    - Ratio features (meaningful business ratios)
    - Profile features (customer demographics and segments)
    
    It also provides feature selection to identify the most predictive features.
    
    Parameters
    ----------
    remove_correlated : bool, default=True
        Whether to remove highly correlated features
    correlation_threshold : float, default=0.85
        Threshold for correlation above which features are considered highly correlated
    n_features : int, default=30
        Number of features to select using SelectKBest
    """
    
    def __init__(self, remove_correlated=True, correlation_threshold=0.85, n_features=30):
        self.remove_correlated = remove_correlated
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.selected_features = None
        self.feature_selector = None
        self.scaler = StandardScaler()
    
    def fit_transform(self, data):
        """
        Create engineered features and fit feature selector.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data containing telecom customer features
            
        Returns
        -------
        pandas.DataFrame
            The data with engineered features
        """
        # Create engineered features
        data_featured = self._create_features(data)
        
        # Select important features if target is present
        if 'Churn' in data_featured.columns:
            data_featured = self._select_important_features(data_featured)
        
        return data_featured
    
    def transform(self, data):
        """
        Apply feature engineering to new data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data containing telecom customer features
            
        Returns
        -------
        pandas.DataFrame
            The data with engineered features
        """
        # Create engineered features
        data_featured = self._create_features(data)
        
        # Apply feature selection if it was fit
        if self.selected_features is not None and set(self.selected_features).issubset(set(data_featured.columns)):
            # Keep only selected features plus ID and target if present
            keep_cols = self.selected_features.copy()
            if 'CustomerID' in data_featured.columns:
                keep_cols.append('CustomerID')
            if 'Churn' in data_featured.columns:
                keep_cols.append('Churn')
            data_featured = data_featured[keep_cols]
        
        return data_featured
    
    def _create_features(self, data):
        """
        Create all engineered features.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data containing telecom customer features
            
        Returns
        -------
        pandas.DataFrame
            The data with engineered features
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Create each category of features
        df = self._create_behavioral_features(df)
        df = self._create_usage_pattern_features(df)
        df = self._create_change_features(df)
        df = self._create_ratio_features(df)
        df = self._create_profile_features(df)
        
        # Remove highly correlated features if specified
        if self.remove_correlated:
            df = self._remove_correlated_features(df)
        
        return df
    
    def _create_behavioral_features(self, data):
        """
        Create behavioral features related to customer interactions and usage.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data
            
        Returns
        -------
        pandas.DataFrame
            The data with behavioral features added
        """
        df = data.copy()
        
        # Normalize customer care calls by tenure
        # Business meaning: Rate of customer service inquiries, higher values may indicate frustration
        df['CustomerCareCallsPerMonth'] = df['CustomerCareCalls'] / df['MonthsInService'].clip(lower=1)
        
        # Total problem calls (dropped, blocked, unanswered)
        # Business meaning: Overall service quality issues experienced by customer
        df['TotalProblemCalls'] = df['DroppedCalls'] + df['BlockedCalls'] + df['UnansweredCalls']
        
        # Problem calls per month
        # Business meaning: Rate of service quality issues, higher values indicate poor experience
        df['ProblemCallsPerMonth'] = df['TotalProblemCalls'] / df['MonthsInService'].clip(lower=1)
        
        # Total calls (inbound + outbound)
        # Business meaning: Overall usage volume
        df['TotalCalls'] = df['InboundCalls'] + df['OutboundCalls']
        
        # Calls per month
        # Business meaning: Usage intensity, higher values indicate more active users
        df['CallsPerMonth'] = df['TotalCalls'] / df['MonthsInService'].clip(lower=1)
        
        # Special features used (call waiting, call forwarding, three-way calls)
        # Business meaning: Customer sophistication and engagement with advanced features
        df['SpecialFeaturesUsed'] = ((df['CallWaitingCalls'] > 0).astype(int) + 
                                    (df['CallForwardingCalls'] > 0).astype(int) + 
                                    (df['ThreewayCalls'] > 0).astype(int))
        
        # Retention calls per month
        # Business meaning: Rate of retention interventions, higher values indicate at-risk customer
        df['RetentionCallsPerMonth'] = df['RetentionCalls'] / df['MonthsInService'].clip(lower=1)
        
        # Retention success rate
        # Business meaning: Effectiveness of retention offers for this customer
        df['RetentionSuccessRate'] = df['RetentionOffersAccepted'] / df['RetentionCalls'].replace(0, 1)
        
        return df
    
    def _create_usage_pattern_features(self, data):
        """
        Create features related to how customers use the service.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data
            
        Returns
        -------
        pandas.DataFrame
            The data with usage pattern features added
        """
        df = data.copy()
        
        # Peak calls ratio (peak calls / total calls)
        # Business meaning: When customer uses service, higher values indicate business usage
        total_calls_in_out = df['PeakCallsInOut'] + df['OffPeakCallsInOut']
        df['PeakCallsRatio'] = df['PeakCallsInOut'] / total_calls_in_out.replace(0, 1)
        
        # Revenue per minute
        # Business meaning: Customer value per usage, higher values indicate premium services
        df['RevenuePerMinute'] = df['MonthlyRevenue'] / df['MonthlyMinutes'].replace(0, 1)
        
        # Premium service usage (director assisted calls)
        # Business meaning: Willingness to pay for premium services
        df['PremiumServiceUsage'] = (df['DirectorAssistedCalls'] > 0).astype(int)
        
        # Roaming service usage
        # Business meaning: Travel behavior, business users often have higher roaming
        df['RoamingServiceUsage'] = (df['RoamingCalls'] > 0).astype(int)
        
        # Has overages
        # Business meaning: Customer exceeds plan limits, potential upsell opportunity or churn risk
        df['HasOverages'] = (df['OverageMinutes'] > 0).astype(int)
        
        return df
    
    def _create_change_features(self, data):
        """
        Create features related to changes in customer behavior over time.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data
            
        Returns
        -------
        pandas.DataFrame
            The data with change features added
        """
        df = data.copy()
        
        # Normalized change in minutes (between -1 and 1)
        # Business meaning: Direction and magnitude of usage change, negative values indicate decreasing usage
        df['NormalizedChangeMinutes'] = df['PercChangeMinutes'].clip(lower=-100, upper=100) / 100
        
        # Normalized change in revenues
        # Business meaning: Direction and magnitude of revenue change, negative values indicate decreasing value
        df['NormalizedChangeRevenues'] = df['PercChangeRevenues'].clip(lower=-100, upper=100) / 100
        
        # Consistent change direction (both minutes and revenue changing in same direction)
        # Business meaning: Consistent pattern of increase or decrease across metrics
        df['ConsistentChangeDirection'] = ((df['PercChangeMinutes'] * df['PercChangeRevenues']) > 0).astype(int)
        
        # Large negative change (significant decrease in minutes or revenue)
        # Business meaning: Major red flag for potential churn
        df['LargeNegativeChange'] = ((df['PercChangeMinutes'] < -20) | 
                                    (df['PercChangeRevenues'] < -20)).astype(int)
        
        # Tenure buckets
        # Business meaning: Customer lifecycle stage, different stages have different churn patterns
        tenure_conditions = [
            (df['MonthsInService'] <= 3),
            (df['MonthsInService'] <= 6),
            (df['MonthsInService'] <= 12),
            (df['MonthsInService'] <= 24),
            (df['MonthsInService'] > 24)
        ]
        tenure_values = ['0-3 months', '4-6 months', '7-12 months', '1-2 years', '2+ years']
        df['TenureBucket'] = np.select(tenure_conditions, tenure_values, default='Unknown')
        
        # New customer flag
        # Business meaning: New customers have higher churn risk
        df['IsNewCustomer'] = (df['MonthsInService'] <= 3).astype(int)
        
        return df
    
    def _create_ratio_features(self, data):
        """
        Create ratio features that capture relationships between metrics.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data
            
        Returns
        -------
        pandas.DataFrame
            The data with ratio features added
        """
        df = data.copy()
        
        # Average Revenue Per Minute
        # Business meaning: Value of customer's usage, higher values indicate premium customers
        df['ARPM'] = df['MonthlyRevenue'] / df['MonthlyMinutes'].replace(0, 1)
        
        # Problem call ratio (problem calls / total calls)
        # Business meaning: Service quality experience, higher values indicate poor experience
        df['ProblemCallRatio'] = (df['DroppedCalls'] + df['BlockedCalls']) / df['TotalCalls'].replace(0, 1)
        
        # Retention to service ratio (retention calls / months in service)
        # Business meaning: How frequently customer requires retention intervention
        df['RetentionToServiceRatio'] = df['RetentionCalls'] / df['MonthsInService'].clip(lower=1)
        
        # Equipment life ratio (current equipment days / months in service)
        # Business meaning: How frequently customer upgrades equipment, lower values indicate upgrade-seeking behavior
        months_to_days = df['MonthsInService'] * 30
        df['EquipmentLifeRatio'] = df['CurrentEquipmentDays'] / months_to_days.replace(0, 1)
        
        # Recurring revenue ratio (recurring charge / monthly revenue)
        # Business meaning: Stability of revenue, higher values indicate more predictable revenue
        df['RecurringRevenueRatio'] = df['TotalRecurringCharge'] / df['MonthlyRevenue'].replace(0, 1)
        
        return df
    
    def _create_profile_features(self, data):
        """
        Create customer profile and demographic features.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data
            
        Returns
        -------
        pandas.DataFrame
            The data with profile features added
        """
        df = data.copy()
        
        # Tech savvy score (computer ownership + web capable handset)
        # Business meaning: Customer technological sophistication
        df['TechSavvyScore'] = 0
        if 'OwnsComputer' in df.columns:
            df['TechSavvyScore'] += (df['OwnsComputer'] == 'Yes').astype(int)
        if 'HandsetWebCapable' in df.columns:
            df['TechSavvyScore'] += (df['HandsetWebCapable'] == 'Yes').astype(int)
        
        # Has children flag
        # Business meaning: Family status affects usage patterns and churn factors
        if 'ChildrenInHH' in df.columns:
            df['HasChildren'] = (df['ChildrenInHH'] == 'Yes').astype(int)
        
        # Multi-person household
        # Business meaning: Household composition affects plan needs
        df['IsMultiPersonHH'] = ((df['AgeHH1'] > 0) & (df['AgeHH2'] > 0)).astype(int)
        
        # Average household age
        # Business meaning: Age demographic affects usage patterns and churn factors
        age_cols = ['AgeHH1', 'AgeHH2']
        age_data = df[age_cols].copy()
        age_data[age_data <= 0] = np.nan  # Replace 0 or negative ages with NaN
        df['AvgHHAge'] = age_data.mean(axis=1)
        
        # Age segment
        # Business meaning: Different age groups have different usage patterns and churn factors
        age_conditions = [
            (df['AvgHHAge'] < 25),
            (df['AvgHHAge'] < 35),
            (df['AvgHHAge'] < 45),
            (df['AvgHHAge'] < 55),
            (df['AvgHHAge'] >= 55)
        ]
        age_values = ['Under 25', '25-34', '35-44', '45-54', '55+']
        df['AgeSegment'] = np.select(age_conditions, age_values, default='Unknown')
        
        # Has made referrals
        # Business meaning: Customer advocacy, higher values indicate satisfaction
        df['HasMadeReferrals'] = (df['ReferralsMadeBySubscriber'] > 0).astype(int)
        
        # High referrer
        # Business meaning: Strong customer advocate, very satisfied customer
        df['HighReferrer'] = (df['ReferralsMadeBySubscriber'] >= 2).astype(int)
        
        # Credit score (inverse of credit rating adjustments)
        # Business meaning: Financial reliability, higher values indicate better credit
        df['CreditScore'] = 10 - df['AdjustmentsToCreditRating'].clip(upper=10)
        
        # Income segments
        # Business meaning: Financial capacity affects plan selection and churn factors
        df['HighIncome'] = (df['IncomeGroup'] >= 7).astype(int)
        df['LowIncome'] = (df['IncomeGroup'] <= 3).astype(int)
        
        return df
    
    def _remove_correlated_features(self, data):
        """
        Remove highly correlated features.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data
            
        Returns
        -------
        pandas.DataFrame
            The data with correlated features removed
        """
        df = data.copy()
        
        # Keep ID and target columns
        cols_to_keep = []
        if 'CustomerID' in df.columns:
            cols_to_keep.append('CustomerID')
        if 'Churn' in df.columns:
            cols_to_keep.append('Churn')
        
        # Get numeric columns for correlation analysis
        numeric_cols = [col for col in df.columns 
                       if col not in cols_to_keep and pd.api.types.is_numeric_dtype(df[col])]
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        # Drop highly correlated features
        df_filtered = df.drop(columns=to_drop)
        
        return df_filtered
    
    def _select_important_features(self, data):
        """
        Select the most important features using SelectKBest.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The input data with target column
            
        Returns
        -------
        pandas.DataFrame
            The data with only important features
        """
        df = data.copy()
        
        # Separate features from target and ID
        X = df.drop(['Churn'], axis=1)
        if 'CustomerID' in X.columns:
            X = X.drop(['CustomerID'], axis=1)
        y = df['Churn']
        
        # Convert categorical columns to numeric
        X_numeric = X.copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                # For categorical columns, create dummy variables
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_numeric = pd.concat([X_numeric, dummies], axis=1)
                X_numeric = X_numeric.drop(col, axis=1)
        
        # Apply feature selection
        self.feature_selector = SelectKBest(f_classif, k=min(self.n_features, X_numeric.shape[1]))
        self.feature_selector.fit(X_numeric, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = X_numeric.columns[selected_indices].tolist()
        
        # Map back to original feature names for dummy variables
        original_selected = []
        for feature in selected_features:
            # Check if this is a dummy variable
            if '_' in feature:
                # Extract the original column name (before the underscore)
                original_col = feature.split('_')[0]
                if original_col not in original_selected and original_col in X.columns:
                    original_selected.append(original_col)
            else:
                original_selected.append(feature)
        
        # Store selected features
        self.selected_features = original_selected
        
        # Return data with only selected features
        keep_cols = self.selected_features.copy()
        if 'CustomerID' in df.columns:
            keep_cols.append('CustomerID')
        keep_cols.append('Churn')
        
        return df[keep_cols]


def get_feature_importances(model, feature_names):
    """
    Extract feature importances from a trained model.
    
    Parameters
    ----------
    model : trained model object
        Must have a feature_importances_ attribute (like tree-based models)
    feature_names : list
        List of feature names
        
    Returns
    -------
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


def calculate_iv(data, feature, target):
    """
    Calculate Information Value for a single feature.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data
    feature : str
        The feature name
    target : str
        The target column name
        
    Returns
    -------
    float
        The Information Value
    """
    # Create bins for numeric features
    if pd.api.types.is_numeric_dtype(data[feature]):
        # Create 10 bins for numeric features
        data['bin'] = pd.qcut(data[feature], 10, duplicates='drop')
    else:
        # Use categories for categorical features
        data['bin'] = data[feature]
    
    # Calculate WOE and IV
    grouped = data.groupby('bin')[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / grouped['count']
    
    # Calculate totals
    total_events = grouped['sum'].sum()
    total_non_events = grouped['non_event'].sum()
    
    # Calculate WOE and IV
    grouped['event_pct'] = grouped['sum'] / total_events
    grouped['non_event_pct'] = grouped['non_event'] / total_non_events
    grouped['WOE'] = np.log(grouped['event_pct'] / grouped['non_event_pct'])
    grouped['IV'] = (grouped['event_pct'] - grouped['non_event_pct']) * grouped['WOE']
    
    # Return total IV
    return grouped['IV'].sum()


def calculate_all_ivs(data, features, target):
    """
    Calculate Information Value for multiple features.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data
    features : list
        List of feature names
    target : str
        The target column name
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature names and their IV scores,
        sorted by IV
    """
    iv_values = []
    
    for feature in features:
        try:
            iv = calculate_iv(data[[feature, target]], feature, target)
            iv_values.append({'Feature': feature, 'IV': iv})
        except:
            # Skip features that cause errors
            continue
    
    iv_df = pd.DataFrame(iv_values).sort_values('IV', ascending=False)
    
    # Add IV strength labels
    iv_df['Strength'] = pd.cut(
        iv_df['IV'],
        bins=[-float('inf'), 0.02, 0.1, 0.3, 0.5, float('inf')],
        labels=['Useless', 'Weak', 'Medium', 'Strong', 'Very Strong']
    )
    
    return iv_df


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")