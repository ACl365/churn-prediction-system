"""
Gradient Boosting Model Module for Telecom Customer Churn Prediction

This module implements gradient boosting models for churn prediction
using both XGBoost and LightGBM implementations.
"""

import pandas as pd
import numpy as np
from base_model import BaseModel
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting model for churn prediction.
    Supports both XGBoost and LightGBM implementations.
    """
    
    def __init__(
        self,
        model_name="GradientBoostingModel",
        implementation="xgboost",
        params=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ):
        """
        Initialize the Gradient Boosting model.
        
        Parameters:
        -----------
        model_name : str, default="GradientBoostingModel"
            Name of the model
        implementation : str, default="xgboost"
            Gradient boosting implementation to use ("xgboost" or "lightgbm")
        params : dict, default=None
            Model hyperparameters. If None, default parameters will be used.
        random_state : int, default=42
            Random seed for reproducibility
        class_weight : str, default="balanced"
            Weighting scheme for imbalanced classification
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors)
        """
        super().__init__(model_name=model_name, random_state=random_state)
        
        self.implementation = implementation.lower()
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        
        # Set default parameters based on implementation
        if params is None:
            if self.implementation == "xgboost":
                self.params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'scale_pos_weight': 1,  # Will be adjusted for class imbalance
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'random_state': random_state,
                    'n_jobs': n_jobs
                }
            elif self.implementation == "lightgbm":
                self.params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'num_leaves': 31,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'objective': 'binary',
                    'metric': 'auc',
                    'random_state': random_state,
                    'n_jobs': n_jobs
                }
            else:
                raise ValueError(f"Unsupported implementation: {implementation}. Use 'xgboost' or 'lightgbm'.")
        else:
            self.params = params
    
    def build(self):
        """
        Build the model with specified hyperparameters.
        """
        if self.implementation == "xgboost":
            self.model = xgb.XGBClassifier(**self.params)
        elif self.implementation == "lightgbm":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            raise ValueError(f"Unsupported implementation: {self.implementation}. Use 'xgboost' or 'lightgbm'.")
        
        return self
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=True):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        eval_set : list, default=None
            Validation data for early stopping
        early_stopping_rounds : int, default=None
            Number of rounds without improvement to trigger early stopping
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        self : object
            Fitted model instance
        """
        if self.model is None:
            self.build()
        
        # Store feature and target names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        
        # Determine class names if possible
        if hasattr(y, 'unique'):
            self.class_names = sorted(y.unique())
        
        # Handle class imbalance
        if self.class_weight == "balanced" and self.implementation == "xgboost":
            # Calculate class weight based on class distribution
            if len(np.unique(y)) == 2:
                neg_count, pos_count = np.bincount(y.astype(int))
                scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
                self.model.scale_pos_weight = scale_pos_weight
                if verbose:
                    print(f"Set scale_pos_weight to {scale_pos_weight} based on class distribution")
        
        # Fit the model
        if eval_set is not None and early_stopping_rounds is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        
        return self
    
    def tune_hyperparameters(
        self, 
        X, 
        y, 
        param_grid=None, 
        cv=5,
        scoring='roc_auc',
        n_iter=10,
        method='random',
        verbose=1
    ):
        """
        Tune model hyperparameters using cross-validation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        param_grid : dict, default=None
            Grid of parameters to search. If None, will use a default grid.
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='roc_auc'
            Scoring metric for evaluation
        n_iter : int, default=10
            Number of parameter settings sampled (for RandomizedSearchCV)
        method : str, default='random'
            Hyperparameter search method ('grid' or 'random')
        verbose : int, default=1
            Verbosity level
            
        Returns:
        --------
        self : object
            Fitted model instance with tuned hyperparameters
        """
        # Create model
        if self.model is None:
            self.build()
        
        # Set default parameter grid if not provided
        if param_grid is None:
            if self.implementation == "xgboost":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2]
                }
            elif self.implementation == "lightgbm":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_samples': [5, 20, 50],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                }
        
        # Choose search method
        if method.lower() == 'grid':
            search = GridSearchCV(
                self.model,
                param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=verbose
            )
        elif method.lower() == 'random':
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=verbose,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported search method: {method}. Use 'grid' or 'random'.")
        
        # Store feature and target names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        
        # Determine class names if possible
        if hasattr(y, 'unique'):
            self.class_names = sorted(y.unique())
        
        # Perform search
        search.fit(X, y)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        self.params = search.best_params_
        self.is_fitted = True
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        
        return self
    
    def plot_training_history(self, figsize=(12, 5)):
        """
        Plot the training history for models that support it.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 5)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if self.implementation == "xgboost" and hasattr(self.model, 'evals_result'):
            # Get evaluation results
            results = self.model.evals_result()
            if not results:
                print("No evaluation results available. Use eval_set and early_stopping_rounds in fit().")
                return
            
            # Plot
            plt.figure(figsize=figsize)
            for eval_set_name, metrics in results.items():
                for metric_name, values in metrics.items():
                    plt.plot(values, label=f"{eval_set_name}-{metric_name}")
            
            plt.title(f"{self.model_name} Training History")
            plt.xlabel("Iteration")
            plt.ylabel("Metric Value")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        elif self.implementation == "lightgbm" and hasattr(self.model, 'evals_result_'):
            # Get evaluation results
            results = self.model.evals_result_
            if not results:
                print("No evaluation results available. Use eval_set and early_stopping_rounds in fit().")
                return
            
            # Plot
            plt.figure(figsize=figsize)
            for eval_set_name, metrics in results.items():
                for metric_name, values in metrics.items():
                    plt.plot(values, label=f"{eval_set_name}-{metric_name}")
            
            plt.title(f"{self.model_name} Training History")
            plt.xlabel("Iteration")
            plt.ylabel("Metric Value")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        else:
            print("Training history not available for this model.")
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 10), importance_type='gain'):
        """
        Plot feature importance for gradient boosting models.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(12, 10)
            Figure size
        importance_type : str, default='gain'
            Type of feature importance to plot for XGBoost ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing feature importance values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"Feature {i}" for i in range(self.model.n_features_in_)]
        
        # XGBoost has more detailed feature importance options
        if self.implementation == "xgboost":
            # Convert to booster for more options if not already
            if hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                importance = booster.get_score(importance_type=importance_type)
                # Convert to DataFrame
                feature_importance = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=False)
                
                # Map numeric features to feature names if needed
                if all(f.startswith('f') and f[1:].isdigit() for f in feature_importance['Feature']):
                    feature_importance['Feature'] = feature_importance['Feature'].apply(
                        lambda x: feature_names[int(x[1:])] if int(x[1:]) < len(feature_names) else x
                    )
            else:
                # Fall back to standard feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
        else:
            # LightGBM standard feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f"Top {top_n} Feature Importance - {self.model_name}")
        plt.tight_layout()
        plt.show()
        
        return feature_importance


class XGBoostModel(GradientBoostingModel):
    """
    XGBoost implementation of the Gradient Boosting model.
    """
    
    def __init__(
        self,
        model_name="XGBoostModel",
        params=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ):
        """
        Initialize the XGBoost model.
        
        Parameters:
        -----------
        model_name : str, default="XGBoostModel"
            Name of the model
        params : dict, default=None
            Model hyperparameters. If None, default parameters will be used.
        random_state : int, default=42
            Random seed for reproducibility
        class_weight : str, default="balanced"
            Weighting scheme for imbalanced classification
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors)
        """
        super().__init__(
            model_name=model_name,
            implementation="xgboost",
            params=params,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs
        )


class LightGBMModel(GradientBoostingModel):
    """
    LightGBM implementation of the Gradient Boosting model.
    """
    
    def __init__(
        self,
        model_name="LightGBMModel",
        params=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ):
        """
        Initialize the LightGBM model.
        
        Parameters:
        -----------
        model_name : str, default="LightGBMModel"
            Name of the model
        params : dict, default=None
            Model hyperparameters. If None, default parameters will be used.
        random_state : int, default=42
            Random seed for reproducibility
        class_weight : str, default="balanced"
            Weighting scheme for imbalanced classification
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors)
        """
        super().__init__(
            model_name=model_name,
            implementation="lightgbm",
            params=params,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs
        )
