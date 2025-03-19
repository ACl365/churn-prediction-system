"""
Ensemble Model Module for Telecom Customer Churn Prediction

This module implements ensemble techniques for combining multiple churn prediction models,
including voting, averaging, stacking, and blending methods optimized for business metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
import os
import time
from datetime import datetime

from base_model import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ensemble')


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models for improved churn prediction.
    """
    
    def __init__(
        self,
        base_models=None,
        ensemble_method='weighted',
        weights=None,
        meta_model=None,
        optimize_metric='f1',
        cv=5,
        model_name="EnsembleModel",
        random_state=42
    ):
        """
        Initialize the ensemble model.
        
        Parameters:
        -----------
        base_models : list, default=None
            List of base model instances to ensemble
        ensemble_method : str, default='weighted'
            Method for combining base models:
            - 'voting': Hard voting (majority rule)
            - 'averaging': Simple averaging of probabilities
            - 'weighted': Weighted averaging of probabilities
            - 'stacking': Train a meta-model on base model predictions
            - 'blending': Weighted average optimized on validation set
        weights : list or None, default=None
            Weights for weighted averaging or initial weights for blending.
            If None, equal weights are used.
        meta_model : object, default=None
            Meta-model for stacking. If None, LogisticRegression is used.
        optimize_metric : str, default='f1'
            Metric to optimize for blending ('accuracy', 'precision', 'recall', 'f1', 'auc')
        cv : int, default=5
            Number of cross-validation folds for stacking
        model_name : str, default="EnsembleModel"
            Name of the ensemble model
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(model_name=model_name, random_state=random_state)
        
        self.base_models = base_models if base_models is not None else []
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_model = meta_model
        self.optimize_metric = optimize_metric
        self.cv = cv
        
        # Initialize the meta-model if not provided
        if self.ensemble_method == 'stacking' and self.meta_model is None:
            self.meta_model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=random_state
            )
    
    def build(self):
        """
        Build the ensemble model.
        """
        # Validate that we have base models
        if not self.base_models:
            raise ValueError("No base models provided for ensemble.")
        
        # Validate all base models are fitted
        for i, model in enumerate(self.base_models):
            if not hasattr(model, 'is_fitted') or not model.is_fitted:
                raise ValueError(f"Base model {i} is not fitted.")
        
        # If weights not provided, use equal weights
        if self.weights is None:
            self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        else:
            # Normalize weights to sum to 1
            self.weights = np.array(self.weights) / np.sum(self.weights)
        
        logger.info(f"Built {self.model_name} with {len(self.base_models)} base models "
                  f"using {self.ensemble_method} method.")
        
        return self
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the ensemble model.
        
        For stacking, this method trains a meta-model on base model predictions.
        For blending, this method optimizes the weights on a validation set.
        For voting and averaging, this is a no-op as base models are already trained.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        X_val : array-like, shape (n_val_samples, n_features), default=None
            Validation data for blending. Required for 'blending' method.
        y_val : array-like, shape (n_val_samples,), default=None
            Validation target values for blending
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store feature and target names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        
        # Determine class names if possible
        if hasattr(y, 'unique'):
            self.class_names = sorted(y.unique())
        
        # Build the ensemble if not already built
        if not hasattr(self, 'weights') or self.weights is None:
            self.build()
        
        # Handle different ensemble methods
        if self.ensemble_method == 'stacking':
            self._fit_stacking(X, y)
        elif self.ensemble_method == 'blending':
            if X_val is None or y_val is None:
                raise ValueError("Validation data required for blending method.")
            self._fit_blending(X, y, X_val, y_val)
        else:
            # For voting and averaging, no additional fitting is needed
            logger.info(f"Using {self.ensemble_method} ensemble with pre-trained models.")
        
        self.is_fitted = True
        return self
    
    def _fit_stacking(self, X, y):
        """
        Fit a stacking ensemble.
        
        This method trains a meta-model using cross-validation predictions from base models.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        logger.info("Fitting stacking ensemble...")
        
        # Create cross-validation folds
        kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Generate cross-validation predictions for each base model
        cv_preds = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Generating cross-validation predictions for base model {i}...")
            fold_preds = np.zeros(X.shape[0])
            
            for train_idx, val_idx in kf.split(X, y):
                # Create train/val split
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                
                if isinstance(y, pd.Series):
                    y_train = y.iloc[train_idx]
                else:
                    y_train = y[train_idx]
                
                # Create a copy of the model to avoid refitting the original
                import copy
                model_copy = copy.deepcopy(model)
                
                # Fit on the training fold
                model_copy.fit(X_train, y_train)
                
                # Predict on the validation fold
                fold_preds[val_idx] = model_copy.predict_proba(X_val)[:, 1]
            
            # Store predictions for this model
            cv_preds[:, i] = fold_preds
            
            # Calculate AUC for this model
            auc = roc_auc_score(y, fold_preds)
            logger.info(f"Base model {i} cross-validation AUC: {auc:.4f}")
        
        # Train meta-model on base model predictions
        logger.info("Training meta-model on base model predictions...")
        self.meta_model.fit(cv_preds, y)
        
        # Store cross-validation predictions for later use
        self.cv_preds = cv_preds
        
        # Store meta-model coefficients
        if hasattr(self.meta_model, 'coef_'):
            self.meta_weights = self.meta_model.coef_[0]
            logger.info(f"Meta-model coefficients: {self.meta_weights}")
    
    def _fit_blending(self, X, y, X_val, y_val):
        """
        Fit a blending ensemble by optimizing weights on a validation set.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data (not used in this method but kept for API consistency)
        y : array-like, shape (n_samples,)
            Target values (not used in this method but kept for API consistency)
        X_val : array-like, shape (n_val_samples, n_features)
            Validation data
        y_val : array-like, shape (n_val_samples,)
            Validation target values
        """
        logger.info("Fitting blending ensemble...")
        
        # Get base model predictions on validation set
        val_preds = np.zeros((X_val.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            val_preds[:, i] = model.predict_proba(X_val)[:, 1]
            
            # Calculate performance metrics for this model
            y_pred = (val_preds[:, i] >= 0.5).astype(int)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, val_preds[:, i])
            
            logger.info(f"Base model {i} validation metrics - "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                       f"F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Optimize weights using a grid search or optimization algorithm
        logger.info("Optimizing ensemble weights...")
        
        # Simple grid search for 2-3 models
        if len(self.base_models) <= 3:
            self.weights = self._grid_search_weights(val_preds, y_val)
        else:
            # For more models, use a more efficient optimization approach
            self.weights = self._optimize_weights(val_preds, y_val)
        
        logger.info(f"Optimized weights: {self.weights}")
    
    def _grid_search_weights(self, val_preds, y_val, steps=10):
        """
        Perform grid search to find optimal weights for blending.
        
        Parameters:
        -----------
        val_preds : array-like, shape (n_samples, n_models)
            Validation predictions from base models
        y_val : array-like, shape (n_samples,)
            Validation target values
        steps : int, default=10
            Number of steps for the grid search
            
        Returns:
        --------
        array-like
            Optimal weights for blending
        """
        n_models = val_preds.shape[1]
        best_score = -1
        best_weights = np.ones(n_models) / n_models  # Start with equal weights
        
        if n_models == 2:
            # For 2 models, we only need to tune one weight (w1 + w2 = 1)
            weight_range = np.linspace(0, 1, steps)
            
            for w1 in weight_range:
                w2 = 1 - w1
                weights = np.array([w1, w2])
                
                # Calculate weighted predictions
                weighted_preds = np.sum(val_preds * weights.reshape(1, -1), axis=1)
                
                # Evaluate performance
                score = self._calculate_metric(y_val, weighted_preds)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
        
        elif n_models == 3:
            # For 3 models, we need to tune two weights (w1 + w2 + w3 = 1)
            weight_range = np.linspace(0, 1, steps)
            
            for w1 in weight_range:
                for w2 in weight_range:
                    if w1 + w2 <= 1:
                        w3 = 1 - w1 - w2
                        weights = np.array([w1, w2, w3])
                        
                        # Calculate weighted predictions
                        weighted_preds = np.sum(val_preds * weights.reshape(1, -1), axis=1)
                        
                        # Evaluate performance
                        score = self._calculate_metric(y_val, weighted_preds)
                        
                        if score > best_score:
                            best_score = score
                            best_weights = weights
        
        logger.info(f"Best {self.optimize_metric} score: {best_score:.4f}")
        return best_weights
    
    def _optimize_weights(self, val_preds, y_val):
        """
        Optimize weights using a more efficient algorithm for more than 3 models.
        
        Parameters:
        -----------
        val_preds : array-like, shape (n_samples, n_models)
            Validation predictions from base models
        y_val : array-like, shape (n_samples,)
            Validation target values
            
        Returns:
        --------
        array-like
            Optimal weights for blending
        """
        from scipy.optimize import minimize
        
        n_models = val_preds.shape[1]
        
        # Define the objective function (negative metric to minimize)
        def objective(weights):
            # Normalize weights to sum to 1
            weights_norm = weights / np.sum(weights)
            
            # Calculate weighted predictions
            weighted_preds = np.sum(val_preds * weights_norm.reshape(1, -1), axis=1)
            
            # Return negative metric (for minimization)
            return -self._calculate_metric(y_val, weighted_preds)
        
        # Constraint: weights sum to 1
        def constraint(weights):
            return np.sum(weights) - 1.0
        
        # Initial weights (equal weights)
        initial_weights = np.ones(n_models) / n_models
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Constraint specification
        constraints = {'type': 'eq', 'fun': constraint}
        
        # Run optimization
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Normalize weights to sum to 1
        optimal_weights = result.x / np.sum(result.x)
        
        # Calculate best score
        weighted_preds = np.sum(val_preds * optimal_weights.reshape(1, -1), axis=1)
        best_score = self._calculate_metric(y_val, weighted_preds)
        
        logger.info(f"Best {self.optimize_metric} score: {best_score:.4f}")
        return optimal_weights
    
    def _calculate_metric(self, y_true, y_pred_proba, threshold=0.5):
        """
        Calculate the specified metric for optimization.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred_proba : array-like
            Predicted probabilities
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        float
            Metric value
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate the specified metric
        if self.optimize_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.optimize_metric == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        elif self.optimize_metric == 'recall':
            return recall_score(y_true, y_pred)
        elif self.optimize_metric == 'f1':
            return f1_score(y_true, y_pred)
        elif self.optimize_metric == 'auc':
            return roc_auc_score(y_true, y_pred_proba)
        elif self.optimize_metric == 'average_precision':
            return average_precision_score(y_true, y_pred_proba)
        else:
            raise ValueError(f"Unsupported metric: {self.optimize_metric}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get predictions from all base models
        base_preds = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_preds[:, i] = model.predict_proba(X)[:, 1]
        
        # Apply ensemble method
        if self.ensemble_method == 'voting':
            # Hard voting (majority rule)
            binary_preds = (base_preds >= 0.5).astype(int)
            voted_preds = np.sum(binary_preds * self.weights.reshape(1, -1), axis=1) / np.sum(self.weights)
            # Convert to probabilities (0 or 1 based on threshold)
            proba = (voted_preds >= 0.5).astype(float)
            # Return in scikit-learn format [P(class=0), P(class=1)]
            return np.vstack([1 - proba, proba]).T
        
        elif self.ensemble_method == 'averaging':
            # Simple averaging of probabilities
            avg_proba = np.mean(base_preds, axis=1)
            return np.vstack([1 - avg_proba, avg_proba]).T
        
        elif self.ensemble_method == 'weighted':
            # Weighted averaging of probabilities
            weighted_proba = np.sum(base_preds * self.weights.reshape(1, -1), axis=1)
            return np.vstack([1 - weighted_proba, weighted_proba]).T
        
        elif self.ensemble_method == 'stacking':
            # Use meta-model for prediction
            meta_proba = self.meta_model.predict_proba(base_preds)
            return meta_proba
        
        elif self.ensemble_method == 'blending':
            # Weighted averaging with optimized weights
            weighted_proba = np.sum(base_preds * self.weights.reshape(1, -1), axis=1)
            return np.vstack([1 - weighted_proba, weighted_proba]).T
        
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
    
    def predict(self, X, threshold=0.5):
        """
        Predict class for the input samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted classes
        """
        # Get probability predictions
        proba = self.predict_proba(X)[:, 1]
        
        # Apply threshold
        return (proba >= threshold).astype(int)
    
    def get_model_contributions(self, X, detailed=False):
        """
        Get the contribution of each base model to the ensemble predictions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        detailed : bool, default=False
            Whether to return detailed contribution metrics
            
        Returns:
        --------
        dict or pd.DataFrame
            Contribution metrics for each model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get predictions from all base models
        base_preds = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_preds[:, i] = model.predict_proba(X)[:, 1]
        
        # Get ensemble predictions
        ensemble_preds = self.predict_proba(X)[:, 1]
        
        if not detailed:
            # Calculate correlations between each model and the ensemble
            correlations = [np.corrcoef(base_preds[:, i], ensemble_preds)[0, 1] 
                           for i in range(len(self.base_models))]
            
            # Calculate effective weights for each model
            if self.ensemble_method == 'stacking':
                # For stacking, the weights are the meta-model coefficients
                effective_weights = self.meta_weights / np.sum(np.abs(self.meta_weights))
            else:
                # For other methods, use the actual weights
                effective_weights = self.weights
            
            return pd.DataFrame({
                'Model': [f"Model {i}" for i in range(len(self.base_models))],
                'Correlation': correlations,
                'Weight': effective_weights
            })
        
        else:
            # Calculate more detailed contribution metrics
            contributions = []
            
            for i, model in enumerate(self.base_models):
                # Create a hypothetical ensemble without this model
                weights_without = self.weights.copy()
                weights_without[i] = 0
                
                if np.sum(weights_without) > 0:
                    # Normalize remaining weights
                    weights_without = weights_without / np.sum(weights_without)
                    
                    # Calculate predictions without this model
                    preds_without = np.sum(base_preds * weights_without.reshape(1, -1), axis=1)
                    
                    # For detailed contributions, we can't calculate actual performance metrics
                    # without true labels, so we'll use correlation as a proxy for contribution
                    corr_with = np.corrcoef(base_preds[:, i], ensemble_preds)[0, 1]
                    # Use weight as a proxy for contribution
                    perf_drop = self.weights[i]
                else:
                    perf_drop = 1.0  # If only one model, removing it drops performance to baseline
                
                contributions.append({
                    'Model': f"Model {i}",
                    'Weight': self.weights[i],
                    'Correlation': np.corrcoef(base_preds[:, i], ensemble_preds)[0, 1],
                    'Unique_Contribution': perf_drop
                })
            
            return pd.DataFrame(contributions)
    
    def plot_ensemble_weights(self, figsize=(10, 6)):
        """
        Plot the weights of each base model in the ensemble.
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 6)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot weights
        x = range(len(self.base_models))
        plt.bar(x, self.weights)
        plt.xticks(x, [f"Model {i}" for i in range(len(self.base_models))])
        plt.xlabel('Base Models')
        plt.ylabel('Weight')
        plt.title(f'Ensemble Weights ({self.ensemble_method.capitalize()} Method)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_model_contributions(self, X, figsize=(12, 8)):
        """
        Plot the contribution of each base model to the ensemble.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get model contributions
        contributions = self.get_model_contributions(X, detailed=True)
        
        # Plot contributions
        plt.figure(figsize=figsize)
        
        # Create a grouped bar chart
        x = range(len(contributions))
        width = 0.3
        
        plt.bar(x, contributions['Weight'], width=width, label='Weight',
               color='skyblue', alpha=0.7)
        plt.bar([i + width for i in x], contributions['Correlation'], width=width,
               label='Correlation with Ensemble', color='orange', alpha=0.7)
        plt.bar([i + 2*width for i in x], contributions['Unique_Contribution'], width=width,
               label='Unique Contribution', color='green', alpha=0.7)
        
        plt.xticks([i + width for i in x], contributions['Model'])
        plt.xlabel('Base Models')
        plt.ylabel('Value')
        plt.title('Model Contributions to Ensemble')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_curve(self, X, y, n_bins=10, figsize=(10, 8)):
        """
        Plot calibration curve for base models and ensemble.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples,)
            Target values
        n_bins : int, default=10
            Number of bins for the calibration curve
        figsize : tuple, default=(10, 8)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        plt.figure(figsize=figsize)
        
        # Plot the perfectly calibrated line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Plot base models
        base_colors = plt.cm.tab10.colors
        for i, model in enumerate(self.base_models):
            # Get predictions
            y_pred = model.predict_proba(X)[:, 1]
            
            # Calculate calibration curve
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_pred, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_sums = np.bincount(bin_indices, weights=y_pred, minlength=n_bins)
            bin_counts = np.bincount(bin_indices, minlength=n_bins)
            bin_true = np.bincount(bin_indices, weights=y.astype(float), minlength=n_bins)
            
            nonzero = bin_counts > 0
            prob_pred = bin_sums[nonzero] / bin_counts[nonzero]
            prob_true = bin_true[nonzero] / bin_counts[nonzero]
            
            # Plot calibration curve
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Model {i}',
                    color=base_colors[i % len(base_colors)])
        
        # Plot ensemble model
        y_pred_ensemble = self.predict_proba(X)[:, 1]
        
        # Calculate calibration curve for ensemble
        bin_indices = np.digitize(y_pred_ensemble, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.bincount(bin_indices, weights=y_pred_ensemble, minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=y.astype(float), minlength=n_bins)
        
        nonzero = bin_counts > 0
        prob_pred = bin_sums[nonzero] / bin_counts[nonzero]
        prob_true = bin_true[nonzero] / bin_counts[nonzero]
        
        # Plot ensemble calibration curve
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Ensemble',
                color='red', linestyle='-')
        
        plt.title('Calibration Curves', fontsize=14)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('True Probability', fontsize=12)
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_dir='models'):
        """
        Save the ensemble model to disk.
        
        Parameters:
        -----------
        model_dir : str, default='models'
            Directory to save the model
            
        Returns:
        --------
        str
            Path to the saved model directory
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a subdirectory for this ensemble model
        model_path = os.path.join(model_dir, self.model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            'model_name': self.model_name,
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'optimize_metric': self.optimize_metric,
            'cv': self.cv,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'class_names': self.class_names,
            'n_base_models': len(self.base_models)
        }
        
        config_path = os.path.join(model_path, 'ensemble_config.joblib')
        joblib.dump(config, config_path)
        
        # Save base models
        base_models_dir = os.path.join(model_path, 'base_models')
        os.makedirs(base_models_dir, exist_ok=True)
        
        for i, model in enumerate(self.base_models):
            model_file = os.path.join(base_models_dir, f'base_model_{i}.joblib')
            joblib.dump(model, model_file)
        
        # Save meta-model if using stacking
        if self.ensemble_method == 'stacking' and hasattr(self, 'meta_model'):
            meta_model_path = os.path.join(model_path, 'meta_model.joblib')
            joblib.dump(self.meta_model, meta_model_path)
        
        logger.info(f"Ensemble model saved to {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load an ensemble model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model directory
            
        Returns:
        --------
        EnsembleModel
            Loaded ensemble model
        """
        # Check if model path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load ensemble configuration
        config_path = os.path.join(model_path, 'ensemble_config.joblib')
        config = joblib.load(config_path)
        
        # Load base models
        base_models_dir = os.path.join(model_path, 'base_models')
        n_models = config['n_base_models']
        
        base_models = []
        for i in range(n_models):
            model_file = os.path.join(base_models_dir, f'base_model_{i}.joblib')
            model = joblib.load(model_file)
            base_models.append(model)
        
        # Create ensemble instance
        ensemble = cls(
            base_models=base_models,
            ensemble_method=config['ensemble_method'],
            weights=config['weights'],
            optimize_metric=config['optimize_metric'],
            cv=config['cv'],
            model_name=config['model_name'],
            random_state=config['random_state']
        )
        
        # Load meta-model if needed
        if ensemble.ensemble_method == 'stacking':
            meta_model_path = os.path.join(model_path, 'meta_model.joblib')
            if os.path.exists(meta_model_path):
                ensemble.meta_model = joblib.load(meta_model_path)
        
        # Set attributes
        ensemble.feature_names = config.get('feature_names')
        ensemble.target_name = config.get('target_name')
        ensemble.class_names = config.get('class_names')
        ensemble.is_fitted = True
        
        logger.info(f"Ensemble model loaded from {model_path}")
        return ensemble


class TimeSeriesEnsemble(EnsembleModel):
    """
    Ensemble model that incorporates time-series elements for churn prediction.
    """
    
    def __init__(
        self,
        base_models=None,
        time_window=3,
        sequence_model=None,
        ensemble_method='stacking',
        decay_factor=0.85,
        optimize_metric='f1',
        model_name="TimeSeriesEnsemble",
        random_state=42
    ):
        """
        Initialize the time-series ensemble model.
        
        Parameters:
        -----------
        base_models : list, default=None
            List of base model instances to ensemble
        time_window : int, default=3
            Number of time periods to consider for forecasting
        sequence_model : object, default=None
            Model for sequence prediction. If None, SimpleRNN is used.
        ensemble_method : str, default='stacking'
            Method for combining base models
        decay_factor : float, default=0.85
            Weight decay for older predictions
        optimize_metric : str, default='f1'
            Metric to optimize
        model_name : str, default="TimeSeriesEnsemble"
            Name of the ensemble model
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(
            base_models=base_models,
            ensemble_method=ensemble_method,
            optimize_metric=optimize_metric,
            model_name=model_name,
            random_state=random_state
        )
        
        self.time_window = time_window
        self.sequence_model = sequence_model
        self.decay_factor = decay_factor
        
        # Initialize sequence model if needed
        if self.sequence_model is None:
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import SimpleRNN, Dense
                
                model = Sequential([
                    SimpleRNN(32, input_shape=(time_window, len(base_models) if base_models else 1)),
                    Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                self.sequence_model = model
            except ImportError:
                logger.warning("TensorFlow not available, using logistic regression for sequence model.")
                self.sequence_model = LogisticRegression(random_state=random_state)
    
    def fit(self, X, y, time_col=None, id_col=None, X_val=None, y_val=None):
        """
        Fit the time-series ensemble model.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Training data. If DataFrame, should contain time-series data.
        y : array-like
            Target values
        time_col : str, default=None
            Column name for time information. Required if X is a DataFrame.
        id_col : str, default=None
            Column name for customer IDs. Required if X is a DataFrame.
        X_val : array-like or DataFrame, default=None
            Validation data
        y_val : array-like, default=None
            Validation target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store feature and target names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        
        # Determine class names if possible
        if hasattr(y, 'unique'):
            self.class_names = sorted(y.unique())
        
        # Check if time-series data is provided
        if isinstance(X, pd.DataFrame) and time_col is not None and id_col is not None:
            # Time-series data processing
            self._fit_time_series(X, y, time_col, id_col, X_val, y_val)
        else:
            # Fall back to regular ensemble fitting
            logger.warning("Time-series data not properly specified. Falling back to regular ensemble.")
            super().fit(X, y, X_val, y_val)
        
        self.is_fitted = True
        return self
    
    def _fit_time_series(self, X, y, time_col, id_col, X_val=None, y_val=None):
        """
        Fit the model using time-series data.
        
        Parameters:
        -----------
        X : DataFrame
            Training data with time and ID columns
        y : array-like
            Target values
        time_col : str
            Column name for time information
        id_col : str
            Column name for customer IDs
        X_val : DataFrame, default=None
            Validation data
        y_val : array-like, default=None
            Validation target values
        """
        logger.info("Fitting time-series ensemble...")
        
        # Ensure X is sorted by ID and time
        X = X.sort_values([id_col, time_col])
        
        # Get unique time periods and customers
        times = X[time_col].unique()
        customers = X[id_col].unique()
        
        logger.info(f"Data spans {len(times)} time periods and {len(customers)} customers")
        
        # Train base models on each time period
        self.time_models = {}
        
        for t in times:
            # Get data for this time period
            X_t = X[X[time_col] == t].drop([time_col, id_col], axis=1)
            y_t = y.loc[X_t.index]
            
            # Train base models for this time period
            period_models = []
            
            for i, base_model_template in enumerate(self.base_models):
                # Clone the model template
                import copy
                model_copy = copy.deepcopy(base_model_template)
                
                # Fit on this time period
                model_copy.fit(X_t, y_t)
                period_models.append(model_copy)
            
            # Store models for this time period
            self.time_models[t] = period_models
        
        # Generate sequences for each customer
        X_sequences, y_sequences = self._generate_sequences(X, y, times, customers, time_col, id_col)
        
        # Train sequence model
        if hasattr(self.sequence_model, 'fit_generator'):
            # TensorFlow model
            from tensorflow.keras.callbacks import EarlyStopping
            
            self.sequence_model.fit(
                X_sequences, y_sequences,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
        else:
            # Reshape for sklearn models
            X_sequences_reshaped = X_sequences.reshape(X_sequences.shape[0], -1)
            self.sequence_model.fit(X_sequences_reshaped, y_sequences)
    
    def _generate_sequences(self, X, y, times, customers, time_col, id_col):
        """
        Generate time-series sequences for each customer.
        
        Parameters:
        -----------
        X : DataFrame
            Training data
        y : array-like
            Target values
        times : array-like
            Unique time periods
        customers : array-like
            Unique customer IDs
        time_col : str
            Column name for time information
        id_col : str
            Column name for customer IDs
            
        Returns:
        --------
        tuple
            (X_sequences, y_sequences)
        """
        # Initialize sequences
        sequences = []
        labels = []
        
        # Generate sequences for each customer
        for customer in customers:
            # Get data for this customer
            customer_data = X[X[id_col] == customer].sort_values(time_col)
            customer_y = y.loc[customer_data.index]
            
            # Skip if not enough time periods
            if len(customer_data) < self.time_window + 1:
                continue
            
            # Generate predictions for each time period
            customer_preds = np.zeros((len(customer_data), len(self.base_models)))
            
            for i, t in enumerate(customer_data[time_col]):
                if t in self.time_models:
                    # Get models for this time period
                    period_models = self.time_models[t]
                    
                    # Get features for this customer at this time
                    X_ct = customer_data.iloc[[i]].drop([time_col, id_col], axis=1)
                    
                    # Generate predictions
                    for j, model in enumerate(period_models):
                        customer_preds[i, j] = model.predict_proba(X_ct)[0, 1]
            
            # Create sequences
            for i in range(len(customer_data) - self.time_window):
                # Extract window
                window = customer_preds[i:i+self.time_window]
                
                # Get label (next time period)
                label = customer_y.iloc[i+self.time_window]
                
                sequences.append(window)
                labels.append(label)
        
        # Convert to arrays
        X_sequences = np.array(sequences)
        y_sequences = np.array(labels)
        
        return X_sequences, y_sequences
    
    def predict_proba(self, X, time_col=None, id_col=None):
        """
        Predict class probabilities considering time-series information if available.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Input data
        time_col : str, default=None
            Column name for time information
        id_col : str, default=None
            Column name for customer IDs
            
        Returns:
        --------
        array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Check if time-series prediction is possible
        if (isinstance(X, pd.DataFrame) and time_col is not None and id_col is not None 
            and hasattr(self, 'time_models')):
            # Use time-series prediction
            return self._predict_proba_time_series(X, time_col, id_col)
        else:
            # Fall back to regular ensemble prediction
            logger.warning("Time-series prediction not possible. Using regular ensemble.")
            return super().predict_proba(X)
    
    def _predict_proba_time_series(self, X, time_col, id_col):
        """
        Predict class probabilities using time-series information.
        
        Parameters:
        -----------
        X : DataFrame
            Input data with time and ID columns
        time_col : str
            Column name for time information
        id_col : str
            Column name for customer IDs
            
        Returns:
        --------
        array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        # Ensure X is sorted by ID and time
        X = X.sort_values([id_col, time_col])
        
        # Get unique customers
        customers = X[id_col].unique()
        
        # Initialize predictions
        all_probs = np.zeros((len(X), 2))
        
        # Generate predictions for each customer
        for customer in customers:
            # Get data for this customer
            customer_data = X[X[id_col] == customer].sort_values(time_col)
            
            # Skip if not enough time periods
            if len(customer_data) < self.time_window:
                # Use regular ensemble for customers with insufficient history
                customer_X = customer_data.drop([time_col, id_col], axis=1)
                customer_probs = super().predict_proba(customer_X)
                all_probs[customer_data.index.to_numpy()] = customer_probs
                continue
            
            # Generate base model predictions for each time period
            customer_preds = np.zeros((len(customer_data), len(self.base_models)))
            
            for i, _ in enumerate(customer_data.index):
                # Get features for this customer at this time
                X_ct = customer_data.iloc[[i]].drop([time_col, id_col], axis=1)
                
                # Generate predictions from all base models
                for j, model in enumerate(self.base_models):
                    customer_preds[i, j] = model.predict_proba(X_ct)[0, 1]
            
            # Generate sequence predictions
            for i in range(len(customer_data) - self.time_window + 1):
                # Extract window
                window = customer_preds[i:i+self.time_window]
                window = window.reshape(1, self.time_window, -1)
                
                # Make prediction using sequence model
                if hasattr(self.sequence_model, 'predict'):
                    if hasattr(self.sequence_model, 'predict_on_batch'):
                        # TensorFlow model
                        prob = self.sequence_model.predict(window)[0, 0]
                    else:
                        # Reshape for sklearn models
                        window_reshaped = window.reshape(1, -1)
                        prob = self.sequence_model.predict_proba(window_reshaped)[0, 1]
                    
                    # Store prediction
                    idx = customer_data.index[i+self.time_window-1]
                    all_probs[X.index.get_loc(idx)] = [1-prob, prob]
            
            # For the first time_window-1 periods, use regular ensemble
            if self.time_window > 1:
                early_idx = customer_data.index[:self.time_window-1]
                early_X = customer_data.loc[early_idx].drop([time_col, id_col], axis=1)
                early_probs = super().predict_proba(early_X)
                for i, idx in enumerate(early_idx):
                    all_probs[X.index.get_loc(idx)] = early_probs[i]
        
        return all_probs
    
    def forecast_churn_probability(self, X, time_col, id_col, future_periods=3):
        """
        Forecast churn probability for future periods.
        
        Parameters:
        -----------
        X : DataFrame
            Historical data with time and ID columns
        time_col : str
            Column name for time information
        id_col : str
            Column name for customer IDs
        future_periods : int, default=3
            Number of future periods to forecast
            
        Returns:
        --------
        DataFrame
            Forecasted churn probabilities for each customer and time period
        """
        if not self.is_fitted or not hasattr(self, 'time_models'):
            raise ValueError("Model must be fitted with time-series data.")
        
        # Ensure X is sorted by ID and time
        X = X.sort_values([id_col, time_col])
        
        # Get unique customers and time periods
        customers = X[id_col].unique()
        times = X[time_col].unique()
        
        # Get the last time period
        last_time = times[-1]
        
        # Determine future time periods
        if isinstance(last_time, (int, float)):
            future_times = np.array([last_time + i + 1 for i in range(future_periods)])
        elif isinstance(last_time, pd.Timestamp):
            # Assume monthly data if datetime
            future_times = np.array([last_time + pd.DateOffset(months=i+1) for i in range(future_periods)])
        else:
            # Use integer indices
            future_times = np.array([i+1 for i in range(future_periods)])
        
        # Initialize forecast dataframe
        forecast_data = []
        
        # Generate forecasts for each customer
        for customer in customers:
            # Get data for this customer
            customer_data = X[X[id_col] == customer].sort_values(time_col)
            
            # Skip if not enough time periods
            if len(customer_data) < self.time_window:
                continue
            
            # Generate base model predictions for historical data
            customer_preds = np.zeros((len(customer_data), len(self.base_models)))
            
            for i, _ in enumerate(customer_data.index):
                # Get features for this customer at this time
                X_ct = customer_data.iloc[[i]].drop([time_col, id_col], axis=1)
                
                # Generate predictions from all base models
                for j, model in enumerate(self.base_models):
                    customer_preds[i, j] = model.predict_proba(X_ct)[0, 1]
            
            # Get the most recent window
            latest_window = customer_preds[-self.time_window:]
            
            # Forecast for future periods
            forecasted_probs = np.zeros(future_periods)
            
            for i in range(future_periods):
                # Use the current window for prediction
                window = latest_window.reshape(1, self.time_window, -1)
                
                # Make prediction using sequence model
                if hasattr(self.sequence_model, 'predict'):
                    if hasattr(self.sequence_model, 'predict_on_batch'):
                        # TensorFlow model
                        prob = self.sequence_model.predict(window)[0, 0]
                    else:
                        # Reshape for sklearn models
                        window_reshaped = window.reshape(1, -1)
                        prob = self.sequence_model.predict_proba(window_reshaped)[0, 1]
                    
                    # Store prediction
                    forecasted_probs[i] = prob
                    
                    # Update window for next forecast
                    if i < future_periods - 1:
                        # Shift window and add new prediction
                        latest_window = np.vstack([latest_window[1:], customer_preds[-1:]])
                
                # Store forecast data
                forecast_data.append({
                    id_col: customer,
                    time_col: future_times[i],
                    'Forecasted_Churn_Probability': forecasted_probs[i],
                    'Period': i + 1
                })
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame(forecast_data)
        
        return forecast_df
    
    def plot_churn_forecast(self, X, time_col, id_col, future_periods=3, n_customers=10, figsize=(15, 10)):
        """
        Plot churn probability forecasts for selected customers.
        
        Parameters:
        -----------
        X : DataFrame
            Historical data with time and ID columns
        time_col : str
            Column name for time information
        id_col : str
            Column name for customer IDs
        future_periods : int, default=3
            Number of future periods to forecast
        n_customers : int, default=10
            Number of customers to plot
        figsize : tuple, default=(15, 10)
            Figure size
        """
        # Get forecasts
        forecasts = self.forecast_churn_probability(X, time_col, id_col, future_periods)
        
        # Get unique customers and times
        customers = X[id_col].unique()
        times = X[time_col].unique()
        future_times = forecasts[time_col].unique()
        
        # Sample customers
        sample_customers = np.random.choice(customers, min(n_customers, len(customers)), replace=False)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Historical data
        for i, customer in enumerate(sample_customers):
            # Get historical data
            customer_data = X[X[id_col] == customer].sort_values(time_col)
            
            # Skip if not enough data
            if len(customer_data) < self.time_window:
                continue
            
            # Generate predictions for historical data
            historical_probs = []
            
            for _, row in customer_data.iterrows():
                X_ct = pd.DataFrame([row]).drop([time_col, id_col], axis=1)
                prob = super().predict_proba(X_ct)[0, 1]
                historical_probs.append(prob)
            
            # Get forecast data
            customer_forecast = forecasts[forecasts[id_col] == customer]
            forecast_times = customer_forecast[time_col].values
            forecast_probs = customer_forecast['Forecasted_Churn_Probability'].values
            
            # Plot historical and forecasted probabilities
            plt.plot(customer_data[time_col], historical_probs, marker='o', 
                    label=f"Customer {customer} (Historical)")
            plt.plot(forecast_times, forecast_probs, marker='s', linestyle='--',
                    label=f"Customer {customer} (Forecast)")
        
        # Add vertical line to separate historical and forecasted data
        plt.axvline(x=times[-1], color='black', linestyle='-', alpha=0.3)
        plt.text(times[-1], 1.01, 'Historical | Forecast', ha='center', va='bottom')
        
        # Customize plot
        plt.title('Churn Probability Forecast', fontsize=14)
        plt.xlabel(time_col, fontsize=12)
        plt.ylabel('Churn Probability', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()