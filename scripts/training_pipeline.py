"""
Training Pipeline Module for Telecom Customer Churn Prediction

This module provides functions and classes for training, tuning, and evaluating
churn prediction models with proper handling of class imbalance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, accuracy_score, precision_score,
    recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from neural_network import NeuralNetworkModel
import joblib
import os
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('training_pipeline')


class ModelTrainer:
    """
    Class for training, tuning, and evaluating churn prediction models.
    """
    
    def __init__(
        self,
        model,
        random_state=42,
        resampling_strategy=None,
        resampling_ratio=0.5,
        model_dir='models'
    ):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        model : BaseModel
            Model instance to train
        random_state : int, default=42
            Random seed for reproducibility
        resampling_strategy : str, default=None
            Strategy for handling class imbalance. Options:
            - None: No resampling
            - 'smote': SMOTE oversampling
            - 'undersample': Random undersampling
            - 'hybrid': Combination of undersampling and SMOTE
        resampling_ratio : float, default=0.5
            Target ratio of minority to majority class after resampling
        model_dir : str, default='models'
            Directory to save trained models
        """
        self.model = model
        self.random_state = random_state
        self.resampling_strategy = resampling_strategy
        self.resampling_ratio = resampling_ratio
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize training history
        self.training_history = {}
    
    def preprocess_data(self, X, y, test_size=0.2, stratify=True):
        """
        Preprocess the data and split into train and test sets.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        test_size : float, default=0.2
            Proportion of data to use for testing
        stratify : bool, default=True
            Whether to use stratified sampling for train-test split
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Perform train-test split
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        # Apply resampling if needed
        if self.resampling_strategy:
            X_train, y_train = self._apply_resampling(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def _apply_resampling(self, X, y):
        """
        Apply resampling strategy to handle class imbalance.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
            
        Returns:
        --------
        tuple
            (X_resampled, y_resampled)
        """
        logger.info(f"Applying resampling strategy: {self.resampling_strategy}")
        
        # Get class counts
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Calculate sampling strategy
        if len(unique) == 2:
            minority_class = unique[np.argmin(counts)]
            majority_class = unique[np.argmax(counts)]
            
            if self.resampling_strategy == 'smote':
                # SMOTE oversampling
                resampler = SMOTE(
                    sampling_strategy=self.resampling_ratio,
                    random_state=self.random_state
                )
                X_resampled, y_resampled = resampler.fit_resample(X, y)
                
            elif self.resampling_strategy == 'undersample':
                # Random undersampling
                resampler = RandomUnderSampler(
                    sampling_strategy=self.resampling_ratio,
                    random_state=self.random_state
                )
                X_resampled, y_resampled = resampler.fit_resample(X, y)
                
            elif self.resampling_strategy == 'hybrid':
                # First undersample, then apply SMOTE
                undersampler = RandomUnderSampler(
                    sampling_strategy=0.3,  # Less aggressive undersampling
                    random_state=self.random_state
                )
                X_temp, y_temp = undersampler.fit_resample(X, y)
                
                # Then apply SMOTE
                oversampler = SMOTE(
                    sampling_strategy=self.resampling_ratio,
                    random_state=self.random_state
                )
                X_resampled, y_resampled = oversampler.fit_resample(X_temp, y_temp)
            else:
                raise ValueError(f"Unsupported resampling strategy: {self.resampling_strategy}")
            
            # Log new class distribution
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampled class distribution: {dict(zip(unique_new, counts_new))}")
            
            return X_resampled, y_resampled
        else:
            logger.warning("Resampling requires binary classification. Returning original data.")
            return X, y
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model on the given data.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training feature matrix
        y_train : array-like, shape (n_samples,)
            Training target vector
        X_val : array-like, default=None
            Validation feature matrix
        y_val : array-like, default=None
            Validation target vector
        **kwargs : dict
            Additional arguments to pass to the model's fit method
            
        Returns:
        --------
        BaseModel
            Trained model instance
        """
        logger.info(f"Training {self.model.model_name}...")
        start_time = time.time()
        
        # Build the model if not already built
        if not hasattr(self.model, 'model') or self.model.model is None:
            # For neural network models, we need to set the input dimension
            if hasattr(self.model, 'input_dim') and self.model.input_dim is None:
                input_dim = X_train.shape[1]
                self.model.build(input_dim=input_dim)
            else:
                self.model.build()
        
        # Prepare evaluation set for early stopping if available
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Fit the model - handle different model types
        if isinstance(self.model, NeuralNetworkModel):
            # Neural network models don't use eval_set
            self.model.fit(X_train, y_train, **kwargs)
        else:
            # Gradient boosting models use eval_set
            self.model.fit(X_train, y_train, eval_set=eval_set, **kwargs)
        
        # Record training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Store training history
        self.training_history['training_time'] = training_time
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, threshold=0.5):
        """
        Evaluate the trained model on test data.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like, shape (n_samples,)
            Test target vector
        threshold : float, default=0.5
            Threshold for binary classification
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {self.model.model_name}...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Check if there's only one class in the target
        unique_classes = np.unique(y_test)
        
        # Initialize metrics dictionary
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'threshold': threshold
        }
        
        # Only calculate AUC and average precision if there are multiple classes
        if len(unique_classes) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)
        else:
            logger.warning("Only one class present in y_test. ROC AUC and Average Precision are not defined.")
            metrics['auc'] = None
            metrics['average_precision'] = None
        
        # Log metrics
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Store evaluation results in history
        self.training_history['evaluation_metrics'] = metrics
        
        return metrics
    
    def find_optimal_threshold(self, X_test, y_test, metric='f1', plot=True):
        """
        Find the optimal threshold for binary classification.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like, shape (n_samples,)
            Test target vector
        metric : str, default='f1'
            Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        plot : bool, default=True
            Whether to plot the threshold analysis
            
        Returns:
        --------
        float
            Optimal threshold value
        """
        logger.info(f"Finding optimal threshold for {metric} score...")
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Try different thresholds and compute metrics
        thresholds = np.arange(0.01, 1.0, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_test, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_test, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_test, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            scores.append(score)
        
        # Find the optimal threshold
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.2f} with {metric} = {optimal_score:.4f}")
        
        # Plot threshold analysis if requested
        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(thresholds, scores)
            plt.vlines(optimal_threshold, 0, 1, linestyle='--', color='r', label=f'Optimal threshold = {optimal_threshold:.2f}')
            plt.title(f'Threshold vs {metric.capitalize()} Score')
            plt.xlabel('Threshold')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        # Store in history
        self.training_history['optimal_threshold'] = {
            'value': optimal_threshold,
            'metric': metric,
            'score': optimal_score
        }
        
        return optimal_threshold
    
    def plot_confusion_matrix(self, X_test, y_test, threshold=0.5, figsize=(10, 8)):
        """
        Plot confusion matrix for the model.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like, shape (n_samples,)
            Test target vector
        threshold : float, default=0.5
            Threshold for binary classification
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        array-like
            Confusion matrix values
        """
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate and plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Negative (No Churn)', 'Positive (Churn)'],
            yticklabels=['Negative (No Churn)', 'Positive (Churn)']
        )
        plt.title(f"Confusion Matrix (Threshold={threshold:.2f})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
        
        # Calculate and log classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        logger.info(f"Classification Report:\n{pd.DataFrame(report).transpose()}")
        
        # Return confusion matrix values for further analysis
        return cm
    
    def plot_roc_curve(self, X_test, y_test, figsize=(10, 8)):
        """
        Plot ROC curve for the model.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like, shape (n_samples,)
            Test target vector
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        tuple
            (fpr, tpr, thresholds, auc_score)
        """
        # Check if there's only one class in the target
        unique_classes = np.unique(y_test)
        if len(unique_classes) <= 1:
            logger.warning("Only one class present in y_test. ROC curve cannot be computed.")
            return None, None, None, None
            
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Plot ROC curve
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.fill_between(fpr, tpr, alpha=0.3)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fpr, tpr, thresholds, auc_score
    
    def plot_precision_recall_curve(self, X_test, y_test, figsize=(10, 8)):
        """
        Plot Precision-Recall curve for the model.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like, shape (n_samples,)
            Test target vector
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        tuple
            (precision, recall, thresholds, average_precision)
        """
        # Check if there's only one class in the target
        unique_classes = np.unique(y_test)
        if len(unique_classes) <= 1:
            logger.warning("Only one class present in y_test. Precision-Recall curve cannot be computed.")
            return None, None, None, None
            
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate Precision-Recall curve and Average Precision
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
        plt.axhline(y=sum(y_test) / len(y_test), color='r', linestyle='--', 
                   label=f'Baseline (Prevalence = {sum(y_test) / len(y_test):.3f})')
        plt.fill_between(recall, precision, alpha=0.3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return precision, recall, thresholds, avg_precision
    
    def plot_probability_distribution(self, X_test, y_test, figsize=(10, 8)):
        """
        Plot the distribution of predicted probabilities by class.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test feature matrix
        y_test : array-like, shape (n_samples,)
            Test target vector
        figsize : tuple, default=(10, 8)
            Figure size
        """
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Plot distributions
        plt.figure(figsize=figsize)
        
        # Plot density for each class
        for target_class in np.unique(y_test):
            sns.kdeplot(
                y_pred_proba[y_test == target_class],
                label=f'Class {target_class}',
                fill=True,
                alpha=0.5
            )
        
        plt.title('Distribution of Predicted Probabilities by Class')
        plt.xlabel('Predicted Probability of Class 1')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def cross_validate(self, X, y, cv=5, scoring='roc_auc'):
        """
        Perform cross-validation to evaluate model robustness.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='roc_auc'
            Scoring metric to use
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation with {scoring} scoring...")
        
        # Define the model building and fitting function
        def build_and_fit(X_train, y_train):
            model_copy = type(self.model)(
                model_name=self.model.model_name,
                random_state=self.random_state
            )
            
            # For neural network models, we need to set the input dimension
            if isinstance(model_copy, NeuralNetworkModel):
                input_dim = X_train.shape[1]
                model_copy.build(input_dim=input_dim)
            else:
                model_copy.build()
                
            model_copy.fit(X_train, y_train)
            return model_copy
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_scores = []
        fold_models = []
        
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {i+1}/{cv}...")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Apply resampling if needed (only to training data)
            if self.resampling_strategy:
                X_train, y_train = self._apply_resampling(X_train, y_train)
            
            # Build and fit the model
            fold_model = build_and_fit(X_train, y_train)
            fold_models.append(fold_model)
            
            # Evaluate the model
            if scoring == 'roc_auc':
                # Check if there's only one class in the test set
                if len(np.unique(y_test)) <= 1:
                    logger.warning(f"Fold {i+1}/{cv}: Only one class in test set. Skipping ROC AUC calculation.")
                    continue
                    
                y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred_proba)
            else:
                y_pred = fold_model.predict(X_test)
                if scoring == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif scoring == 'precision':
                    score = precision_score(y_test, y_pred, zero_division=0)
                elif scoring == 'recall':
                    score = recall_score(y_test, y_pred, zero_division=0)
                elif scoring == 'f1':
                    score = f1_score(y_test, y_pred, zero_division=0)
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring}")
            
            cv_scores.append(score)
            logger.info(f"Fold {i+1}/{cv} {scoring}: {score:.4f}")
        
        # Calculate overall statistics
        if cv_scores:
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            logger.info(f"Cross-validation results ({scoring}): {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Store in history
            self.training_history['cross_validation'] = {
                'scores': cv_scores,
                'mean': cv_mean,
                'std': cv_std,
                'scoring': scoring
            }
            
            return {
                'scores': cv_scores,
                'mean': cv_mean,
                'std': cv_std,
                'models': fold_models
            }
        else:
            logger.warning("No valid cross-validation scores. Check if data has multiple classes.")
            return {
                'scores': [],
                'mean': None,
                'std': None,
                'models': fold_models
            }
    
    def save_training_history(self, filename=None):
        """
        Save the training history to disk.
        
        Parameters:
        -----------
        filename : str, default=None
            Filename to save the history. If None, a default name is generated.
            
        Returns:
        --------
        str
            Path to the saved history file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model.model_name}_history_{timestamp}.joblib"
        
        # Add metadata
        self.training_history['model_name'] = self.model.model_name
        self.training_history['timestamp'] = datetime.now().isoformat()
        self.training_history['resampling_strategy'] = self.resampling_strategy
        
        # Save history
        history_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.training_history, history_path)
        
        logger.info(f"Training history saved to {history_path}")
        return history_path
    
    def run_training_pipeline(
        self, 
        X, 
        y, 
        test_size=0.2,
        tune_hyperparameters=False,
        tune_threshold=True,
        cross_validate=True,
        cv=5,
        save_model=True,
        save_history=True,
        **kwargs
    ):
        """
        Run the complete training pipeline.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        test_size : float, default=0.2
            Proportion of data to use for testing
        tune_hyperparameters : bool, default=False
            Whether to tune hyperparameters
        tune_threshold : bool, default=True
            Whether to find the optimal classification threshold
        cross_validate : bool, default=True
            Whether to perform cross-validation
        cv : int, default=5
            Number of cross-validation folds
        save_model : bool, default=True
            Whether to save the trained model to disk
        save_history : bool, default=True
            Whether to save the training history to disk
        **kwargs : dict
            Additional arguments to pass to model training methods
            
        Returns:
        --------
        dict
            Training pipeline results summary
        """
        # Start time tracking
        start_time = time.time()
        logger.info(f"Starting training pipeline for {self.model.model_name}...")
        
        # Initialize results dictionary
        results = {}
        
        # Step 1: Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y, test_size=test_size)
        
        # Step 2: Tune hyperparameters if requested
        if tune_hyperparameters and hasattr(self.model, 'tune_hyperparameters'):
            logger.info("Tuning hyperparameters...")
            self.model.tune_hyperparameters(X_train, y_train, **kwargs.get('tune_params', {}))
        
        # Step 3: Train the model
        logger.info("Training the model...")
        self.train_model(X_train, y_train, **kwargs.get('train_params', {}))
        
        # Step 4: Evaluate on test set
        logger.info("Evaluating on test set...")
        metrics = self.evaluate_model(X_test, y_test)
        results['metrics'] = metrics
        
        # Step 5: Tune threshold if requested
        if tune_threshold:
            logger.info("Finding optimal threshold...")
            optimal_threshold = self.find_optimal_threshold(
                X_test, y_test, 
                metric=kwargs.get('threshold_metric', 'f1')
            )
            results['optimal_threshold'] = optimal_threshold
            
            # Re-evaluate with optimal threshold
            updated_metrics = self.evaluate_model(X_test, y_test, threshold=optimal_threshold)
            results['metrics_optimal_threshold'] = updated_metrics
        
        # Step 6: Cross-validation if requested
        if cross_validate:
            logger.info("Performing cross-validation...")
            cv_results = self.cross_validate(
                X, y, cv=cv, 
                scoring=kwargs.get('cv_scoring', 'roc_auc')
            )
            results['cross_validation'] = {
                'mean': cv_results['mean'],
                'std': cv_results['std']
            }
        
        # Step 7: Plot visualizations
        logger.info("Creating evaluation visualizations...")
        
        # Confusion matrix
        if kwargs.get('plot_cm', True):
            threshold = optimal_threshold if tune_threshold else 0.5
            cm = self.plot_confusion_matrix(X_test, y_test, threshold=threshold)
            results['confusion_matrix'] = cm
        
        # ROC curve
        if kwargs.get('plot_roc', True):
            # Check if there are multiple classes in the test set
            if len(np.unique(y_test)) > 1:
                roc_results = self.plot_roc_curve(X_test, y_test)
                if roc_results[3] is not None:  # If AUC was calculated
                    results['roc_auc'] = roc_results[3]
        
        # Precision-Recall curve
        if kwargs.get('plot_pr', True):
            # Check if there are multiple classes in the test set
            if len(np.unique(y_test)) > 1:
                pr_results = self.plot_precision_recall_curve(X_test, y_test)
                if pr_results[3] is not None:  # If average precision was calculated
                    results['avg_precision'] = pr_results[3]
        
        # Probability distribution
        if kwargs.get('plot_prob_dist', True):
            self.plot_probability_distribution(X_test, y_test)
        
        # Step 8: Plot feature importance if available
        if kwargs.get('plot_importance', True) and hasattr(self.model, 'plot_feature_importance'):
            logger.info("Plotting feature importance...")
            try:
                # Check if the plot_feature_importance method requires X and y parameters
                import inspect
                sig = inspect.signature(self.model.plot_feature_importance)
                if 'X' in sig.parameters and 'y' in sig.parameters:
                    importance_df = self.model.plot_feature_importance(
                        X=X_test,
                        y=y_test,
                        top_n=kwargs.get('importance_top_n', 20)
                    )
                elif 'X' in sig.parameters:
                    importance_df = self.model.plot_feature_importance(
                        X=X_test,
                        top_n=kwargs.get('importance_top_n', 20)
                    )
                else:
                    importance_df = self.model.plot_feature_importance(
                        top_n=kwargs.get('importance_top_n', 20)
                    )
                results['feature_importance'] = importance_df
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"Could not plot feature importance: {e}")
        
        # Step 9: Save model if requested
        if save_model:
            logger.info("Saving model...")
            model_path = self.model.save_model(model_dir=self.model_dir)
            results['model_path'] = model_path
        
        # Step 10: Save training history if requested
        if save_history:
            logger.info("Saving training history...")
            history_path = self.save_training_history()
            results['history_path'] = history_path
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Training pipeline completed in {total_time:.2f} seconds")
        
        # Add timing info to results
        results['total_time'] = total_time
        
        return results


def compare_models(trainers, X, y, test_size=0.2, metrics=None, plot=True, figsize=(12, 8)):
    """
    Compare multiple models on the same dataset.
    
    Parameters:
    -----------
    trainers : list
        List of ModelTrainer instances
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target vector
    test_size : float, default=0.2
        Proportion of data to use for testing
    metrics : list, default=None
        List of metrics to compare. If None, defaults to ['accuracy', 'precision', 'recall', 'f1', 'auc']
    plot : bool, default=True
        Whether to plot the comparison
    figsize : tuple, default=(12, 8)
        Figure size for the plot
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with comparison results
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Split data or use the entire dataset if test_size is 0
    if test_size <= 0:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=trainers[0].random_state, stratify=y
        )
    
    # Initialize results
    results = []
    
    # Train and evaluate each model
    for trainer in trainers:
        logger.info(f"Training and evaluating {trainer.model.model_name}...")
        
        # Apply resampling if needed
        if trainer.resampling_strategy:
            X_train_resampled, y_train_resampled = trainer._apply_resampling(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # Train model
        trainer.train_model(X_train_resampled, y_train_resampled)
        
        # Evaluate model
        model_metrics = trainer.evaluate_model(X_test, y_test)
        
        # Store results
        results.append({
            'model_name': trainer.model.model_name,
            **{metric: model_metrics[metric] for metric in metrics if metric in model_metrics}
        })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(results)
    
    # Transpose the dataframe so metrics are rows and model names are columns
    comparison_df = comparison_df.set_index('model_name').T
    
    # Plot comparison if requested
    if plot:
        plt.figure(figsize=figsize)
        
        # Reset index to make the metrics a column
        plot_df = comparison_df.reset_index()
        
        # Rename the index column to 'Metric'
        plot_df = plot_df.rename(columns={'index': 'Metric'})
        
        # Melt the dataframe for easier plotting
        plot_df = plot_df.melt(
            id_vars='Metric',
            value_name='Value',
            var_name='Model'
        )
        
        # Create grouped bar chart
        sns.barplot(x='Metric', y='Value', hue='Model', data=plot_df)
        plt.title('Model Comparison on Key Metrics')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Model Name')
        plt.tight_layout()
        plt.show()
    
    return comparison_df