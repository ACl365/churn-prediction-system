"""
Base Model Module for Telecom Customer Churn Prediction

This module defines the abstract base class for churn prediction models.
All model implementations should inherit from this base class to ensure
consistency in the model development process.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class BaseModel(ABC):
    """
    Abstract base class for churn prediction models.
    
    This class defines the interface that all model implementations must follow,
    and provides common functionality for model evaluation and visualization.
    """
    
    def __init__(self, model_name, random_state=42):
        """
        Initialize the base model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.target_name = None
        self.class_names = None
        self.is_fitted = False
    
    @abstractmethod
    def build(self):
        """
        Build the model with specified hyperparameters.
        This method needs to be implemented by each model subclass.
        """
        pass
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the training data.
        This method needs to be implemented by each model subclass.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        """
        pass
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction
            
        Returns:
        --------
        array-like, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the model performance on the given data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for evaluation
        y : array-like, shape (n_samples,)
            True target values
        threshold : float, default=0.5
            Threshold for converting probabilities to binary predictions
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get predictions
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
        }
        
        return metrics
    
    def plot_evaluation(self, X, y, threshold=0.5, figsize=(18, 12)):
        """
        Plot evaluation metrics and visualizations.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for evaluation
        y : array-like, shape (n_samples,)
            True target values
        threshold : float, default=0.5
            Threshold for converting probabilities to binary predictions
        figsize : tuple, default=(18, 12)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get predictions and metrics
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = self.evaluate(X, y, threshold)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=self.class_names if self.class_names else ['Negative', 'Positive'],
            yticklabels=self.class_names if self.class_names else ['Negative', 'Positive'],
            ax=axes[0, 0]
        )
        axes[0, 0].set_title(f"Confusion Matrix (Threshold={threshold})")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Receiver Operating Characteristic')
        axes[0, 1].legend(loc="lower right")
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        avg_precision = average_precision_score(y, y_pred_proba)
        
        axes[1, 0].plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend(loc="upper right")
        
        # Distribution of Predicted Probabilities
        axes[1, 1].hist(
            [y_pred_proba[y == 0], y_pred_proba[y == 1]], 
            bins=20, 
            alpha=0.5, 
            color=['blue', 'red'], 
            label=[
                f"Non-Churners (n={sum(y == 0)})", 
                f"Churners (n={sum(y == 1)})"
            ]
        )
        axes[1, 1].axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
        axes[1, 1].set_title('Distribution of Predicted Probabilities')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        # Add metrics as text
        plt.figtext(
            0.5, 0.01, 
            f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}",
            ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5)
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"{self.model_name} Evaluation", fontsize=16, y=0.98)
        plt.show()
        
    def plot_feature_importance(self, top_n=20, figsize=(12, 10)):
        """
        Plot feature importance for models that support it.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(12, 10)
            Figure size
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for plotting
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
            
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f"Top {top_n} Feature Importance - {self.model_name}")
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        model_dir : str, default='models'
            Directory to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'class_names': self.class_names,
            'random_state': self.random_state
        }
        
        metadata_path = os.path.join(model_dir, f"{self.model_name}_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path, metadata_path=None):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        metadata_path : str, default=None
            Path to the saved metadata. If None, will try to infer from model_path.
            
        Returns:
        --------
        BaseModel
            Loaded model instance
        """
        # Load model
        model = joblib.load(model_path)
        
        # Determine metadata path if not provided
        if metadata_path is None:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).split('.')[0]
            metadata_path = os.path.join(model_dir, f"{model_name}_metadata.joblib")
        
        # Load metadata if it exists
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            
            # Create an instance of the class
            instance = cls(model_name=metadata['model_name'], random_state=metadata['random_state'])
            
            # Set attributes
            instance.model = model
            instance.feature_names = metadata['feature_names']
            instance.target_name = metadata['target_name']
            instance.class_names = metadata['class_names']
            instance.is_fitted = True
            
            return instance
        else:
            # If no metadata, create a minimal instance
            instance = cls(model_name=os.path.basename(model_path).split('.')[0])
            instance.model = model
            instance.is_fitted = True
            
            return instance


class ModelFactory:
    """
    Factory class for creating churn prediction models.
    """
    
    @staticmethod
    def get_model(model_type, **kwargs):
        """
        Get a model instance of the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('gradient_boosting', 'neural_network', etc.)
        **kwargs : dict
            Additional arguments to pass to the model constructor
            
        Returns:
        --------
        BaseModel
            Instance of the specified model type
        """
        if model_type == 'gradient_boosting':
            from gradient_boosting import GradientBoostingModel
            return GradientBoostingModel(**kwargs)
        elif model_type == 'neural_network':
            from neural_network import NeuralNetworkModel
            return NeuralNetworkModel(**kwargs)
        elif model_type == 'random_forest':
            from random_forest import RandomForestModel
            return RandomForestModel(**kwargs)
        elif model_type == 'logistic_regression':
            from logistic_regression import LogisticRegressionModel
            return LogisticRegressionModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
