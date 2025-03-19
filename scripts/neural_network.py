"""
Neural Network Model Module for Telecom Customer Churn Prediction

This module implements neural network models for tabular data churn prediction
using TensorFlow/Keras.
"""

import pandas as pd
import numpy as np
from base_model import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os


class NeuralNetworkModel(BaseModel):
    """
    Neural Network model for churn prediction.
    Uses TensorFlow/Keras for implementation.
    """
    
    def __init__(
        self,
        model_name="NeuralNetworkModel",
        hidden_layers=None,
        activations=None,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        early_stopping_patience=10,
        class_weight="balanced",
        random_state=42
    ):
        """
        Initialize the Neural Network model.
        
        Parameters:
        -----------
        model_name : str, default="NeuralNetworkModel"
            Name of the model
        hidden_layers : list of int, default=None
            Sizes of hidden layers. If None, defaults to [64, 32, 16]
        activations : str or list of str, default=None
            Activation functions for hidden layers. If None, defaults to 'relu' for all layers
        dropout_rate : float, default=0.3
            Dropout rate for regularization
        learning_rate : float, default=0.001
            Learning rate for optimizer
        batch_size : int, default=64
            Batch size for training
        epochs : int, default=100
            Maximum number of epochs for training
        early_stopping_patience : int, default=10
            Number of epochs with no improvement after which training will be stopped
        class_weight : str, default="balanced"
            Weighting scheme for imbalanced classification. Set to "balanced", "auto", None, or a dictionary
        random_state : int, default=42
            Random seed for reproducibility
        """
        super().__init__(model_name=model_name, random_state=random_state)
        
        # Set hidden layer structure
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 32, 16]
        
        # Set activation functions
        if activations is not None:
            if isinstance(activations, str):
                self.activations = [activations] * len(self.hidden_layers)
            else:
                if len(activations) != len(self.hidden_layers):
                    raise ValueError("Length of activations must match length of hidden_layers")
                self.activations = activations
        else:
            self.activations = ['relu'] * len(self.hidden_layers)
        
        # Set other parameters
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.class_weight = class_weight
        
        # Initialize preprocessing and model attributes
        self.scaler = StandardScaler()
        self.history = None
        self.input_dim = None
        
        # Set random seed for TensorFlow
        tf.random.set_seed(random_state)
    
    def build(self, input_dim=None):
        """
        Build the neural network model architecture.
        
        Parameters:
        -----------
        input_dim : int, default=None
            Input dimension (number of features). Required if not already set.
            
        Returns:
        --------
        self : object
            Model instance
        """
        # Set input dimension if provided
        if input_dim is not None:
            self.input_dim = input_dim
        
        # Ensure input_dim is set
        if self.input_dim is None:
            raise ValueError("Input dimension must be specified.")
        
        # Create model
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers
        for units, activation in zip(self.hidden_layers, self.activations):
            model.add(layers.Dense(units, activation=activation))
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer (binary classification)
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        return self
    
    def fit(self, X, y, validation_split=0.2, verbose=1):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix
        y : array-like, shape (n_samples,)
            Target vector
        validation_split : float, default=0.2
            Fraction of training data to use as validation data
        verbose : int, default=1
            Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
        --------
        self : object
            Fitted model instance
        """
        # Store feature and target names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        if isinstance(y, pd.Series):
            self.target_name = y.name
        
        # Determine class names if possible
        if hasattr(y, 'unique'):
            self.class_names = sorted(y.unique())
        
        # Convert inputs to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Set input dimension and build model if not already built
        if self.model is None:
            self.input_dim = X_scaled.shape[1]
            self.build()
        
        # Calculate class weights if needed
        if self.class_weight == "balanced" or self.class_weight == "auto":
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y_array),
                y=y_array
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        elif isinstance(self.class_weight, dict):
            class_weight_dict = self.class_weight
        else:
            class_weight_dict = None
        
        # Set up callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Fit model
        history = self.model.fit(
            X_scaled,
            y_array,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        self.history = history.history
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction
            
        Returns:
        --------
        np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Convert to numpy array if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Predict probabilities
        y_prob = self.model.predict(X_scaled)
        
        # Reshape to match scikit-learn's format (n_samples, n_classes)
        # For binary classification: P(class=0) = 1 - P(class=1)
        return np.hstack([1 - y_prob, y_prob])
    
    def predict(self, X, threshold=0.5):
        """
        Predict classes for X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for prediction
        threshold : float, default=0.5
            Threshold for binary classification
            
        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Predicted classes
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get probability predictions
        y_prob = self.predict_proba(X)[:, 1]
        
        # Apply threshold
        return (y_prob >= threshold).astype(int)
    
    def plot_training_history(self, figsize=(18, 6)):
        """
        Plot the training history.
        
        Parameters:
        -----------
        figsize : tuple, default=(18, 6)
            Figure size
        """
        if not self.is_fitted or self.history is None:
            raise ValueError("Model has not been fitted with history tracking. Call fit() first.")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        
        # Loss plot
        axes[0].plot(self.history['loss'], label='Training Loss')
        axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # AUC plot
        axes[2].plot(self.history['auc'], label='Training AUC')
        axes[2].plot(self.history['val_auc'], label='Validation AUC')
        axes[2].set_title('AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.suptitle(f"{self.model_name} Training History", fontsize=16, y=1.05)
        plt.show()
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        model_dir : str, default='models'
            Directory to save the model
            
        Returns:
        --------
        str
            Path to the saved model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model subdirectory
        model_path = os.path.join(model_dir, self.model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save Keras model
        keras_model_path = os.path.join(model_path, 'keras_model.keras')
        self.model.save(keras_model_path)
        
        # Save scaler
        import joblib
        scaler_path = os.path.join(model_path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'class_names': self.class_names,
            'hidden_layers': self.hidden_layers,
            'activations': self.activations,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'random_state': self.random_state,
            'history': self.history
        }
        
        metadata_path = os.path.join(model_path, 'metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model directory
            
        Returns:
        --------
        NeuralNetworkModel
            Loaded model instance
        """
        import joblib
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.joblib')
        metadata = joblib.load(metadata_path)
        
        # Create instance with saved parameters
        instance = cls(
            model_name=metadata['model_name'],
            hidden_layers=metadata['hidden_layers'],
            activations=metadata['activations'],
            dropout_rate=metadata['dropout_rate'],
            learning_rate=metadata['learning_rate'],
            batch_size=metadata['batch_size'],
            epochs=metadata['epochs'],
            random_state=metadata['random_state']
        )
        
        # Load Keras model
        keras_model_path = os.path.join(model_path, 'keras_model.keras')
        instance.model = keras.models.load_model(keras_model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_path, 'scaler.joblib')
        instance.scaler = joblib.load(scaler_path)
        
        # Set other attributes
        instance.feature_names = metadata['feature_names']
        instance.target_name = metadata['target_name']
        instance.class_names = metadata['class_names']
        instance.history = metadata['history']
        instance.input_dim = instance.model.layers[0].input_shape[1]
        instance.is_fitted = True
        
        return instance
    
    def plot_feature_importance(self, X, y, top_n=20, figsize=(12, 10)):
        """
        Calculate and plot permutation feature importance.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to use for importance calculation
        y : array-like, shape (n_samples,)
            Target vector to use for importance calculation
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing feature importance values
        """
        from sklearn.inspection import permutation_importance
        
        if not self.is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Convert to numpy array if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"Feature {i}" for i in range(X_array.shape[1])]
        
        # Define a prediction function that matches sklearn's API
        def predict_fn(X_pred):
            return self.predict_proba(X_pred)[:, 1]
        
        # Calculate permutation importance
        result = permutation_importance(
            predict_fn, X_array, y_array,
            n_repeats=10, random_state=self.random_state, n_jobs=-1
        )
        
        # Create DataFrame for importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': result.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
        plt.title(f"Top {top_n} Feature Importance - {self.model_name} (Permutation)")
        plt.tight_layout()
        plt.show()
        
        return feature_importance