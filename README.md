# Telecom Customer Churn Prediction System

A machine learning system for predicting customer churn in a telecom company.

## Project Structure

```
project/
├── data/
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── models/                   # Saved models
├── notebooks/                # Jupyter notebooks
│   ├── 03_model_training_evaluation.ipynb         # Model Training and Evaluation
│   ├── 03_model_training_evaluation_updated.ipynb # Updated Training Notebook
│   ├── 04_model_deployment.ipynb                  # Model Deployment
│   └── test_imports.ipynb                         # Test Imports
└── scripts/                  # Python scripts
    ├── base_model.py                # Base model class
    ├── gradient_boosting.py         # XGBoost and LightGBM implementations
    ├── neural_network.py            # Neural Network implementation
    ├── training_pipeline.py         # Training pipeline
    ├── utils.py                     # Utility functions
    ├── prepare_data.py              # Data preparation script
    └── fix_notebook.py              # Notebook fixing script
```

## Features

- **Multiple Model Support**: Implements XGBoost, LightGBM, and Neural Network models
- **Class Imbalance Handling**: Uses SMOTE for handling imbalanced classes
- **Hyperparameter Tuning**: Supports hyperparameter optimization
- **Threshold Optimization**: Finds optimal classification threshold
- **Cross-Validation**: Evaluates models using k-fold cross-validation
- **Feature Importance**: Visualizes feature importance for model interpretability
- **Model Comparison**: Compares different models on key metrics
- **Feature Alignment**: Ensures consistent features between training and prediction
- **Robust Evaluation**: Handles edge cases like single-class datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

```bash
python scripts/prepare_data.py
```

### Model Training

Run the Jupyter notebook:
```bash
jupyter notebook notebooks/03_model_training_evaluation_updated.ipynb
```

### Model Deployment

Run the deployment notebook:
```bash
jupyter notebook notebooks/04_model_deployment.ipynb
```

## Model Performance

The system evaluates models using multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Average Precision

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.