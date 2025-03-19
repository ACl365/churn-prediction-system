from setuptools import setup, find_packages

setup(
    name="telechurn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    author="Alexander Clarke",
    author_email="alexanderclarke365@gmail.com",
    description="A machine learning system for telecom customer churn prediction",
    keywords="machine learning, churn prediction, telecom",
    url="https://github.com/ACl365/churn-prediction-system",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)