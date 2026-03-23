"""
Data Preprocessing Module

Handles data loading, cleaning, normalization, and exploratory analysis.
Provides utilities for preparing raw data for logistic curve fitting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Data preprocessing and analysis utilities."""
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.scaler = None
        self.original_data = None
        self.processed_data = None
        self.statistics = None
    
    @staticmethod
    def load_data(filepath, **kwargs):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        **kwargs : dict
            Additional arguments for pd.read_csv()
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        try:
            data = pd.read_csv(filepath, **kwargs)
            print(f"✓ Data loaded successfully. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    @staticmethod
    def handle_missing_values(data, method='mean', threshold=0.5):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        method : str
            Method to handle missing values:
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'drop': Drop rows with missing values
            - 'forward_fill': Forward fill
            - 'interpolate': Linear interpolation
        threshold : float
            Drop columns with missing ratio > threshold
            
        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        data = data.copy()
        
        # Drop columns with too many missing values
        missing_ratio = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        data = data.drop(cols_to_drop, axis=1)
        
        # Handle remaining missing values
        if method == 'mean':
            data = data.fillna(data.mean())
        elif method == 'median':
            data = data.fillna(data.median())
        elif method == 'drop':
            data = data.dropna()
        elif method == 'forward_fill':
            data = data.fillna(method='ffill')
        elif method == 'interpolate':
            data = data.interpolate(method='linear')
        
        print(f"✓ Missing values handled using '{method}' method")
        return data
    
    @staticmethod
    def remove_outliers(data, method='iqr', threshold=1.5):
        """
        Remove outliers from data.
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Input data
        method : str
            Method to detect outliers:
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method (|z| > 3)
            - 'mad': Median absolute deviation
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        pd.Series or pd.DataFrame
            Data with outliers removed
        indices : array
            Indices of non-outlier rows
        """
        data_clean = data.copy()
        
        if method == 'iqr':
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (data_clean >= lower_bound) & (data_clean <= upper_bound)
            
        elif method == 'zscore':
            mean = data_clean.mean()
            std = data_clean.std()
            z_scores = np.abs((data_clean - mean) / std)
            mask = z_scores <= 3
            
        elif method == 'mad':
            median = data_clean.median()
            mad = np.median(np.abs(data_clean - median))
            mask = np.abs((data_clean - median) / (mad + 1e-10)) <= threshold
        
        indices = np.where(mask)[0]
        data_clean = data_clean.iloc[indices]
        
        n_removed = len(data) - len(data_clean)
        print(f"✓ Removed {n_removed} outliers using '{method}' method")
        
        return data_clean, indices
    
    @staticmethod
    def normalize_data(data, method='standard'):
        """
        Normalize/standardize data.
        
        Parameters:
        -----------
        data : pd.DataFrame or array-like
            Input data
        method : str
            Normalization method:
            - 'standard': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'robust': Robust scaling with median and IQR
            
        Returns:
        --------
        pd.DataFrame or array
            Normalized data
        scaler : object
            Fitted scaler for inverse transformation
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if isinstance(data, pd.DataFrame):
            data_normalized = pd.DataFrame(
                scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        else:
            data_normalized = scaler.fit_transform(data)
        
        print(f"✓ Data normalized using '{method}' method")
        return data_normalized, scaler
    
    @staticmethod
    def get_data_statistics(data):
        """
        Calculate descriptive statistics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
            
        Returns:
        --------
        dict
            Dictionary of statistics
        """
        stats = {
            'count': data.count(),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
        return stats
    
    @staticmethod
    def detect_anomalies(data, method='isolation_forest', contamination=0.1):
        """
        Detect anomalies in data.
        
        Parameters:
        -----------
        data : array-like
            Input data
        method : str
            Detection method:
            - 'isolation_forest': Isolation Forest algorithm
            - 'local_outlier': Local Outlier Factor
        contamination : float
            Expected proportion of anomalies
            
        Returns:
        --------
        array
            Anomaly labels (-1 for anomaly, 1 for normal)
        """
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(contamination=contamination)
        elif method == 'local_outlier':
            from sklearn.neighbors import LocalOutlierFactor
            detector = LocalOutlierFactor(contamination=contamination)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        predictions = detector.fit_predict(data.reshape(-1, 1))
        n_anomalies = np.sum(predictions == -1)
        print(f"✓ Found {n_anomalies} anomalies using '{method}' method")
        
        return predictions


def preprocess_data(data, handle_missing='mean', remove_outliers_flag=True,
                    normalize=True, normalize_method='standard'):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw input data
    handle_missing : str
        Method to handle missing values
    remove_outliers_flag : bool
        Whether to remove outliers
    normalize : bool
        Whether to normalize data
    normalize_method : str
        Normalization method
        
    Returns:
    --------
    X : array
        Preprocessed independent variable
    y : array
        Preprocessed dependent variable
    preprocessing_info : dict
        Information about preprocessing steps
    """
    print("Starting data preprocessing pipeline...")
    print("-" * 50)
    
    # Create preprocessor instance
    preprocessor = DataPreprocessor()
    preprocessor.original_data = data.copy()
    
    # Handle missing values
    data = preprocessor.handle_missing_values(data, method=handle_missing)
    
    # Remove outliers if requested
    if remove_outliers_flag:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col], _ = preprocessor.remove_outliers(data[col], method='iqr')
    
    # Calculate statistics
    preprocessor.statistics = preprocessor.get_data_statistics(data)
    print("\n✓ Data Statistics:")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: {preprocessor.statistics['mean'].mean():.4f}")
    print(f"  Std Dev: {preprocessor.statistics['std'].mean():.4f}")
    
    # Extract X and y (assuming first column is index and second is target)
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) < 2:
        raise ValueError("Data must contain at least 2 numeric columns")
    
    X = numeric_data.iloc[:, 0].values
    y = numeric_data.iloc[:, 1].values
    
    # Normalize if requested
    if normalize:
        X_norm, scaler_X = preprocessor.normalize_data(X.reshape(-1, 1), method=normalize_method)
        y_norm, scaler_y = preprocessor.normalize_data(y.reshape(-1, 1), method=normalize_method)
        X = X_norm.flatten()
        y = y_norm.flatten()
    else:
        scaler_X, scaler_y = None, None
    
    preprocessor.processed_data = (X, y)
    
    preprocessing_info = {
        'original_shape': preprocessor.original_data.shape,
        'processed_shape': (len(X), 2),
        'rows_removed': len(data) - len(X),
        'missing_handled': handle_missing,
        'outliers_removed': remove_outliers_flag,
        'normalized': normalize,
        'normalize_method': normalize_method,
        'statistics': preprocessor.statistics,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    print("-" * 50)
    print("✓ Preprocessing pipeline completed!")
    print(f"\nFinal data shape: {X.shape[0]} observations")
    
    return X, y, preprocessing_info


def load_data(filepath, **kwargs):
    """
    Load data from file.
    
    Parameters:
    -----------
    filepath : str
        Path to data file
    **kwargs : dict
        Additional arguments for pd.read_csv()
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    return DataPreprocessor.load_data(filepath, **kwargs)