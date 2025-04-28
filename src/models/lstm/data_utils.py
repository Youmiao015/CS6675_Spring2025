import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress

# Custom Dataset
class TimeSeriesTextDataset(Dataset):
    def __init__(self, ts_data, text_data, targets, tokenizer, bert_model, device, embedding_model):
        self.ts_data = ts_data
        self.text_data = text_data
        self.targets = targets
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.device = device
        self.embedding_model = embedding_model
        self.seq_len = ts_data.shape[1]  # Get sequence length from data

    def __len__(self):
        return len(self.ts_data)

    def __getitem__(self, idx):
        ts = torch.tensor(self.ts_data[idx], dtype=torch.float32)  # (seq_len, features)
        text = self.text_data[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # Tokenize + Embed text
        with torch.no_grad():
            if self.embedding_model == 'distilbert':
                encoded = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                # Move encoded tensors to the same device as the model
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.bert_model(**encoded)
                text_vector = output.last_hidden_state[:, 0, :].squeeze(0)  # CLS token
            else:  # E5 model
                # For E5, we need to add a prefix to the text
                encoded = self.tokenizer(f"passage: {text}", return_tensors='pt', truncation=True, padding=True)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                output = self.bert_model(**encoded)
                text_vector = output.last_hidden_state[:, 0, :].squeeze(0)  # CLS token
            
            # Move text_vector back to CPU for the DataLoader
            text_vector = text_vector.cpu()

        return ts, text_vector, target

# Function to preprocess time series data
def preprocess_time_series(data, preprocessing_type):
    if preprocessing_type == 'log':
        # Apply log1p transformation
        data = np.log1p(data)
    
    # Min-max scaling to [0, 1] for each row (time series)
    data_min = data.min(axis=1, keepdims=True)
    data_max = data.max(axis=1, keepdims=True)
    data_scaled = (data - data_min) / (data_max - data_min + 1e-8)
    
    return data_scaled

# Function to apply log1p to target values
def preprocess_target(target_values):
    return np.log1p(target_values)

def minmax_scale(data, min_val=None, max_val=None):
    """Min-max scaling to [0, 1] range.
    If min_val and max_val are provided, use them for scaling.
    Otherwise, compute min and max from the data.
    """
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    return (data - min_val) / (max_val - min_val + 1e-8)

def standardize(data, mean=None, std=None):
    """Standardize data to zero mean and unit variance.
    If mean and std are provided, use them for scaling.
    Otherwise, compute mean and std from the data.
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    return (data - mean) / (std + 1e-8)

# Global year range for year_index feature
GLOBAL_YEAR_RANGE = {
    'min': 2007,  # Update these based on your data
    'max': 2027
}

def rolling_slope(series, window=3):
    """
    Compute the slope (trend) at each timestep using local linear regression.
    Pads the beginning where window can't be applied.
    """
    # Ensure series is a numpy array
    series = np.asarray(series, dtype=np.float64)
    slopes = np.zeros_like(series, dtype=float)
    half_window = window // 2

    for t in range(len(series)):
        left = max(0, t - half_window)
        right = min(len(series), t + half_window + 1)
        window_data = series[left:right]
        
        # Ensure we have at least 2 points for regression
        if len(window_data) < 2:
            slopes[t] = 0
            continue
            
        # Create x values (time indices) for the regression
        x = np.arange(len(window_data), dtype=np.float64)
        y = window_data.astype(np.float64)
        
        try:
            slope, _, _, _, _ = linregress(x, y)
            slopes[t] = slope
        except:
            slopes[t] = 0

    return slopes

def create_time_features(input_years, ts_data, feature_configs=None):
    """Helper function to create time-based features.
    
    Args:
        input_years: List of years to create features for
        ts_data: Time series data for the window (can be 1D or 2D array)
        feature_configs: Dictionary of feature configurations. If None, uses default configs.
    """
    if feature_configs is None:
        feature_configs = {
            'year_index': {
                'scale_method': 'minmax',
                'scale_global': True
            },
            'pct_change': {
                'scale_method': 'standardize',
                'scale_global': False
            },
            'slope': {
                'scale_method': 'standardize',
                'scale_global': False
            }
        }
    
    features = {}
    
    # Year index feature
    if 'year_index' in feature_configs:
        config = feature_configs['year_index']
        year_indices = np.array([int(year) for year in input_years])
        
        if config['scale_global']:
            year_indices = minmax_scale(year_indices, GLOBAL_YEAR_RANGE['min'], GLOBAL_YEAR_RANGE['max'])
        else:
            year_indices = minmax_scale(year_indices)
        
        # Create feature vector with the same year index for all samples
        features['year_index'] = np.tile(year_indices.reshape(-1, 1), (ts_data.shape[0], 1, 1))
    
    # Percentage change feature
    if 'pct_change' in feature_configs:
        config = feature_configs['pct_change']
        # Ensure ts_data is 2D
        if len(ts_data.shape) == 1:
            ts_data = ts_data.reshape(1, -1)
        
        # Calculate percentage changes for each row
        pct_changes = np.zeros_like(ts_data)
        epsilon = 1e-8  # Small value to avoid division by zero
        
        for row_idx in range(ts_data.shape[0]):
            row_data = ts_data[row_idx]
            for i in range(1, len(row_data)):
                prev_val = row_data[i-1]
                curr_val = row_data[i]
                
                # Handle different cases for percentage change calculation
                if abs(prev_val) < epsilon and abs(curr_val) < epsilon:
                    # Both values are effectively zero
                    pct_changes[row_idx, i] = 0
                elif abs(prev_val) < epsilon:
                    # Previous value is zero, current is not
                    pct_changes[row_idx, i] = 100 if curr_val > 0 else -100
                else:
                    # Normal case: calculate percentage change
                    pct_changes[row_idx, i] = (curr_val - prev_val) / (abs(prev_val) + epsilon) * 100
        
        # Set the first year's pct_change to zero for all rows
        pct_changes[:, 0] = 0
        
        if config['scale_method'] == 'minmax':
            pct_changes = minmax_scale(pct_changes)
        else:  # standardize
            pct_changes = standardize(pct_changes)
        
        # Reshape to match the expected format (samples, time_steps, features)
        features['pct_change'] = pct_changes.reshape(ts_data.shape[0], -1, 1)
    
    # Slope feature
    if 'slope' in feature_configs:
        config = feature_configs['slope']
        # Ensure ts_data is 2D
        if len(ts_data.shape) == 1:
            ts_data = ts_data.reshape(1, -1)
        
        # Calculate rolling slope for each time step in each row
        slopes = np.zeros_like(ts_data)
        for row_idx in range(ts_data.shape[0]):
            # Calculate rolling slope for each time step
            slopes[row_idx] = rolling_slope(ts_data[row_idx], window=3)
        
        # Standardize slopes per window
        slopes = standardize(slopes)
        
        # Reshape to match the expected format (samples, time_steps, features)
        features['slope'] = slopes.reshape(ts_data.shape[0], -1, 1)
    return features

def load_data_from_csv_sliding(file_path, preprocessing_type, window_size=10, use_features=None, cutoff_year=2022):
    """Load data with sliding window, optionally including additional features.
    
    Args:
        file_path: Path to the CSV file
        preprocessing_type: Type of preprocessing to apply
        window_size: Size of the sliding window
        use_features: List of additional features to include
        cutoff_year: Year to cut off the sliding window (inclusive). Windows will not include years after this.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract topics and time series data
    topics = df['topic'].tolist()
    
    # Extract all year columns (excluding 'topic')
    year_columns = [col for col in df.columns if col != 'topic']
    all_years = sorted(year_columns)  # Ensure years are in order
    
    # Filter years up to cutoff_year
    all_years = [year for year in all_years if int(year) <= cutoff_year]
    
    # Prepare sliding window data
    ts_data_list = []
    topics_list = []
    targets_list = []
    
    # For each possible window position
    for i in range(len(all_years) - window_size):
        input_years = all_years[i:i+window_size]
        target_year = all_years[i+window_size]
        
        # Skip if target year is after cutoff_year
        if int(target_year) > cutoff_year:
            continue
            
        # Get input and target data
        window_ts_data = df[input_years].values
        window_targets = df[target_year].values

        # Create features using raw data
        if use_features:
            features = create_time_features(input_years, window_ts_data)
        
        # Apply preprocessing to input data
        window_ts_data_processed = preprocess_time_series(window_ts_data, preprocessing_type)
        
        # Apply log1p to target values
        window_targets_processed = preprocess_target(window_targets)
        
        # Reshape time series data
        window_ts_data_reshaped = window_ts_data_processed.reshape(window_ts_data_processed.shape[0], window_ts_data_processed.shape[1], 1)
        
        if use_features:
            # Combine all requested features
            feature_list = []
            for feature_name in use_features:
                if feature_name in features:
                    feature_list.append(features[feature_name])
                else:
                    raise ValueError(f"Feature '{feature_name}' not found. Available features: {list(features.keys())}")
            
            # Combine time series data with features
            window_ts_data_with_features = np.concatenate([window_ts_data_reshaped] + feature_list, axis=2)
        else:
            window_ts_data_with_features = window_ts_data_reshaped
        
        # Add to lists
        ts_data_list.append(window_ts_data_with_features)
        topics_list.extend([topics] * len(topics))  # Duplicate topics for each window
        targets_list.append(window_targets_processed)
    
    # Combine all window data
    combined_ts_data = np.vstack(ts_data_list)
    combined_targets = np.concatenate(targets_list)
    
    # For topics, we need to flatten the list of lists
    combined_topics = []
    for topic_list in topics_list[:len(ts_data_list)]:  # Only use as many topic lists as we have windows
        combined_topics.extend(topic_list)
    
    print(f"\nCreated {len(ts_data_list)} windows, total samples: {combined_ts_data.shape[0]}")
    print(f"Using years up to {cutoff_year} for sliding window training")
    
    return combined_ts_data, combined_topics, combined_targets

# Function to load and preprocess data from CSV without sliding window
def load_data_from_csv_no_sliding(file_path, preprocessing_type, input_years, target_year, use_features=None):
    """Load data without sliding window, optionally including additional features."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract topics and time series data
    topics = df['topic'].tolist()
    
    # Extract specified years for input and target
    years_input = [str(year) for year in input_years]
    
    # Prepare time series data
    ts_data = df[years_input].values
    targets = df[target_year].values
    
    # Create features using raw data
    if use_features:
        features = create_time_features(years_input, ts_data)
    
    # Apply preprocessing to input data
    ts_data_processed = preprocess_time_series(ts_data, preprocessing_type)
    
    # Always apply log1p to target values
    targets_processed = preprocess_target(targets)
    
    # Reshape time series data
    ts_data_reshaped = ts_data_processed.reshape(ts_data_processed.shape[0], ts_data_processed.shape[1], 1)
    
    if use_features:
        # Combine all requested features
        feature_list = []
        for feature_name in use_features:
            if feature_name in features:
                feature_list.append(features[feature_name])
            else:
                raise ValueError(f"Feature '{feature_name}' not found. Available features: {list(features.keys())}")
        
        # Combine time series data with features
        ts_data_with_features = np.concatenate([ts_data_reshaped] + feature_list, axis=2)
    else:
        ts_data_with_features = ts_data_reshaped
    
    return ts_data_with_features, topics, targets_processed

# Function to create data loaders
def create_data_loaders(train_ts, train_texts, train_targets, val_ts, val_texts, val_targets, 
                        test_ts, test_texts, test_targets, tokenizer, bert_model, device, 
                        embedding_model, batch_size):
    train_dataset = TimeSeriesTextDataset(train_ts, train_texts, train_targets, tokenizer, bert_model, device, embedding_model)
    val_dataset = TimeSeriesTextDataset(val_ts, val_texts, val_targets, tokenizer, bert_model, device, embedding_model)
    test_dataset = TimeSeriesTextDataset(test_ts, test_texts, test_targets, tokenizer, bert_model, device, embedding_model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 