#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import pandas as pd
import json
import time
import os
from transformers import AutoTokenizer, AutoModel

# Import necessary components
from lstm_sliding_window_features import (
    TimeSeriesWithTextModel,
    get_text_model_name,
    get_text_dim
)

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions with a trained LSTM Sliding Window Model')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save prediction results (csv file)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (if -1, will use CPU)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add is_sliding_window flag
    config['is_sliding_window'] = True
    
    return config

def load_model(model_path, config, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get features from the nested structure
    features = config['features']['used_features']
    ts_input_dim = 1 + len(features)
    print(f"Number of features: {len(features)}")
    print(f"Features: {features}")
    
    # Create model
    model = TimeSeriesWithTextModel(
        ts_input_dim=ts_input_dim,
        lstm_hidden_dim=config['lstm_hidden_dim'],
        text_dim=config['text_dim'],
        fc_hidden_dim=config['fc_hidden_dim'],
        dropout_rate=config.get('dropout_rate', 0.3)
    ).to(device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def load_text_model(config, device):
    # Load tokenizer and text model
    text_model_name = config['text_model_name']
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)
    text_model.eval()
    text_model.to(device)
    
    return tokenizer, text_model

def preprocess_time_series(time_series, config):
    """Preprocess time series data according to config settings."""
    preprocessing = config.get('preprocessing', 'minmax')
    
    if preprocessing == 'log':
        # Apply log1p transformation
        processed = np.log1p(time_series)
    else:
        # Just use the original values for minmax
        processed = time_series.copy()
    
    # Scale values between 0 and 1 (minmax)
    min_val = np.min(processed)
    max_val = np.max(processed)
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    return processed

def calculate_features(time_series, features):
    """Calculate additional features for the time series."""
    ts_with_features = []
    window_len = len(time_series)
    
    for i in range(window_len):
        features_at_i = [time_series[i]]  # Start with the time series value
        
        if 'year_index' in features:
            # Normalize the year index to be between 0 and 1
            year_index = i / (window_len - 1) if window_len > 1 else 0
            features_at_i.append(year_index)
        
        if 'pct_change' in features:
            # Calculate percentage change from previous time step
            if i > 0 and time_series[i-1] != 0:
                pct_change = (time_series[i] - time_series[i-1]) / time_series[i-1]
            else:
                pct_change = 0
            features_at_i.append(pct_change)
        
        if 'slope' in features:
            # Calculate local slope using 3 points if possible
            if i >= 2:
                slope = (time_series[i] - time_series[i-2]) / 2
            elif i == 1:
                slope = time_series[i] - time_series[i-1]
            else:
                slope = 0
            features_at_i.append(slope)
        
        ts_with_features.append(features_at_i)
    
    return np.array(ts_with_features)

def get_text_embedding(text, tokenizer, text_model, device, embedding_model):
    """Get text embedding for a given text."""
    with torch.no_grad():
        if embedding_model == 'e5':
            # Prepare text for E5 model - add prefix for instructional embedding
            instruction_text = f"Represent this academic research topic: {text}"
            encoded = tokenizer(instruction_text, padding=True, truncation=True, 
                               return_tensors='pt').to(device)
            outputs = text_model(**encoded)
            # Use mean pooling for E5
            attention_mask = encoded['attention_mask']
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.sum(mask_expanded, 1)
            embedding = sum_embeddings / sum_mask
        else:  # distilbert
            encoded = tokenizer(text, padding=True, truncation=True, 
                               return_tensors='pt').to(device)
            outputs = text_model(**encoded)
            embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token for DistilBERT
        
        return embedding.cpu().numpy()[0]

def make_prediction(model, time_series_with_features, text_embedding, device):
    """Make a prediction with the model."""
    model.eval()
    with torch.no_grad():
        # Convert inputs to torch tensors
        ts_tensor = torch.FloatTensor(time_series_with_features).unsqueeze(0).to(device)  # Add batch dimension
        text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0).to(device)  # Add batch dimension
        
        # Get prediction
        start_time = time.time()
        output = model(ts_tensor, text_tensor)
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time:.4f} seconds")
        prediction = output.item()
        
        return prediction

def main():
    args = parse_args()
    
    # Set device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load config
    config = load_config(args.config_path)
    print(f"Loaded config from {args.config_path}")
    
    # Load model
    model = load_model(args.model_path, config, device)
    print(f"Model loaded from {args.model_path}")
    
    # Load tokenizer and text model
    tokenizer, text_model = load_text_model(config, device)
    print(f"Text model loaded: {config['text_model_name']}")
    
    # Example time series data (replace with your actual data)
    # This is a sample of time series data for the last n years
    # For example, if window_size is 12, this would be the last 12 years of data
    # Note: In a real scenario, this data should be loaded from a file or database

    # learning stability
    # raw_time_series = [295, 348, 434, 549, 729, 903, 1145, 1625, 1990, 2207, 2140, 2351]
    # machine learning for security
    # raw_time_series = [93, 114, 187, 247, 348, 572, 1023, 1574, 2075, 2350, 2405, 2498]
    # adaptive gradient methods
    # raw_time_series = [938, 1196, 1324, 1966, 2369, 2493, 2939, 3566, 4251, 4469, 4164, 4529]
    # attention economy
    # raw_time_series = [9, 19, 26, 29, 26, 31, 37, 50, 63, 71, 70, 66]
    # tuberculosis detection and diagnosis
    raw_time_series = [6, 11, 6, 7, 8, 11, 14, 10, 44, 42, 28, 35]

    # Example topic
    topic = "tuberculosis detection and diagnosis"
    
    # Process time series
    window_size = config['window_size']
    if len(raw_time_series) != window_size:
        print(f"Warning: Input time series length ({len(raw_time_series)}) does not match model window size ({window_size})")
        print("Adjusting input to match window size...")
        # Pad or truncate to match window size
        if len(raw_time_series) < window_size:
            # Pad with zeros at the beginning
            raw_time_series = [0] * (window_size - len(raw_time_series)) + raw_time_series
        else:
            # Use the last window_size values
            raw_time_series = raw_time_series[-window_size:]
    
    # Preprocess time series
    processed_time_series = preprocess_time_series(np.array(raw_time_series), config)
    
    # Calculate features
    features = config['features']['used_features']
    time_series_with_features = calculate_features(processed_time_series, features)
    
    # Get text embedding
    text_embedding = get_text_embedding(topic, tokenizer, text_model, device, config['embedding_model'])
    
    # Make prediction
    log_prediction = make_prediction(model, time_series_with_features, text_embedding, device)
    
    # Convert prediction back to original scale if log scaling was used
    # if config.get('preprocessing') == 'log':
    original_prediction = np.expm1(log_prediction)
    # else:
    #     # For minmax, we would need to reverse the scaling, but we don't have the original min/max values here
    #     # This is a simplification; in practice, you'd need to store and use the original scaling parameters
    #     original_prediction = log_prediction
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"Topic: {topic}")
    if config.get('preprocessing') == 'log':
        print(f"Predicted value (log scale): {log_prediction:.4f}")
    print(f"Predicted value: {original_prediction:.4f}")
    
    # Save prediction to CSV
    results = pd.DataFrame({
        'topic': [topic],
        'raw_time_series': [raw_time_series],
        'prediction': [original_prediction]
    })
    results.to_csv(args.output_path, index=False)
    print(f"Prediction saved to {args.output_path}")

if __name__ == "__main__":
    main() 