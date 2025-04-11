#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Import necessary components from lstm_late_fusion.py
from lstm_late_fusion import (
    TimeSeriesTextDataset, 
    TimeSeriesWithTextModel, 
    preprocess_time_series, 
    preprocess_target,
    get_text_model_name,
    get_text_dim
)

def parse_args():
    parser = argparse.ArgumentParser(description='Predict using a trained LSTM Late Fusion Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--params_path', type=str, required=True,
                        help='Path to the parameters JSON file (e.g., config_*.json from Optuna trial)')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file with data from 2007-2023')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Output CSV file to save predictions')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='GPU ID to use (if -1, will use CPU)')
    return parser.parse_args()

# Load model configuration
def load_params(params_path):
    """Load parameters from a JSON file."""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Check if this is an Optuna study config file
    if 'text_model_name' in params and 'seq_len' in params:
        # This is already in the correct format
        config = params
    else:
        # Create a config dictionary with the necessary fields
        config = {
            'preprocessing': params.get('preprocessing', 'log1p'),
            'embedding_model': params.get('embedding_model', 'e5'),
            'lstm_hidden_dim': params.get('lstm_hidden_dim', 64),
            'fc_hidden_dim': params.get('fc_hidden_dim', 64),
            'dropout_rate': params.get('dropout_rate', 0.3),
            'text_model_name': get_text_model_name(params.get('embedding_model', 'e5')),
            'text_dim': get_text_dim(params.get('embedding_model', 'e5'))
        }
    
    return config

# Load model
def load_model(model_path, config, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model
    model = TimeSeriesWithTextModel(
        ts_input_dim=1,
        lstm_hidden_dim=config['lstm_hidden_dim'],
        text_dim=config['text_dim'],
        fc_hidden_dim=config['fc_hidden_dim'],
        dropout_rate=config.get('dropout_rate', config.get('drop_out', 0.3))
    ).to(device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

# Load tokenizer and text model
def load_text_model(config, device):
    # Load tokenizer and text model
    text_model_name = config['text_model_name']
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)
    text_model.eval()
    text_model.to(device)
    
    return tokenizer, text_model

# Load data from CSV
def load_data_from_csv(file_path, preprocessing_type):
    """Load and preprocess data from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract topics and time series data
    topics = df['topic'].tolist()
    
    # Extract years from 2007 to 2023 for input
    years_input = [str(year) for year in range(2007, 2024)]
    
    # Prepare time series data
    ts_data = df[years_input].values
    
    # Apply preprocessing to input data
    ts_data_processed = preprocess_time_series(ts_data, preprocessing_type)
    
    # Reshape to (n_samples, seq_len, 1) for LSTM
    ts_data_reshaped = ts_data_processed.reshape(ts_data_processed.shape[0], ts_data_processed.shape[1], 1)
    
    return ts_data_reshaped, topics

# Make predictions
def predict(model, tokenizer, text_model, config, ts_data, topics, device):
    # Create dataset
    dataset = TimeSeriesTextDataset(
        ts_data, topics, [0] * len(topics),  # Dummy targets since we're just predicting
        tokenizer, text_model, device, 
        config['embedding_model']
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32,  # Adjust batch size as needed
        shuffle=False
    )
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            ts_batch, text_vec_batch, _ = batch
            ts_batch = ts_batch.to(device)
            text_vec_batch = text_vec_batch.to(device)
            
            output = model(ts_batch, text_vec_batch)
            predictions.extend(output.cpu().numpy())
    
    # Convert predictions back to original scale
    original_predictions = np.expm1(predictions)
    
    return predictions, original_predictions

def main():
    args = parse_args()
    
    # Set device
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load parameters
    config = load_params(args.params_path)
    print(f"Loaded parameters from {args.params_path}")
    print(f"Preprocessing: {config.get('preprocessing', 'log1p')}")
    print(f"Embedding model: {config.get('embedding_model', 'e5')}")
    print(f"Text model: {config.get('text_model_name', 'intfloat/multilingual-e5-large-instruct')}")
    
    # Load model
    model = load_model(args.model_path, config, device)
    print(f"Model loaded from {args.model_path}")
    
    # Load tokenizer and text model
    tokenizer, text_model = load_text_model(config, device)
    print(f"Text model loaded: {config['text_model_name']}")
    
    # Load data from CSV
    ts_data, topics = load_data_from_csv(args.input_csv, config['preprocessing'])
    print(f"Loaded data from {args.input_csv}: {len(topics)} samples")
    
    # Make predictions
    log_predictions, original_predictions = predict(model, tokenizer, text_model, config, ts_data, topics, device)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'topic': topics,
        '2024': original_predictions
    })
    
    # Save to CSV
    output_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    
    # Print example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(topics))):
        print(f"Topic: {topics[i]}")
        print(f"Predicted 2024 value: {original_predictions[i]:.2f}")
        print("---")

if __name__ == "__main__":
    main() 