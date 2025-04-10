#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Predict using a trained LSTM Late Fusion Model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model and config file')
    parser.add_argument('--topic', type=str, 
                        help='CS topic to predict popularity for')
    parser.add_argument('--popularity_sequence', type=str,
                        help='Comma-separated sequence of topic popularity values for years 2007-2023 (17 values)')
    parser.add_argument('--input_json', type=str,
                        help='JSON file containing multiple prediction inputs')
    parser.add_argument('--output_json', type=str, default='predictions.json',
                        help='Output JSON file to save predictions')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU ID to use (if None, will use CPU)')
    return parser.parse_args()

# Load model configuration
def load_config(model_dir):
    config_files = list(Path(model_dir).glob('config_*.json'))
    if not config_files:
        raise FileNotFoundError(f"No config file found in {model_dir}")
    
    # Sort by timestamp (newest first)
    config_files.sort(key=lambda x: x.name, reverse=True)
    config_path = config_files[0]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# Load model
def load_model(model_dir, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find model file
    model_files = list(Path(model_dir).glob('model_*.pth'))
    if not model_files:
        raise FileNotFoundError(f"No model file found in {model_dir}")
    
    # Sort by timestamp (newest first)
    model_files.sort(key=lambda x: x.name, reverse=True)
    model_path = model_files[0]
    
    # Create model
    from lstm_late_fusion import TimeSeriesWithTextModel
    
    model = TimeSeriesWithTextModel(
        ts_input_dim=1,
        lstm_hidden_dim=config['lstm_hidden_dim'],
        text_dim=config['text_dim'],
        fc_hidden_dim=config['fc_hidden_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

# Load tokenizer and text model
def load_text_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and text model
    text_model_name = config['text_model_name']
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModel.from_pretrained(text_model_name)
    text_model.eval()
    text_model.to(device)
    
    return tokenizer, text_model, device

# Preprocess time series data
def preprocess_time_series(data, preprocessing_type):
    data = np.array(data).reshape(1, -1)  # Reshape to [1, seq_len]
    
    if preprocessing_type == 'log':
        # Apply log1p transformation
        data = np.log1p(data)
    
    # Min-max scaling to [0, 1] for each row (time series)
    data_min = data.min(axis=1, keepdims=True)
    data_max = data.max(axis=1, keepdims=True)
    data_scaled = (data - data_min) / (data_max - data_min + 1e-8)
    
    # Reshape to [1, seq_len, 1] for LSTM
    data_reshaped = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
    
    return data_reshaped

# Get text embedding
def get_text_embedding(text, tokenizer, text_model, device, embedding_model):
    with torch.no_grad():
        if embedding_model == 'distilbert':
            encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = text_model(**encoded)
            text_vector = output.last_hidden_state[:, 0, :]  # CLS token
        else:  # E5 model
            # For E5, we need to add a prefix to the text
            encoded = tokenizer(f"passage: {text}", return_tensors='pt', truncation=True, padding=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = text_model(**encoded)
            text_vector = output.last_hidden_state[:, 0, :]  # CLS token
        
    return text_vector

# Make prediction
def predict(model, tokenizer, text_model, config, topic, popularity_sequence, device):
    # Process time series data
    ts_data = preprocess_time_series(popularity_sequence, config['preprocessing'])
    ts_tensor = torch.tensor(ts_data, dtype=torch.float32).to(device)
    
    # Process text data
    text_vector = get_text_embedding(topic, tokenizer, text_model, device, config['embedding_model'])
    
    # Make prediction
    with torch.no_grad():
        prediction = model(ts_tensor, text_vector).item()
    
    # Convert prediction back to original scale
    original_prediction = np.expm1(prediction)
    
    return {
        'topic': topic,
        'prediction_log_scale': prediction,
        'prediction': original_prediction,
        'input_sequence': popularity_sequence
    }

# Process input JSON file
def process_json_input(input_file, model, tokenizer, text_model, config, device):
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    
    predictions = []
    for input_data in inputs:
        topic = input_data.get('topic')
        popularity_sequence = input_data.get('popularity_sequence')
        
        if not topic or not popularity_sequence:
            print(f"Skipping invalid input: {input_data}")
            continue
        
        # Convert popularity_sequence to list of floats if it's a string
        if isinstance(popularity_sequence, str):
            popularity_sequence = [float(x) for x in popularity_sequence.split(',')]
        
        # Ensure we have 17 values (2007-2023)
        if len(popularity_sequence) != 17:
            print(f"Warning: Expected 17 values for years 2007-2023, got {len(popularity_sequence)}. Skipping.")
            continue
        
        prediction = predict(model, tokenizer, text_model, config, topic, popularity_sequence, device)
        predictions.append(prediction)
    
    return predictions

def main():
    args = parse_args()
    
    # Set device
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(device)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load config
    config = load_config(args.model_dir)
    print(f"Loaded config from {args.model_dir}")
    
    # Load model
    model, _ = load_model(args.model_dir, config)
    print("Model loaded successfully")
    
    # Load tokenizer and text model
    tokenizer, text_model, _ = load_text_model(config)
    print(f"Text model loaded: {config['text_model_name']}")
    
    results = []
    
    # Process input JSON if provided
    if args.input_json:
        results = process_json_input(args.input_json, model, tokenizer, text_model, config, device)
        print(f"Processed {len(results)} inputs from {args.input_json}")
    
    # Process single prediction if provided
    elif args.topic and args.popularity_sequence:
        # Convert popularity sequence to list of floats
        popularity_sequence = [float(x) for x in args.popularity_sequence.split(',')]
        
        # Ensure we have 17 values (2007-2023)
        if len(popularity_sequence) != 17:
            raise ValueError(f"Expected 17 values for years 2007-2023, got {len(popularity_sequence)}")
        
        result = predict(model, tokenizer, text_model, config, args.topic, popularity_sequence, device)
        results.append(result)
        
        # Print result
        print(f"\nPrediction for topic '{args.topic}':")
        print(f"2024 popularity prediction: {result['prediction']:.2f}")
        print(f"Log-scale prediction: {result['prediction_log_scale']:.4f}")
    
    else:
        print("No input provided. Please specify --topic and --popularity_sequence, or --input_json")
        return
    
    # Save results to output file
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main() 