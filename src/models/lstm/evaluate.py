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

# Import necessary components
from lstm_sliding_window_features import (
    TimeSeriesWithTextModel as SlidingWindowModel,
    get_text_model_name as get_sliding_window_text_model_name,
    get_text_dim as get_sliding_window_text_dim
)
from data_utils import (
    TimeSeriesTextDataset,
    load_data_from_csv_no_sliding,
    create_data_loaders
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained LSTM Sliding Window Model')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--eval_output_path', type=str, required=True,
                        help='Path to save evaluation results (txt file)')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to save predictions (csv file)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (if -1, will use CPU)')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], required=True,
                        help='Specify the mode: train, val, or test')
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
    model = SlidingWindowModel(
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

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for ts_batch, text_vec_batch, target_batch in test_loader:
            ts_batch = ts_batch.to(device)
            text_vec_batch = text_vec_batch.to(device)
            target_batch = target_batch.to(device)
            
            output = model(ts_batch, text_vec_batch)
            loss = criterion(output, target_batch)
            test_loss += loss.item() * ts_batch.size(0)
            
            # Store predictions and actuals
            predictions.extend(output.cpu().numpy())
            actuals.extend(target_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Log scale metrics
    mse = float(np.mean((predictions - actuals) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(predictions - actuals)))
    r2 = float(1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    # Original scale metrics
    original_predictions = np.expm1(predictions)
    original_actuals = np.expm1(actuals)
    original_mse = float(np.mean((original_predictions - original_actuals) ** 2))
    original_rmse = float(np.sqrt(original_mse))
    original_mae = float(np.mean(np.abs(original_predictions - original_actuals)))
    original_r2 = float(1 - np.sum((original_actuals - original_predictions) ** 2) / np.sum((original_actuals - np.mean(original_actuals)) ** 2))
    
    # Get topics from dataset (only once)
    test_texts = test_loader.dataset.text_data
    
    return {
        'test_loss': float(avg_test_loss),
        'log_scale': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'original_scale': {
            'mse': original_mse,
            'rmse': original_rmse,
            'mae': original_mae,
            'r2': original_r2
        },
        'predictions': original_predictions,
        'actuals': original_actuals,
        'topics': test_texts
    }

def save_evaluation_results(results, eval_output_path, predictions_path, target_year):
    # Save evaluation metrics to txt file
    with open(eval_output_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        
        f.write("Log Scale Metrics:\n")
        f.write(f"MSE: {results['log_scale']['mse']:.4f}\n")
        f.write(f"RMSE: {results['log_scale']['rmse']:.4f}\n")
        f.write(f"MAE: {results['log_scale']['mae']:.4f}\n")
        f.write(f"R²: {results['log_scale']['r2']:.4f}\n\n")
        
        f.write("Original Scale Metrics:\n")
        f.write(f"MSE: {results['original_scale']['mse']:.4f}\n")
        f.write(f"RMSE: {results['original_scale']['rmse']:.4f}\n")
        f.write(f"MAE: {results['original_scale']['mae']:.4f}\n")
        f.write(f"R²: {results['original_scale']['r2']:.4f}\n\n")
        
        f.write("Example Predictions:\n")
        for i in range(min(5, len(results['topics']))):
            f.write(f"Topic: {results['topics'][i]}\n")
            f.write(f"Actual: {results['actuals'][i]:.2f}\n")
            f.write(f"Predicted: {results['predictions'][i]:.2f}\n")
            f.write(f"Error: {abs(results['actuals'][i] - results['predictions'][i]):.2f}\n")
            f.write("---\n")
    
    # Save predictions to csv file with target year
    predictions_df = pd.DataFrame({
        'topic': results['topics'],
        str(target_year): results['predictions']
    })
    predictions_df.to_csv(predictions_path, index=False)

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
    
    # Load test data
    window_size = config['window_size']
    if args.mode == 'train':
        target_year = 2022
        test_input_years = list(range(target_year-window_size, target_year))  # e.g., if window_size=12: 2012-2023
        test_target_year = str(target_year)
    elif args.mode == 'val':
        target_year = 2023
        test_input_years = list(range(target_year-window_size, target_year))  # e.g., if window_size=12: 2012-2023
        test_target_year = str(target_year)
    elif args.mode == 'test':
        target_year = 2024
        test_input_years = list(range(target_year-window_size, target_year))  # e.g., if window_size=12: 2012-2023
        test_target_year = str(target_year)
    
    test_ts, test_texts, test_targets = load_data_from_csv_no_sliding(
        f'data/{args.mode}_data_new.csv', 
        config['preprocessing'], 
        test_input_years, 
        test_target_year,
        use_features=config['features']['used_features']
    )
    
    # Create test dataset and loader
    test_dataset = TimeSeriesTextDataset(
        test_ts, test_texts, test_targets,
        tokenizer, text_model, device,
        config['embedding_model']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Evaluate model
    criterion = torch.nn.MSELoss()
    results = evaluate_model(model, test_loader, criterion, device)
    
    # Save results with target year
    save_evaluation_results(results, args.eval_output_path, args.predictions_path, target_year)
    print(f"Evaluation results saved to {args.eval_output_path}")
    print(f"Predictions saved to {args.predictions_path}")

if __name__ == "__main__":
    main() 