import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import json
import datetime
import time
from data_utils import load_data_from_csv_no_sliding, create_data_loaders

# Define the argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description='LSTM Late Fusion for Time Series Prediction')
    parser.add_argument('--preprocessing', type=str, choices=['log', 'minmax'], default='minmax',
                        help='Preprocessing method: "log" for log1p + min-max scaling, "minmax" for just min-max scaling')
    parser.add_argument('--embedding_model', type=str, choices=['distilbert', 'e5'], default='distilbert',
                        help='Embedding model to use: "distilbert" for distilbert-base-uncased, "e5" for intfloat/multilingual-e5-large-instruct')
    parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='Hidden dimension of LSTM')
    parser.add_argument('--fc_hidden_dim', type=int, default=128, help='Hidden dimension of fully connected layers')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models and results')
    parser.add_argument('--disable_progress_bar', action='store_true', help='Disable tqdm progress bars')
    return parser.parse_args()

# Define global variables that will be used by functions
SEQ_LEN = 17  # Years from 2007 to 2023 (excluding 2024 which is the target)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get text model name based on embedding model
def get_text_model_name(embedding_model):
    return 'distilbert-base-uncased' if embedding_model == 'distilbert' else 'intfloat/multilingual-e5-large-instruct'

# Function to get text dimension based on embedding model
def get_text_dim(embedding_model):
    return 768 if embedding_model == 'distilbert' else 1024  # E5-large has 1024 dimensions

# Model
class TimeSeriesWithTextModel(nn.Module):
    def __init__(self, ts_input_dim, lstm_hidden_dim, text_dim, fc_hidden_dim, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(ts_input_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim + text_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ts_input, text_vector):
        lstm_out, _ = self.lstm(ts_input)
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat([last_hidden, text_vector], dim=1)
        x = self.fc1(combined)
        x = self.dropout(x)
        output = self.fc2(x)
        return output.squeeze(-1)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping_patience, output_dir, timestamp, disable_progress_bar=False):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # Use tqdm if progress bars are enabled, otherwise just iterate normally
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if not disable_progress_bar else train_loader
        for ts_batch, text_vec_batch, target_batch in iterator:
            ts_batch = ts_batch.to(device)
            text_vec_batch = text_vec_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            output = model(ts_batch, text_vec_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ts_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(float(avg_train_loss))  # Convert to Python float
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for ts_batch, text_vec_batch, target_batch in val_loader:
                ts_batch = ts_batch.to(device)
                text_vec_batch = text_vec_batch.to(device)
                target_batch = target_batch.to(device)
                output = model(ts_batch, text_vec_batch)
                loss = criterion(output, target_batch)
                val_loss += loss.item() * ts_batch.size(0)

            avg_val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(float(avg_val_loss))  # Convert to Python float
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = f'{output_dir}/model_{timestamp}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"Model saved with validation loss: {best_val_loss:.4f} to {model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, train_losses, val_losses, best_val_loss, training_time

# Function to evaluate the model
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
            
            # Store predictions and actuals for analysis
            predictions.extend(output.cpu().numpy())
            actuals.extend(target_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Calculate and print metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = float(np.mean((predictions - actuals) ** 2))  # Convert to Python float
    rmse = float(np.sqrt(mse))  # Convert to Python float
    mae = float(np.mean(np.abs(predictions - actuals)))  # Convert to Python float
    r2 = float(1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))  # Convert to Python float

    print(f"Test MSE (log scale): {mse:.2f}")
    print(f"Test RMSE (log scale): {rmse:.2f}")
    print(f"Test MAE (log scale): {mae:.2f}")
    print(f"Test R² (log scale): {r2:.4f}")

    # Convert predictions back to original scale for display
    original_predictions = np.expm1(predictions)
    original_actuals = np.expm1(actuals)
    
    # Calculate metrics in original scale
    original_mse = float(np.mean((original_predictions - original_actuals) ** 2))
    original_rmse = float(np.sqrt(original_mse))
    original_mae = float(np.mean(np.abs(original_predictions - original_actuals)))
    original_r2 = float(1 - np.sum((original_actuals - original_predictions) ** 2) / np.sum((original_actuals - np.mean(original_actuals)) ** 2))
    
    print(f"Test MSE (original scale): {original_mse:.2f}")
    print(f"Test RMSE (original scale): {original_rmse:.2f}")
    print(f"Test MAE (original scale): {original_mae:.2f}")
    print(f"Test R² (original scale): {original_r2:.4f}")
    
    return avg_test_loss, mse, rmse, mae, r2, predictions, actuals, original_predictions, original_actuals, original_mse, original_rmse, original_mae, original_r2

# Function to save results
def save_results(results, output_dir, timestamp):
    results_file = f'{output_dir}/results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_file}")

# Function to print example predictions
def print_example_predictions(test_texts, predictions, actuals, original_predictions, original_actuals):
    print("\nExample predictions (in log scale):")
    for i in range(min(5, len(test_texts))):
        print(f"Topic: {test_texts[i]}")
        print(f"Actual 2024 value (log): {actuals[i]:.2f}")
        print(f"Predicted 2024 value (log): {predictions[i]:.2f}")
        print(f"Error (log): {abs(actuals[i] - predictions[i]):.2f}")
        print("---")

    print("\nExample predictions (in original scale):")
    for i in range(min(5, len(test_texts))):
        print(f"Topic: {test_texts[i]}")
        print(f"Actual 2024 value: {original_actuals[i]:.2f}")
        print(f"Predicted 2024 value: {original_predictions[i]:.2f}")
        print(f"Error: {abs(original_actuals[i] - original_predictions[i]):.2f}")
        print("---")

# Main function to run the entire pipeline
def run_pipeline(args, config_file, timestamp, disable_progress_bar=False):
    # Get text model name and dimension based on embedding model
    text_model_name = get_text_model_name(args.embedding_model)
    text_dim = get_text_dim(args.embedding_model)
    
    # Load tokenizer and BERT
    print(f"Loading tokenizer and model: {text_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    bert_model = AutoModel.from_pretrained(text_model_name)
    bert_model.eval()  # Freeze weights
    bert_model.to(DEVICE)

    # Define year ranges for each data split
    train_input_years = list(range(2007, 2022))  # 2007 to 2021
    train_target_year = '2022'
    
    val_input_years = list(range(2008, 2023))    # 2008 to 2022
    val_target_year = '2023'
    
    test_input_years = list(range(2009, 2024))   # 2009 to 2023
    test_target_year = '2024'

    # Load data from CSV files with specific year ranges
    print("Loading data from CSV files...")
    train_ts, train_texts, train_targets = load_data_from_csv_no_sliding(
        'data/train_data_new.csv', args.preprocessing, train_input_years, train_target_year)
    
    val_ts, val_texts, val_targets = load_data_from_csv_no_sliding(
        'data/val_data_new.csv', args.preprocessing, val_input_years, val_target_year)
    
    test_ts, test_texts, test_targets = load_data_from_csv_no_sliding(
        'data/test_data_new.csv', args.preprocessing, test_input_years, test_target_year)

    print(f"Train data: {len(train_ts)} samples, input years: {train_input_years}, target year: {train_target_year}")
    print(f"Validation data: {len(val_ts)} samples, input years: {val_input_years}, target year: {val_target_year}")
    print(f"Test data: {len(test_ts)} samples, input years: {test_input_years}, target year: {test_target_year}")

    # Update global SEQ_LEN based on the number of input years for each dataset
    # Note: We'll use the train sequence length for model initialization
    train_seq_len = len(train_input_years)
    val_seq_len = len(val_input_years)
    test_seq_len = len(test_input_years)
    
    print(f"Sequence lengths - Train: {train_seq_len}, Val: {val_seq_len}, Test: {test_seq_len}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ts, train_texts, train_targets, 
        val_ts, val_texts, val_targets, 
        test_ts, test_texts, test_targets, 
        tokenizer, bert_model, DEVICE, args.embedding_model, args.batch_size
    )

    # Initialize model
    model = TimeSeriesWithTextModel(
        ts_input_dim=1,  # Each time step has 1 feature (the normalized value)
        lstm_hidden_dim=args.lstm_hidden_dim,
        text_dim=text_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        dropout_rate=args.dropout_rate
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train model
    model, train_losses, val_losses, best_val_loss, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, DEVICE, 
        args.epochs, args.early_stopping_patience, args.output_dir, timestamp,
        disable_progress_bar=disable_progress_bar
    )

    # Load best model for testing
    model_path = f'{args.output_dir}/model_{timestamp}.pth'
    model.load_state_dict(torch.load(model_path))

    # Evaluate model
    avg_test_loss, mse, rmse, mae, r2, predictions, actuals, original_predictions, original_actuals, original_mse, original_rmse, original_mae, original_r2 = evaluate_model(
        model, test_loader, criterion, DEVICE
    )

    # Save evaluation results
    results = {
        'timestamp': timestamp,
        'preprocessing': args.preprocessing,
        'embedding_model': args.embedding_model,
        'text_model_name': text_model_name,
        # 'train_input_years': train_input_years,
        # 'train_target_year': train_target_year,
        # 'val_input_years': val_input_years,
        # 'val_target_year': val_target_year,
        # 'test_input_years': test_input_years,
        # 'test_target_year': test_target_year,
        'training_time': float(training_time),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'test_loss': float(avg_test_loss),
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'model_path': model_path,
        'config_path': config_file
    }
    save_results(results, args.output_dir, timestamp)

    # Print example predictions
    print_example_predictions(test_texts, predictions, actuals, original_predictions, original_actuals)
    
    return model, results

# Run the pipeline if this script is executed directly
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save configuration to a JSON file
    config = {
        'timestamp': timestamp,
        'preprocessing': args.preprocessing,
        'embedding_model': args.embedding_model,
        'text_model_name': get_text_model_name(args.embedding_model),
        'train_input_years': list(range(2007, 2022)),
        'train_target_year': '2022',
        'val_input_years': list(range(2008, 2023)),
        'val_target_year': '2023',
        'test_input_years': list(range(2009, 2024)),
        'test_target_year': '2024',
        'text_dim': get_text_dim(args.embedding_model),
        'lstm_hidden_dim': args.lstm_hidden_dim,
        'fc_hidden_dim': args.fc_hidden_dim,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'device': str(DEVICE)
    }
    
    config_file = f'{args.output_dir}/config_{timestamp}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Using device: {DEVICE}")
    print(f"Using preprocessing: {args.preprocessing}")
    print(f"Using embedding model: {get_text_model_name(args.embedding_model)}")
    print(f"Configuration saved to {config_file}")
    
    # Run the pipeline
    run_pipeline(args, config_file, timestamp)
