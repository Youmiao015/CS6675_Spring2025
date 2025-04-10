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

# Function to load and preprocess data from CSV
def load_data_from_csv(file_path, preprocessing_type):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract topics and time series data
    topics = df['topic'].tolist()
    
    # Extract years from 2007 to 2023 for input, and 2024 as target
    years_input = [str(year) for year in range(2007, 2024)]
    year_target = '2024'
    
    # Prepare time series data
    ts_data = df[years_input].values
    targets = df[year_target].values
    
    # Apply preprocessing to input data
    ts_data_processed = preprocess_time_series(ts_data, preprocessing_type)
    
    # Always apply log1p to target values
    targets_processed = preprocess_target(targets)
    
    # Reshape to (n_samples, seq_len, 1) for LSTM
    ts_data_reshaped = ts_data_processed.reshape(ts_data_processed.shape[0], ts_data_processed.shape[1], 1)
    
    return ts_data_reshaped, topics, targets_processed

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

    print(f"Test MSE: {mse:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.4f}")

    # Convert predictions back to original scale for display
    original_predictions = np.expm1(predictions)
    original_actuals = np.expm1(actuals)
    
    return avg_test_loss, mse, rmse, mae, r2, predictions, actuals, original_predictions, original_actuals

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

    # Load data from CSV files
    print("Loading data from CSV files...")
    train_ts, train_texts, train_targets = load_data_from_csv('data/train_data_new.csv', args.preprocessing)
    val_ts, val_texts, val_targets = load_data_from_csv('data/val_data_new.csv', args.preprocessing)
    test_ts, test_texts, test_targets = load_data_from_csv('data/test_data_new.csv', args.preprocessing)

    print(f"Train data: {len(train_ts)} samples")
    print(f"Validation data: {len(val_ts)} samples")
    print(f"Test data: {len(test_ts)} samples")

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
    avg_test_loss, mse, rmse, mae, r2, predictions, actuals, original_predictions, original_actuals = evaluate_model(
        model, test_loader, criterion, DEVICE
    )

    # Save evaluation results
    results = {
        'timestamp': timestamp,
        'preprocessing': args.preprocessing,
        'embedding_model': args.embedding_model,
        'text_model_name': text_model_name,
        'training_time': float(training_time),  # Convert to Python float
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),  # Convert to Python float
        'test_loss': float(avg_test_loss),  # Convert to Python float
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
        'seq_len': SEQ_LEN,
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
