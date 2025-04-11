import optuna
import torch
import numpy as np
import os
import json
import datetime
import argparse
from types import SimpleNamespace
from lstm_sliding_window import run_pipeline, parse_args, get_text_model_name, get_text_dim

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hyperparameter Tuning for LSTM Late Fusion Model')
parser.add_argument('--preprocessing', type=str, choices=['log', 'minmax'], default='minmax',
                    help='Preprocessing method: "log" for log1p + min-max scaling, "minmax" for just min-max scaling')
parser.add_argument('--embedding_model', type=str, choices=['distilbert', 'e5'], default='distilbert',
                    help='Embedding model to use: "distilbert" for distilbert-base-uncased, "e5" for intfloat/multilingual-e5-large-instruct')
parser.add_argument('--worker_id', type=str, default='worker1', help='Worker ID for distributed tuning')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for hyperparameter tuning')
parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs for this worker')
parser.add_argument('--storage', type=str, required=True, 
                    help='Storage URL for Optuna (required, e.g., sqlite:///optuna.db)')
parser.add_argument('--study_name', type=str, required=True, 
                    help='Name of the shared study (required)')
parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models and results')
parser.add_argument('--skip_final_model', action='store_true', 
                    help='Skip training the final model (useful for distributed tuning where the coordinator trains the final model)')
parser.add_argument('--pruning', action='store_true', 
                    help='Enable pruning of unpromising trials')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Set up device
if torch.cuda.is_available() and args.gpu_id >= 0:
    device = torch.device(f'cuda:{args.gpu_id}')
    print(f"Using GPU: {device}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    params = {
        'preprocessing': args.preprocessing,  # Use preprocessing from command-line args
        'embedding_model': args.embedding_model,
        'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [32, 64, 128]),
        'fc_hidden_dim': trial.suggest_categorical('fc_hidden_dim', [32, 64, 128, 256, 512]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'epochs': 20,  # Fixed number of epochs for tuning
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 3, 10, step=1),
        'output_dir': os.path.join(args.output_dir, f'trial_{args.worker_id}_{trial.number}'),
    }
    
    # Create a timestamp for this trial
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory for this trial
    os.makedirs(params['output_dir'], exist_ok=True)
    
    # Create a config file for this trial
    config_file = os.path.join(params['output_dir'], f'config_{timestamp}.json')
    
    # Save trial parameters
    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Convert dict to SimpleNamespace object (similar to argparse.Namespace)
    args_obj = SimpleNamespace(**params)
    
    # Run the pipeline with the suggested hyperparameters
    try:
        _, results = run_pipeline(args_obj, config_file, timestamp, disable_progress_bar=True)
        
        # Report intermediate values if pruning is enabled
        if args.pruning and 'val_losses' in results:
            # Report the validation loss at each epoch
            for step, val_loss in enumerate(results['val_losses']):
                trial.report(val_loss, step)
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        # Return the best validation loss as the objective value
        return results['best_val_loss']
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Return a high value to indicate failure
        return float('inf')

# Create sampler with seed for reproducibility
sampler = optuna.samplers.TPESampler(seed=np.random.randint(0, 10000))

# Add pruner if enabled
pruner = optuna.pruners.MedianPruner() if args.pruning else None

# Create or load the shared study
study = optuna.create_study(
    study_name=args.study_name,
    storage=args.storage,
    load_if_exists=True,
    direction='minimize',
    sampler=sampler,
    pruner=pruner
)

print(f"Worker {args.worker_id} connected to shared study '{args.study_name}' at {args.storage}")
print(f"Running {args.n_trials} trials with {args.n_jobs} parallel jobs")

# Run the optimization
study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

# Print the best hyperparameters seen by this worker
local_best_trial = min(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]), 
                      key=lambda t: t.value if t.value is not None else float('inf'))
print("\nBest trial seen by this worker:")
print(f"  Value: {local_best_trial.value:.4f}")
print("  Params:")
for key, value in local_best_trial.params.items():
    print(f"    {key}: {value}")

# Save this worker's results
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
worker_results_file = os.path.join(args.output_dir, f'worker_results_{args.worker_id}_{timestamp}.json')

with open(worker_results_file, 'w') as f:
    trials_data = [
        {
            'number': t.number,
            'value': t.value if t.value is not None else float('inf'),
            'params': t.params,
            'state': str(t.state)
        }
        for t in study.get_trials()
    ]
    
    json.dump({
        'worker_id': args.worker_id,
        'study_name': args.study_name,
        'storage': args.storage,
        'n_trials_by_worker': args.n_trials,
        'local_best_value': local_best_trial.value,
        'local_best_params': local_best_trial.params,
        'preprocessing': args.preprocessing,
        'embedding_model': args.embedding_model,
        'timestamp': timestamp,
        'trials': trials_data
    }, f, indent=4)

print(f"Worker results saved to {worker_results_file}")

# Train a final model only if not skipped and we're the last worker
if not args.skip_final_model:
    print("\nTraining final model with best hyperparameters from this worker...")
    
    # Get best parameters from this worker's trials
    best_params = {
        'preprocessing': args.preprocessing,  # Use preprocessing from command-line args
        'embedding_model': args.embedding_model,
        'lstm_hidden_dim': local_best_trial.params['lstm_hidden_dim'],
        'fc_hidden_dim': local_best_trial.params['fc_hidden_dim'],
        'dropout_rate': local_best_trial.params['dropout_rate'],
        'batch_size': local_best_trial.params['batch_size'],
        'epochs': 50,  # Use more epochs for the final model
        'learning_rate': local_best_trial.params['learning_rate'],
        'weight_decay': local_best_trial.params['weight_decay'],
        'early_stopping_patience': local_best_trial.params['early_stopping_patience'],
        'output_dir': os.path.join(args.output_dir, f'worker_{args.worker_id}_best_model')
    }
    
    # Create output directory for final model
    os.makedirs(best_params['output_dir'], exist_ok=True)
    
    # Create a timestamp for the final model
    final_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a config file for the final model
    final_config_file = os.path.join(best_params['output_dir'], f'config_{final_timestamp}.json')
    
    # Convert dict to SimpleNamespace object for final model
    final_args = SimpleNamespace(**best_params)
    
    # Train the final model
    final_model, final_results = run_pipeline(final_args, final_config_file, final_timestamp)
    
    print(f"Worker {args.worker_id} best model saved to {best_params['output_dir']}")

print(f"Worker {args.worker_id} completed all {args.n_trials} trials!") 