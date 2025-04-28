#!/usr/bin/env python3
import argparse
import subprocess
import os
import time
import json
import datetime
import numpy as np
from pathlib import Path
import optuna

def parse_args():
    parser = argparse.ArgumentParser(description='Run distributed hyperparameter tuning')
    parser.add_argument('--preprocessing', type=str, choices=['log', 'minmax'], default='minmax',
                        help='Preprocessing method: "log" for log1p + min-max scaling, "minmax" for just min-max scaling')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers to run')
    parser.add_argument('--n_trials_per_worker', type=int, default=25, help='Number of trials per worker')
    parser.add_argument('--n_jobs_per_worker', type=int, default=1, help='Number of parallel jobs per worker')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna.db', 
                        help='Storage URL for Optuna (e.g., sqlite:///optuna.db, mysql://user:password@localhost/optuna)')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models and results')
    parser.add_argument('--study_name', type=str, default='lstm_late_fusion', help='Name of the study for Optuna')
    parser.add_argument('--embedding_model', type=str, choices=['distilbert', 'e5'], default='distilbert',
                        help='Embedding model to use')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--cpu_cores', type=int, default=128, help='Number of CPU cores available')
    parser.add_argument('--memory_per_trial', type=int, default=3.5, help='Memory per trial in GB')
    parser.add_argument('--pruning', action='store_true', help='Enable pruning of unpromising trials')
    return parser.parse_args()

def calculate_resource_allocation(args):
    """Calculate optimal resource allocation based on available hardware"""
    # Parse GPU IDs
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    n_gpus = len(gpu_ids)
    
    # Calculate available memory
    total_memory_gb = 0
    for gpu_id in gpu_ids:
        try:
            import torch
            if torch.cuda.is_available():
                total_memory_gb += torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        except:
            print(f"Warning: Could not get memory for GPU {gpu_id}")
    
    # Calculate maximum concurrent trials based on memory
    max_concurrent_trials = int(total_memory_gb / args.memory_per_trial)
    
    # Calculate CPU cores per worker
    cpu_cores_per_worker = max(1, args.cpu_cores // args.n_workers)
    
    # Adjust n_jobs_per_worker based on available resources
    n_jobs_per_worker = min(args.n_jobs_per_worker, max_concurrent_trials // args.n_workers)
    
    return {
        'gpu_ids': gpu_ids,
        'n_gpus': n_gpus,
        'total_memory_gb': total_memory_gb,
        'max_concurrent_trials': max_concurrent_trials,
        'cpu_cores_per_worker': cpu_cores_per_worker,
        'n_jobs_per_worker': n_jobs_per_worker
    }

def initialize_shared_study(args):
    """Initialize the shared study that all workers will connect to"""
    # Create a new shared study or load an existing one
    print(f"Initializing shared study '{args.study_name}' at {args.storage}")
    
    # Define the sampler with a seed for reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # Define pruner if enabled
    pruner = optuna.pruners.MedianPruner() if args.pruning else None
    
    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Get the number of existing trials if any
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"Found existing study with {existing_trials} trials")
        print(f"Best value so far: {study.best_value:.4f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("Created new study")
    
    return study

def run_worker(worker_id, gpu_id, args, resources):
    """Run a single worker for hyperparameter tuning"""
    # Create output directory for this worker
    worker_dir = os.path.join(args.output_dir, f'worker_{worker_id}')
    os.makedirs(worker_dir, exist_ok=True)
    
    # Set up environment variables for the worker
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build the command to run the tuning script
    cmd = [
        'python', 'scripts/lstm/tune_hyperparameters.py',
        '--preprocessing', args.preprocessing,
        '--worker_id', str(worker_id),
        '--gpu_id', '0',  # The worker will see only one GPU (the one specified by CUDA_VISIBLE_DEVICES)
        '--n_trials', str(args.n_trials_per_worker),
        '--n_jobs', str(resources['n_jobs_per_worker']),
        '--storage', args.storage,
        '--study_name', args.study_name,
        '--embedding_model', args.embedding_model,
        '--output_dir', worker_dir,
        '--skip_final_model'  # Skip individual worker final model training
    ]
    
    # Add pruning if enabled
    if args.pruning:
        cmd.append('--pruning')
    
    # Run the worker
    print(f"Starting worker {worker_id} on GPU {gpu_id} with {resources['n_jobs_per_worker']} jobs")
    process = subprocess.Popen(cmd, env=env)
    return process

def train_final_model(args, study, resources):
    """Train the final model using the best hyperparameters from the study"""
    # Create output directory for the final model
    final_dir = os.path.join(args.output_dir, 'final_model')
    os.makedirs(final_dir, exist_ok=True)
    
    # Determine which GPU to use for the final model (use the first available one)
    final_gpu = resources['gpu_ids'][0] if resources['gpu_ids'] else None
    if final_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(final_gpu)
        print(f"Using GPU {final_gpu} for final model training")
    
    # Create command for training the final model
    final_cmd = [
        'python', 'scripts/lstm_late_fusion.py',
        '--preprocessing', args.preprocessing,
        '--embedding_model', args.embedding_model,
        '--lstm_hidden_dim', str(study.best_params.get('lstm_hidden_dim', 64)),
        '--fc_hidden_dim', str(study.best_params.get('fc_hidden_dim', 128)),
        '--dropout_rate', str(study.best_params.get('dropout_rate', 0.3)),
        '--batch_size', str(study.best_params.get('batch_size', 32)),
        '--epochs', '50',  # Use more epochs for final model
        '--learning_rate', str(study.best_params.get('learning_rate', 1e-3)),
        '--weight_decay', str(study.best_params.get('weight_decay', 0.0)),
        '--early_stopping_patience', str(study.best_params.get('early_stopping_patience', 5)),
        '--output_dir', final_dir
    ]
    
    # Run the final training
    print("Running command:", " ".join(final_cmd))
    final_process = subprocess.run(final_cmd)
    
    if final_process.returncode == 0:
        print(f"Final model training completed successfully. Model saved to {final_dir}")
    else:
        print(f"Final model training failed with exit code {final_process.returncode}")
    
    return final_dir

# python -u scripts/run_distributed_tuning.py --output_dir models/e5_optuna_study --embedding_model e5 --n_workers 20 --n_trials_per_worker 5
# nohup python -u scripts/run_distributed_tuning.py --output_dir models/e5_optuna_study --embedding_model e5 --n_workers 20 --n_trials_per_worker 5 > nohup_e5_tuning.out 2>&1 &
def main():
    args = parse_args()
    
    # Calculate resource allocation
    resources = calculate_resource_allocation(args)
    
    # Print resource allocation
    print("Resource allocation:")
    print(f"  GPUs: {resources['n_gpus']} ({args.gpu_ids})")
    print(f"  Total GPU memory: {resources['total_memory_gb']:.1f} GB")
    print(f"  Max concurrent trials: {resources['max_concurrent_trials']}")
    print(f"  CPU cores per worker: {resources['cpu_cores_per_worker']}")
    print(f"  Jobs per worker: {resources['n_jobs_per_worker']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = os.path.join(args.output_dir, f'distributed_config_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'resources': resources,
            'timestamp': timestamp
        }, f, indent=4)
    
    # Initialize the shared study
    study = initialize_shared_study(args)
    
    # Start workers
    processes = []
    for i in range(args.n_workers):
        # Assign GPU in a round-robin fashion
        gpu_id = resources['gpu_ids'][i % resources['n_gpus']]
        process = run_worker(i, gpu_id, args, resources)
        processes.append(process)
        # Small delay to avoid race conditions
        time.sleep(1)
    
    # Wait for all workers to complete
    for i, process in enumerate(processes):
        process.wait()
        print(f"Worker {i} completed with exit code {process.returncode}")
    
    print("All workers completed!")
    
    # Reload the study to get the final results
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage
    )
    
    # Print study statistics
    print("\nStudy Statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of completed trials: {len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
    print(f"  Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    print(f"  Number of failed trials: {len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))}")
    
    # Print the best hyperparameters
    print("\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("Fixed parameters:")
    print(f"    preprocessing: {args.preprocessing}")
    print(f"    embedding_model: {args.embedding_model}")
    
    # Calculate and print hyperparameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importance:")
        for param, score in importance.items():
            print(f"  {param}: {score:.4f}")
    except:
        print("\nCould not calculate hyperparameter importance (requires at least 2 completed trials)")
    
    # Save the study results
    study_results_file = os.path.join(args.output_dir, f'study_results_{timestamp}.json')
    with open(study_results_file, 'w') as f:
        # Convert trials to serializable format
        trials_data = [
            {
                'number': t.number,
                'value': t.value if t.value is not None else None,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
        
        # Save the results
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'fixed_params': {
                'preprocessing': args.preprocessing,
                'embedding_model': args.embedding_model
            },
            'n_trials': len(study.trials),
            'n_completed': len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
            'n_pruned': len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
            'n_failed': len(study.get_trials(states=[optuna.trial.TrialState.FAIL])),
            'timestamp': timestamp,
            'importance': importance if 'importance' in locals() else None,
            'trials': trials_data
        }, f, indent=4)
    
    print(f"Study results saved to {study_results_file}")
    
    # Train the final model with the best hyperparameters
    if study.best_trial:
        print("\nTraining final model with best hyperparameters...")
        final_model_dir = train_final_model(args, study, resources)
        print(f"Hyperparameter tuning completed! Final model saved to {final_model_dir}")
    else:
        print("\nNo successful trials found. Skipping final model training.")

if __name__ == '__main__':
    main() 