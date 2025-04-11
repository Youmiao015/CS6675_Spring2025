#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate predictions against ground truth')
    parser.add_argument('--pred_csv', type=str, required=True,
                        help='Path to the prediction CSV file')
    parser.add_argument('--ground_truth_csv', type=str, required=True,
                        help='Path to the ground truth CSV file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results (optional)')
    parser.add_argument('--detailed_csv', type=str, default=None,
                        help='Path to save a detailed CSV with predictions, ground truth, and differences (optional)')
    return parser.parse_args()

def load_data(pred_path, ground_truth_path):
    """Load prediction and ground truth data from CSV files."""
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    # Load prediction data
    pred_df = pd.read_csv(pred_path)
    
    # Load ground truth data
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    return pred_df, ground_truth_df

def get_target_column(pred_df, ground_truth_df):
    """Get the target column name from the prediction file."""
    # The prediction file should have exactly two columns: 'topic' and the target column
    if len(pred_df.columns) != 2:
        raise ValueError(f"Prediction file should have exactly 2 columns, but found {len(pred_df.columns)}")
    
    # The target column is the one that's not 'topic'
    target_column = [col for col in pred_df.columns if col != 'topic'][0]
    
    # Check if the target column exists in the ground truth file
    if target_column not in ground_truth_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in ground truth file")
    
    return target_column

def calculate_metrics(pred_df, ground_truth_df, target_column):
    """Calculate MSE and RMSE between predictions and ground truth."""
    # Merge prediction and ground truth data on 'topic'
    merged_df = pd.merge(pred_df, ground_truth_df[['topic', target_column]], 
                         on='topic', how='inner', suffixes=('_pred', '_true'))
    
    if len(merged_df) == 0:
        raise ValueError("No matching topics found between prediction and ground truth files")
    
    # Calculate metrics
    mse = np.mean((merged_df[f'{target_column}_pred'] - merged_df[f'{target_column}_true']) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate metrics in log scale (if values are positive)
    if (merged_df[f'{target_column}_pred'] > 0).all() and (merged_df[f'{target_column}_true'] > 0).all():
        log_mse = np.mean((np.log(merged_df[f'{target_column}_pred']) - np.log(merged_df[f'{target_column}_true'])) ** 2)
        log_rmse = np.sqrt(log_mse)
    else:
        log_mse = None
        log_rmse = None
    
    # Calculate relative error metrics
    mape = np.mean(np.abs((merged_df[f'{target_column}_true'] - merged_df[f'{target_column}_pred']) / merged_df[f'{target_column}_true'])) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'log_mse': log_mse,
        'log_rmse': log_rmse,
        'mape': mape,
        'num_samples': len(merged_df),
        'merged_data': merged_df,
        'target_column': target_column
    }

def save_results(results, output_file, detailed_csv=None):
    """Save evaluation results to a file."""
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"MSE: {results['mse']:.4f}\n")
            f.write(f"RMSE: {results['rmse']:.4f}\n")
            if results['log_mse'] is not None:
                f.write(f"Log MSE: {results['log_mse']:.4f}\n")
                f.write(f"Log RMSE: {results['log_rmse']:.4f}\n")
            f.write(f"MAPE: {results['mape']:.2f}%\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
        
        # Save detailed comparison to CSV
        detailed_file = output_file.replace('.txt', '_detailed.csv')
        results['merged_data'].to_csv(detailed_file, index=False)
        print(f"Detailed comparison saved to {detailed_file}")
    
    # Save a more detailed CSV with predictions, ground truth, and differences
    if detailed_csv:
        # Create a new DataFrame with the desired columns
        detailed_df = pd.DataFrame({
            'topic': results['merged_data']['topic'],
            'predicted': results['merged_data'][f"{results['target_column']}_pred"],
            'ground_truth': results['merged_data'][f"{results['target_column']}_true"],
            'absolute_difference': np.abs(results['merged_data'][f"{results['target_column']}_pred"] - 
                                         results['merged_data'][f"{results['target_column']}_true"]),
            'percentage_difference': np.abs((results['merged_data'][f"{results['target_column']}_pred"] - 
                                           results['merged_data'][f"{results['target_column']}_true"]) / 
                                          results['merged_data'][f"{results['target_column']}_true"] * 100)
        })
        
        # Sort by absolute difference (descending) to highlight the largest errors
        detailed_df = detailed_df.sort_values('absolute_difference', ascending=False)
        
        # Save to CSV
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"Detailed CSV with predictions, ground truth, and differences saved to {detailed_csv}")

def main():
    args = parse_args()
    
    # Load data
    pred_df, ground_truth_df = load_data(args.pred_csv, args.ground_truth_csv)
    print(f"Loaded prediction file: {args.pred_csv}")
    print(f"Loaded ground truth file: {args.ground_truth_csv}")
    
    # Get target column
    target_column = get_target_column(pred_df, ground_truth_df)
    print(f"Target column: {target_column}")
    
    # Calculate metrics
    results = calculate_metrics(pred_df, ground_truth_df, target_column)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    if results['log_mse'] is not None:
        print(f"Log MSE: {results['log_mse']:.4f}")
        print(f"Log RMSE: {results['log_rmse']:.4f}")
    print(f"MAPE: {results['mape']:.2f}%")
    print(f"Number of samples: {results['num_samples']}")
    
    # Save results if output file is specified
    if args.output_file:
        save_results(results, args.output_file, args.detailed_csv)
        print(f"Results saved to {args.output_file}")
    elif args.detailed_csv:
        # If only detailed CSV is requested
        save_results(results, None, args.detailed_csv)

if __name__ == "__main__":
    main() 