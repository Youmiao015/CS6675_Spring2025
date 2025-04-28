import optuna
import pandas as pd

def view_study_results():
    # Create a study object from the existing database
    study = optuna.load_study(
        study_name="lstm_late_fusion",
        storage="sqlite:///optuna.db"
    )
    
    # Get the best trial
    best_trial = study.best_trial
    
    print("\n=== Best Trial Results ===")
    print(f"Best Trial Number: {best_trial.number}")
    print(f"Best Value: {best_trial.value}")
    print("\nBest Parameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
    
    # Get all trials and create a DataFrame
    trials_df = study.trials_dataframe()
    print("\n=== Study Statistics ===")
    print(f"Number of Trials: {len(trials_df)}")
    print(f"Best Value: {study.best_value}")
    print(f"Best Parameters: {study.best_params}")

if __name__ == "__main__":
    view_study_results() 