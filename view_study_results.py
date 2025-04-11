import optuna

def main():
    # Replace with your actual study name and storage path
    study_name = 'lstm_late_fusion'
    storage = 'sqlite:///optuna.db'

    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Get the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Get parameter importances
    importances = optuna.importance.get_param_importances(study)
    print("\nParameter importances:")
    for param, importance in importances.items():
        print(f"  {param}: {importance}")

if __name__ == "__main__":
    main()