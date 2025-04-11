import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import LinearRegression


def add_advanced_features(df: pd.DataFrame, input_len: int):
    feature_cols = [f"f{i}" for i in range(input_len)]

    x_idx = np.arange(input_len).reshape(-1, 1)
    trends = []
    for row in df[feature_cols].values:
        reg = LinearRegression().fit(x_idx, row)
        trends.append(reg.coef_[0])
    df["trend"] = trends

    df["last_over_mean"] = df[f"f{input_len-1}"] / (df[feature_cols].mean(axis=1) + 1e-8)
    df["diff_mean"] = df[feature_cols].diff(axis=1).mean(axis=1)

    return df

def add_monster_features(df, input_len):
    fcols = [f"f{i}" for i in range(input_len)]

    df["mean"] = df[fcols].mean(axis=1)
    df["std"] = df[fcols].std(axis=1)
    df["min"] = df[fcols].min(axis=1)
    df["max"] = df[fcols].max(axis=1)
    df["last_minus_first"] = df[f"f{input_len-1}"] - df["f0"]
    df["sum"] = df[fcols].sum(axis=1)
    df["median"] = df[fcols].median(axis=1)
    df["range"] = df["max"] - df["min"]
    df["cv"] = df["std"] / (df["mean"] + 1e-8)

    x_idx = np.arange(input_len).reshape(-1, 1)
    trends = []
    for row in df[fcols].values:
        reg = LinearRegression().fit(x_idx, row)
        trends.append(reg.coef_[0])
    df["trend"] = trends

    return df

train_df = pd.read_csv("data/train_data_new.csv", index_col=0)
val_df = pd.read_csv("data/val_data_new.csv", index_col=0)
test_df = pd.read_csv("data/test_data_new.csv", index_col=0)

input_len = 5
output_len = 3
random_seed = 42

def make_sliding_samples(df, input_len, output_len):
    df.columns = df.columns.astype(str)
    X, Y = [], []
    for idx in df.index:
        values = df.loc[idx].values.astype(float)
        for i in range(len(values) - input_len - output_len + 1):
            x = values[i : i + input_len]
            y = values[i + input_len : i + input_len + output_len]
            X.append(x)
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    feature_cols = [f"f{i}" for i in range(input_len)]
    target_cols = [f"y{i+1}" for i in range(output_len)]
    data = pd.DataFrame(np.hstack([X, Y]), columns=feature_cols + target_cols)
    return data


train_data = make_sliding_samples(train_df, input_len, output_len)
val_data = make_sliding_samples(val_df, input_len, output_len)
test_data = make_sliding_samples(test_df, input_len, output_len)

def make_test_2024_samples(df, input_len, output_len):
    df.columns = df.columns.astype(str)
    index_2022 = df.columns.get_loc("2022")
    X, Y = [], []
    for idx in df.index:
        values = df.loc[idx].values.astype(float)
        if len(values) >= input_len + output_len:
            x = values[index_2022-input_len:index_2022]  # Use years before 2022 as input
            y = values[index_2022:index_2022+output_len]  # Use years 2022, 2023, and 2024 as output
            X.append(x)
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    feature_cols = [f"f{i}" for i in range(input_len)]
    target_cols = [f"y{i+1}" for i in range(output_len)]
    data = pd.DataFrame(np.hstack([X, Y]), columns=feature_cols + target_cols)
    return data

test_2024_data = make_test_2024_samples(test_df, input_len, output_len)
test_2024_data

def add_stat_features(df: pd.DataFrame, input_len: int):
    feature_cols = [f"f{i}" for i in range(input_len)]
    df["mean"] = df[feature_cols].mean(axis=1)
    df["std"] = df[feature_cols].std(axis=1)
    df["min"] = df[feature_cols].min(axis=1)
    df["max"] = df[feature_cols].max(axis=1)
    df["last_minus_first"] = df[f"f{input_len-1}"] - df["f0"]
    return df

train_data = add_stat_features(train_data, input_len)
val_data = add_stat_features(val_data, input_len)
test_data = add_stat_features(test_data, input_len)
test_2024_data = add_stat_features(test_2024_data, input_len)

train_data = add_advanced_features(train_data, input_len)
val_data = add_advanced_features(val_data, input_len)
test_data = add_advanced_features(test_data, input_len)
test_2024_data = add_advanced_features(test_2024_data, input_len)

train_data = add_monster_features(train_data, input_len)
val_data = add_monster_features(val_data, input_len)
test_data = add_monster_features(test_data, input_len)
test_2024_data = add_monster_features(test_2024_data, input_len)

X_train = train_data.drop(columns=["y1", "y2", "y3"])
y_train = train_data[["y1", "y2", "y3"]]
X_val = val_data.drop(columns=["y1", "y2", "y3"])
y_val = val_data[["y1", "y2", "y3"]]
X_test = test_data.drop(columns=["y1", "y2", "y3"])
y_test = test_data[["y1", "y2", "y3"]]
X_test_2024 = test_2024_data.drop(columns=["y1", "y2", "y3"])
y_test_2024 = test_2024_data[["y1", "y2", "y3"]]

def train_one_target(y_train, y_val, y_test, label):
    model = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=10,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_weight=1,
        objective="reg:squarederror",
        random_state=random_seed,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    preds = model.predict(X_test_2024)
    mse = mean_squared_error(y_test, preds)
    print(f"[{label}] Test RMSE: {mse ** 0.5:.4f}")
    return model

model_y1 = train_one_target(y_train["y1"], y_val["y1"], y_test_2024["y1"], "T+1")
model_y2 = train_one_target(y_train["y2"], y_val["y2"], y_test_2024["y2"], "T+2")
model_y3 = train_one_target(y_train["y3"], y_val["y3"], y_test_2024["y3"], "T+3")