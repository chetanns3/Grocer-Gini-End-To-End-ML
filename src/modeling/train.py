import pathlib
import yaml
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os


def load_data(train_path, test_path):
    """Load processed train and test CSV files"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def split_features_target(df, target_col='Item_Outlet_Sales'):
    """Split dataframe into X (features) and y (target)"""
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    return X, y


def train_model(X_train, y_train, params):
    """Train XGBoost Regressor using parameters from params.yaml"""
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=params['seed'],
        n_jobs=-1,
        reg_lambda=params.get('reg_lambda', 1.0),
        reg_alpha=params.get('reg_alpha', 0.0)
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    return {"r2": r2, "rmse": rmse, "mae": mae}


def save_model(model, output_path):
    """Save trained model to disk"""
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    model_path = f"{output_path}/xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")


def save_metrics(metrics, output_path):
    """Save evaluation metrics to JSON file"""
    metrics_path = os.path.join(output_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📈 Metrics saved at: {metrics_path}")


def main():
    # Define paths
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir / "params.yaml"
    params = yaml.safe_load(open(params_file))["train"]

    train_path = home_dir / "data" / "interim" / "train.csv"
    test_path = home_dir / "data" / "interim" / "test.csv"

    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # Split features and target
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    # Train model
    model = train_model(X_train, y_train, params)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print("📊 Model Performance:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    # Save model and metrics
    save_model(model, home_dir / "models")
    save_metrics(metrics, home_dir)

    # ✅ Save feature names for inference
    feature_names_path = home_dir / "models" / "feature_names.pkl"
    joblib.dump(X_train.columns.tolist(), feature_names_path)
    print(f"✅ Feature names saved at: {feature_names_path}")
    
if __name__ == "__main__":
    main()
