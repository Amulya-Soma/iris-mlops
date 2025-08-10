# src/train.py
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import save_iris_to_csv

# MLflow settings
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("iris-classification")

def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """Train model, log to MLflow, and return accuracy."""
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params, metrics, and model
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[INFO] {model_name} Accuracy: {acc:.4f}")
        return model, acc, mlflow.active_run().info.run_id

if __name__ == "__main__":
    # Step 1: Ensure CSV exists
    csv_path = save_iris_to_csv()

    # Step 2: Load CSV
    df = pd.read_csv(csv_path)

    # Step 3: Split data
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train models
    results = []
    results.append(train_and_log_model(
        "LogisticRegression",
        LogisticRegression(max_iter=200),
        X_train, X_test, y_train, y_test
    ))
    results.append(train_and_log_model(
        "RandomForestClassifier",
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train, X_test, y_train, y_test
    ))
    results.append(train_and_log_model(
        "SVC",
        SVC(kernel="linear", probability=True, random_state=42),
        X_train, X_test, y_train, y_test
    ))

    # Step 5: Pick best model
    best_model_info = max(results, key=lambda x: x[1])  # (model, acc, run_id)
    best_model_name = [r for r in results if r[1] == best_model_info[1]][0]
    print(f"[INFO] Best model: {best_model_name[0].__class__.__name__} "
          f"with accuracy {best_model_info[1]:.4f}")

    # Step 6: Register best model in MLflow
    mlflow.register_model(
        f"runs:/{best_model_info[2]}/model",
        "IrisClassifier"
    )
