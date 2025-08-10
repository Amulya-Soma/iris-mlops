import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
import os

# Paths
DATA_PATH = "data/raw/iris.csv"
MODEL_DIR = "src/model/iris_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Automatically detect target column as last column
    target_col = df.columns[-1]
    print(f"Detected target column: {target_col}")

    X = df.iloc[:, :-1]
    y = df[target_col]
    return X, y

def log_all_metrics(y_true, y_pred):
    """Logs multiple classification metrics to MLflow."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    # Log scalar metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", prec)
    mlflow.log_metric("recall_weighted", rec)
    mlflow.log_metric("f1_weighted", f1)

    # Log confusion matrix as artifact
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_path = "confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=False)
    mlflow.log_artifact(cm_path)

    return acc

def train_and_log_models(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("iris_classification_Final")

    results = []

    # Logistic Regression
    with mlflow.start_run(run_name="logistic_regression"):
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        acc = log_all_metrics(y_test, preds)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.sklearn.log_model(lr, "model")

        results.append(("LogisticRegression", lr, acc))

    # Random Forest
    with mlflow.start_run(run_name="random_forest"):
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = log_all_metrics(y_test, preds)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.sklearn.log_model(rf, "model")

        results.append(("RandomForestClassifier", rf, acc))

    return results

def save_best_model(results):
    best_model_name, best_model, best_acc = max(results, key=lambda x: x[2])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model: {best_model_name} (Accuracy: {best_acc:.4f}) saved to {MODEL_PATH}")

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = train_and_log_models(X_train, X_test, y_train, y_test)
    save_best_model(results)