import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Rename columns to fixed names
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create folder for saved model
os.makedirs("model", exist_ok=True)

# Save model locally with MLflow
mlflow.sklearn.save_model(model, path="model/iris_model")

print("Model trained and saved at model/iris_model")


# # # src/train.py
# # import time
# # import pandas as pd
# # import mlflow
# # import mlflow.sklearn
# # from mlflow.tracking import MlflowClient
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score
# # from utils import save_iris_to_csv   # expects src/utils.py to provide this
# # import os

# # MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# # EXPERIMENT_NAME = "Iris_Models"
# # REGISTERED_MODEL_NAME = "IrisClassifier"
# # CSV_PATH = "data/raw/iris.csv"


# # def main():
# #     # Ensure data CSV exists
# #     save_iris_to_csv(CSV_PATH)

# #     # MLflow setup
# #     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# #     mlflow.set_experiment(EXPERIMENT_NAME)
# #     client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# #     # Load data
# #     df = pd.read_csv(CSV_PATH)
# #     feature_cols = [c for c in df.columns if c != "target"]
# #     X = df[feature_cols]
# #     y = df["target"]

# #     # Split
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=42
# #     )

# #     # Models to try
# #     models = {
# #         "LogisticRegression": LogisticRegression(max_iter=200),
# #         "DecisionTree": DecisionTreeClassifier(random_state=42),
# #         "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
# #     }

# #     results = []
# #     for name, model in models.items():
# #         with mlflow.start_run(run_name=name) as run:
# #             # Train
# #             model.fit(X_train, y_train)

# #             # Predict & metric
# #             preds = model.predict(X_test)
# #             acc = float(accuracy_score(y_test, preds))

# #             # Log params (safe subset), metrics, and model
# #             mlflow.log_param("model_name", name)
# #             try:
# #                 # only log simple param types to avoid serialization problems
# #                 safe_params = {k: v for k, v in model.get_params().items()
# #                                if isinstance(v, (int, float, str, bool))}
# #                 if safe_params:
# #                     mlflow.log_params(safe_params)
# #             except Exception:
# #                 pass

# #             mlflow.log_metric("accuracy", acc)

# #             # input_example as DataFrame to help MLflow infer signature
# #             input_example = X_train.head(1)

# #             # Log & register model (registered_model_name will create a model version)
# #             mlflow.sklearn.log_model(
# #                 sk_model=model,
# #                 artifact_path="model",               # artifact path under the run
# #                 registered_model_name=REGISTERED_MODEL_NAME,
# #                 input_example=input_example
# #             )

# #             results.append({"name": name, "accuracy": acc, "run_id": run.info.run_id})
# #             print(f"[INFO] {name} logged. run_id={run.info.run_id} accuracy={acc:.4f}")

# #     # Pick best model by validation accuracy
# #     best = max(results, key=lambda r: r["accuracy"])
# #     best_run_id = best["run_id"]
# #     print(f"[INFO] Best model: {best['name']} (run_id={best_run_id}) accuracy={best['accuracy']:.4f}")

# #     # Find the model version corresponding to the best_run_id (retry a few times)
# #     model_version = None
# #     for attempt in range(10):
# #         try:
# #             candidate_versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
# #         except Exception:
# #             candidate_versions = []
# #         for v in candidate_versions:
# #             # v may be a dict or an object with attributes
# #             v_run_id = getattr(v, "run_id", None) or (v.get("run_id") if isinstance(v, dict) else None)
# #             v_version = getattr(v, "version", None) or (v.get("version") if isinstance(v, dict) else None)
# #             if v_run_id == best_run_id:
# #                 model_version = str(v_version)
# #                 break
# #         if model_version:
# #             break
# #         time.sleep(1.0)

# #     # If not found, explicitly register the model from the run artifact
# #     if model_version is None:
# #         print("[INFO] Could not find existing registered version for the best run; explicitly registering...")
# #         mv = mlflow.register_model(f"runs:/{best_run_id}/model", REGISTERED_MODEL_NAME)
# #         # mv should have .version
# #         model_version = str(getattr(mv, "version", None) or mv.version)

# #     print(f"[INFO] Best model registered as version {model_version}")

# #     # Tag the model version with stage=Production (safe; avoids deprecated stage-transition API)
# #     try:
# #         client.set_model_version_tag(
# #             name=REGISTERED_MODEL_NAME,
# #             version=model_version,
# #             key="stage",
# #             value="Production"
# #         )
# #         print(f"[INFO] Tagged model {REGISTERED_MODEL_NAME} v{model_version} with stage=Production")
# #     except Exception as e:
# #         print(f"[WARN] Could not set model version tag: {e}")

# #     print("[DONE]")


# # if __name__ == "__main__":
# #     main()


# # # train.py
# # import pandas as pd
# # import mlflow
# # import mlflow.sklearn
# # from sklearn.datasets import load_iris
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score

# # if __name__ == "__main__":
# #     # Load dataset
# #     iris = load_iris(as_frame=True)
# #     X = iris.data
# #     y = iris.target

# #     # Train-test split
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, random_state=42
# #     )

# #     # Model
# #     clf = RandomForestClassifier(n_estimators=100, random_state=42)
# #     clf.fit(X_train, y_train)

# #     # Evaluate
# #     preds = clf.predict(X_test)
# #     acc = accuracy_score(y_test, preds)
# #     print(f"Accuracy: {acc}")

# #     # Log model to MLflow
# #     mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Adjust if needed
# #     mlflow.set_experiment("iris_classification")

# #     with mlflow.start_run():
# #         mlflow.log_metric("accuracy", acc)
# #         mlflow.sklearn.log_model(clf, artifact_path="model")
