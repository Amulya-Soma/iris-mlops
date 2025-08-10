# src/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn

# Load model from local folder (no artifact download delay)
model = mlflow.pyfunc.load_model("model/iris_model")

# FastAPI app
app = FastAPI(title="Iris Model API")

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    input_df = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import mlflow.pyfunc
# import uvicorn

# app = FastAPI()

# class IrisFeatures(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# # Load model directly from saved folder
# model = mlflow.pyfunc.load_model("model/iris_model")

# @app.post("/predict")
# def predict(features: IrisFeatures):
#     data = pd.DataFrame([features.dict()])
#     prediction = model.predict(data)
#     return {"prediction": prediction.tolist()}

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# # # src/api/app.py
# # import os
# # import logging
# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # import pandas as pd
# # import mlflow
# # from mlflow.tracking import MlflowClient
# # import traceback

# # # Config
# # MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# # MODEL_NAME = "IrisClassifier"
# # LOG_DIR = "logs"
# # os.makedirs(LOG_DIR, exist_ok=True)

# # # Logging
# # logging.basicConfig(
# #     filename=os.path.join(LOG_DIR, "api.log"),
# #     level=logging.INFO,
# #     format="%(asctime)s - %(levelname)s - %(message)s"
# # )

# # # FastAPI app
# # app = FastAPI(title="Iris Classifier API (MLflow-backed)")

# # # Set MLflow tracking
# # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# # client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# # def resolve_production_version(model_name: str):
# #     """
# #     Find the model version tagged with stage=Production.
# #     If none found, return the latest numeric version.
# #     Returns the version as a string, or None if none exist.
# #     """
# #     try:
# #         versions = client.search_model_versions(f"name = '{model_name}'")
# #         if not versions:
# #             return None
# #     except Exception as e:
# #         logging.exception("Error while searching model versions: %s", e)
# #         return None

# #     # Normalize and inspect tags
# #     prod_versions = []
# #     numeric_versions = []
# #     for v in versions:
# #         # v may be object or dict
# #         tags = getattr(v, "tags", None) or (v.get("tags") if isinstance(v, dict) else {})
# #         version = getattr(v, "version", None) or (v.get("version") if isinstance(v, dict) else None)
# #         try:
# #             numeric_versions.append(int(version))
# #         except Exception:
# #             pass
# #         if tags and tags.get("stage") == "Production":
# #             try:
# #                 prod_versions.append(int(version))
# #             except Exception:
# #                 pass

# #     if prod_versions:
# #         selected = str(max(prod_versions))
# #         return selected

# #     # fallback: pick the highest version number
# #     if numeric_versions:
# #         selected = str(max(numeric_versions))
# #         return selected

# #     return None


# # def load_model_from_registry(model_name: str):
# #     version = resolve_production_version(model_name)
# #     if version is None:
# #         raise RuntimeError(f"No registered versions found for model '{model_name}'")

# #     model_uri = f"models:/{model_name}/{version}"
# #     logging.info("Loading model from URI: %s", model_uri)
# #     model = mlflow.pyfunc.load_model(model_uri)
# #     logging.info("Loaded model %s version %s", model_name, version)
# #     return model, version


# # # Attempt to load model at startup
# # try:
# #     model, model_version = load_model_from_registry(MODEL_NAME)
# # except Exception as e:
# #     model = None
# #     model_version = None
# #     logging.error("Failed to load model at startup: %s\n%s", e, traceback.format_exc())


# # class IrisFeatures(BaseModel):
# #     sepal_length: float
# #     sepal_width: float
# #     petal_length: float
# #     petal_width: float


# # @app.get("/")
# # def read_root():
# #     return {
# #         "message": "Iris Classifier API",
# #         "model_loaded": model is not None,
# #         "model_name": MODEL_NAME,
# #         "model_version": model_version,
# #     }


# # @app.post("/predict")
# # def predict(features: IrisFeatures):
# #     if model is None:
# #         logging.error("Prediction request but model not loaded")
# #         raise HTTPException(status_code=503, detail="Model not loaded. Check server logs / MLflow registry.")

# #     # Build DataFrame matching training column order
# #     input_df = pd.DataFrame([features.dict()])
# #     # Logging request
# #     logging.info("Request: %s", features.dict())
# #     # Predict
# #     try:
# #         preds = model.predict(input_df)
# #         pred = int(preds[0])
# #         logging.info("Prediction: %s", pred)
# #         return {"prediction": pred, "model_version": model_version}
# #     except Exception as e:
# #         logging.exception("Prediction error: %s", e)
# #         raise HTTPException(status_code=500, detail=str(e))


# # if __name__ == "__main__":
# #     # run directly with `python src/api/app.py`
# #     import uvicorn
# #     uvicorn.run(app, host="127.0.0.1", port=8001)

# # # app.py
# # import mlflow
# # import pandas as pd
# # from fastapi import FastAPI
# # from pydantic import BaseModel

# # # MLflow tracking setup
# # mlflow.set_tracking_uri("http://127.0.0.1:5000")
# # experiment_name = "iris_classification"

# # # Get latest run ID automatically
# # experiment = mlflow.get_experiment_by_name(experiment_name)
# # if experiment is None:
# #     raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

# # runs = mlflow.search_runs(
# #     experiment_ids=[experiment.experiment_id],
# #     order_by=["start_time desc"]
# # )

# # if runs.empty:
# #     raise ValueError(f"No runs found in experiment '{experiment_name}'.")

# # latest_run_id = runs.iloc[0].run_id
# # print(f"Loading latest model from run ID: {latest_run_id}")

# # # Load latest model
# # model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/model")

# # # Input data schema for API
# # class IrisFeatures(BaseModel):
# #     sepal_length: float
# #     sepal_width: float
# #     petal_length: float
# #     petal_width: float

# # # FastAPI app
# # app = FastAPI()

# # @app.post("/predict")
# # def predict(features: IrisFeatures):
# #     # Convert incoming request to DataFrame
# #     input_df = pd.DataFrame([features.dict()])
# #     input_df.columns = [
# #         "sepal length (cm)",
# #         "sepal width (cm)",
# #         "petal length (cm)",
# #         "petal width (cm)"
# #     ]
    
# #     # Make prediction
# #     prediction = model.predict(input_df)
# #     return {"prediction": int(prediction[0])}


