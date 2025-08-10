# Iris Data saved in path
import os
import pandas as pd
from sklearn.datasets import load_iris

def save_iris_to_csv(data_path="data/raw/iris.csv"):
    """Load Iris dataset and save it as CSV."""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(data_path, index=False)

    print(f"[INFO] Iris dataset saved to {data_path}")
    return data_path

if __name__ == "__main__":
    save_iris_to_csv()
