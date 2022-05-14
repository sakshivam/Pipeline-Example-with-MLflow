import mlflow
from mlflow.tracking import MlflowClient
import sys
import os
import pandas as pd

def workflow():
  with mlflow.start_run() as active_run:
    print(sys.argv)
    input_path = sys.argv[1] if len(sys.argv)  > 1 else "False"
    print("Input Path is", input_path)
    print("Taking input")
    download_run = mlflow.run(".", "taking_input", parameters={"input_path": input_path})
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(download_run.run_id)
    file_path_uri = os.path.join(run.info.artifact_uri, "data1.csv")
    print(file_path_uri)
    artifact = mlflow.artifacts.download_artifacts(file_path_uri)
    print("Artifact saved ", artifact)

    print("Starting Step1")
    step1_run = mlflow.run(".", "step1", parameters={"file_path": artifact})
    process_run = mlflow.tracking.MlflowClient().get_run(step1_run.run_id)
    data_path_uri = os.path.join(process_run.info.artifact_uri, "data_path")

if __name__ == '__main__':
    workflow()