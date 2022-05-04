import mlflow
from mlflow.tracking import MlflowClient
import sys
import os
import pandas as pd

def workflow():
  with mlflow.start_run() as active_run:
    print(sys.argv)
    input_path = sys.argv[1] if len(sys.argv)  > 1 else "False"
    # num = int(num
    # print("Hi dear")
    # new_num= num + 10
    # #metrics =mlflow.log_metrics({"num": num})
    print("Input Path is", input_path)
    print("Taking input")
    download_run = mlflow.run(".", "taking_input", parameters={"input_path": input_path})
    download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
    file_path_uri = os.path.join(download_run.info.artifact_uri, "data1.csv")
    print(file_path_uri)

    print("Starting Step1")
    step1_run = mlflow.run(".", "step1", parameters={"file_path": file_path_uri})
    process_run = mlflow.tracking.MlflowClient().get_run(step1_run.run_id)
    data_path_uri = os.path.join(download_run.info.artifact_uri, "data_path")
    par = download_run.data.metrics
    for k, v in par.items():
      print("Nice to meet you:", par)


    print("Starting Step2")
    process_run = mlflow.run(".", "step2", parameters={"data_path": data_path_uri})

if __name__ == '__main__':
    workflow()