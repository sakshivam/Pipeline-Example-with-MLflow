def task():
  with mlflow.start_run() as mlrun:
        print('Hello World')
        print("Step2 executed successfully")