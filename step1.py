import mlflow
import sys
import pandas as pd

def task():
  with mlflow.start_run() as mlrun:
    # logic of the step goes here
    print('Step1 executed')
    input_path=sys.argv[1]
    df = pd.read_csv(input_path)
    print(df.head())
    print("Step1 executed successfully")

if __name__ == '__main__':
    task()
    

