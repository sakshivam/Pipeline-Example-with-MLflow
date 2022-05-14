import mlflow
import sys
import csv
import tempfile
import os
import pandas as pd


def load_raw_data():
    with mlflow.start_run() as mlrun:
        input_path=sys.argv[1]
        print('Input Path', input_path)
        df = pd.read_csv(input_path)
        print('Dataframe', df.shape)
        
        local_dir = tempfile.mkdtemp()
        data_file = os.path.join(local_dir, "data1.csv")
        print('Dataframe Path', data_file)
        df.to_csv(data_file)
        
        print("Uploading ratings: %s" % data_file)
        mlflow.log_artifact(data_file)
    
if __name__ == '__main__':
    load_raw_data()
