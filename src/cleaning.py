import pandas as pd
import os
import yaml
from src.utils.cleaning_utils import Fill_missing_Val_Columns
from src.utils.cleaning_utils import Drop_Missing_Val_Columns


with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')


current_path = os.getcwd()
# dirname, filename = os.path.split(current_path)
data_dir = os.path.join(current_path, 'files')
print('data_dir: ', data_dir)
train_file_path = os.path.join(data_dir, "step1\\train.parquet")
traindf = pd.read_parquet(train_file_path, engine='pyarrow')
print(traindf.head())

traindf = Fill_missing_Val_Columns(traindf,
                                   config['MISSING_VAL_COLUMNS_TO_FILL'])
traindf = Drop_Missing_Val_Columns(traindf, config['MISSING_VAL_COLUMNS'])
traindf_cleaned = traindf
train_cleaned_file_path = os.path.join(data_dir,
                                       "step2\\train_cleaned.parquet")
traindf_cleaned.to_parquet(train_cleaned_file_path, index=False)
