import pandas as pd
import os
import yaml
from src.utils.feature_engg_utils import replace_values_using_dict

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

current_path = os.getcwd()
# dirname, filename = os.path.split(current_path)
data_dir = os.path.join(current_path, 'files')
train_cleaned_file_path = os.path.join(data_dir,
                                       "step2\\train_cleaned.parquet")
traindf_cleaned = pd.read_parquet(train_cleaned_file_path)
print(traindf_cleaned.head())

traindf_cleaned = replace_values_using_dict(traindf_cleaned,
                                            config['dict_for_clubbing'])
traindf_cleaned = replace_values_using_dict(traindf_cleaned,
                                            config['dict_to_get_ordinal_features'])
traindf_with_feature_engg = traindf_cleaned
train_fengg_file_path = os.path.join(data_dir,
                                     "step3\\traindf_with_feature_engg.parquet"
                                     )
traindf_with_feature_engg.to_parquet(train_fengg_file_path,
                                     index=False)
