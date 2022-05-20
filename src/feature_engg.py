import pandas as pd
import yaml
from src.utils.feature_engg_utils import replace_values_using_dict
from ml_service.utils.env_variables import Env

e = Env()

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

traindf_cleaned = pd.read_parquet(e.train_cleaned_file_path)
print(traindf_cleaned.head())

traindf_cleaned = replace_values_using_dict(traindf_cleaned,
                                            config['dict_for_clubbing'])
traindf_cleaned = replace_values_using_dict(traindf_cleaned,
                                            config['dict_to_get_ordinal_features'])
traindf_with_feature_engg = traindf_cleaned
traindf_with_feature_engg.to_parquet(e._train_fengg_file_path,
                                     index=False)
