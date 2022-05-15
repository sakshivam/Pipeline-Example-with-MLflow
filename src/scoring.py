import pandas as pd
import os
import yaml
from src.utils.scoring_utils import Fill_missing_Val_Columns
from src.utils.scoring_utils import Drop_Missing_Val_Columns
from src.utils.scoring_utils import replace_values_using_dict
from src.utils.scoring_utils import predictions

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

current_path = os.getcwd()
# dirname, filename = os.path.split(current_path)
data_dir = os.path.join(current_path, 'files')
test_file_path = os.path.join(data_dir, "step1\\test.parquet")
testdf = pd.read_parquet(test_file_path)
print(testdf.head())

testdf = Fill_missing_Val_Columns(testdf,
                                  config['MISSING_VAL_COLUMNS_TO_FILL'])
testdf = Drop_Missing_Val_Columns(testdf, config['MISSING_VAL_COLUMNS'])

testdf = replace_values_using_dict(testdf, config['dict_for_clubbing'])
testdf = replace_values_using_dict(testdf,
                                   config['dict_to_get_ordinal_features'])
model_file_path = os.path.join(data_dir, "step5\\model.joblib")
result_df = predictions(testdf, 'Y',
                        config['TEN_BEST_FEATURES_OBSERVED_SELECTION'],
                        model_file_path)
predicted_df_file_path = os.path.join(data_dir,
                                      "step6\\predicted_data.parquet")
result_df.to_parquet(predicted_df_file_path, index=False)
