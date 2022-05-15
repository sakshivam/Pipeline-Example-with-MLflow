import pandas as pd
import os
from joblib import dump
import yaml
from src.utils.modelling_utils import test_train_split
from src.utils.modelling_utils import split_into_XnY, model_logit_sklearn
with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

current_path = os.getcwd()
# dirname, filename = os.path.split(current_path)
data_dir = os.path.join(current_path, 'files')
train_fengg_file_path = os.path.join(data_dir,
                                     "step3\\traindf_with_feature_engg.parquet"
                                     )
traindf_with_feature_engg = pd.read_parquet(train_fengg_file_path)
print(traindf_with_feature_engg.head())

traindf_with_feature_engg.drop(config['DROP_COL_LIST_WITH_ONE_CLASS'],
                               axis=1, inplace=True)
fraction_train = 0.7
traindf, validationdf = test_train_split(traindf_with_feature_engg,
                                         fraction_train)
Xtrain, Ytrain = split_into_XnY(traindf, 'Y')
Xval, Yval = split_into_XnY(validationdf, 'Y')
Xtrain_tenfeat = Xtrain[config['TEN_BEST_FEATURES_OBSERVED_SELECTION']]
Xval_tenfeat = Xval[config['TEN_BEST_FEATURES_OBSERVED_SELECTION']]
model = model_logit_sklearn(Xtrain_tenfeat, Ytrain, Xval_tenfeat, Yval)
model_file_path = os.path.join(data_dir, "step5\\model.joblib")
dump(model, model_file_path)
