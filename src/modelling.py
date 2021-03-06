import pandas as pd
from joblib import dump
import yaml
from src.utils.modelling_utils import (test_train_split,
                                       split_into_XnY,
                                       model_logit_sklearn)
from ml_service.utils.env_variables import Env

e = Env()

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

traindf_with_feature_engg = pd.read_parquet(e.train_fengg_file_path)
print(traindf_with_feature_engg.head())

traindf_with_feature_engg.drop(config['DROP_COL_LIST_WITH_ONE_CLASS'],
                               axis=1, inplace=True)

traindf, validationdf = test_train_split(traindf_with_feature_engg,
                                         config['FRACTION_TRAIN'])
Xtrain, Ytrain = split_into_XnY(traindf, 'Y')
Xval, Yval = split_into_XnY(validationdf, 'Y')
Xtrain_tenfeat = Xtrain[config['TEN_BEST_FEATURES_OBSERVED_SELECTION']]
Xval_tenfeat = Xval[config['TEN_BEST_FEATURES_OBSERVED_SELECTION']]
model = model_logit_sklearn(Xtrain_tenfeat, Ytrain, Xval_tenfeat, Yval)
dump(model, e.model_file_path)
