import pandas as pd
import os
import yaml
from src.utils.feature_selection_utils import spearman_corr
from src.utils.feature_selection_utils import convert_result_series_to_df
from src.utils.feature_selection_utils import chisquare_test
from src.utils.feature_selection_utils import splitdf_into_Xtrain_n_Ytrain
from src.utils.feature_selection_utils import mutualinfo_values
from src.utils.feature_selection_utils import woe_iv_values, Analysis_Report

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

res_spear_coef = spearman_corr(traindf_with_feature_engg, 'Y')
spear_coef_df = convert_result_series_to_df(res_spear_coef,
                                            colname_list=['spearman_coefficient'])

lst_score_series = chisquare_test(traindf_with_feature_engg, 'Y')
chi_square_score_df = convert_result_series_to_df(lst_score_series,
                                                  ['chi_square_scores'])

Xtrain, Ytrain = splitdf_into_Xtrain_n_Ytrain(traindf_with_feature_engg, 'Y')

mutual_info_series = mutualinfo_values(Xtrain, Ytrain)
mutual_info_df = convert_result_series_to_df(mutual_info_series,
                                             ['mutual_info_values'])

weight_of_evidence_Info_series = woe_iv_values(Xtrain, Ytrain)
WOE_IV_Sum_df = convert_result_series_to_df(weight_of_evidence_Info_series,
                                            ['WOE_IV_values'])

dfs_list = [spear_coef_df[:-1], chi_square_score_df[:-1],
            mutual_info_df, WOE_IV_Sum_df]
# key = mutual_info_df.index
Analysisdf = Analysis_Report(dfs_list)

analysis_report_file_path = os.path.join(data_dir,
                                         "step4\\Analysis_report.parquet")
Analysisdf.to_parquet(analysis_report_file_path, index=False)
