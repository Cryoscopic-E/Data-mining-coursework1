import pandas as pd
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import data_operations
import constants


def write_first_n_features(n, array, file_name):
    with open(constants.N_FEATURES_PATH+str(n)+'_'+file_name+'.csv', 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(array[:n])


def select_features(dataframe, name):

    pixels_features = []

    for n in range(10):
        df_is_class_n = pd.read_csv('../data/y_train_smpl_'+str(n)+'.csv')

        transformer = SelectKBest(score_func=chi2, k=10)

        #new_data = transformer.fit_transform(df_pixels.values, df_is_class0)
        fit = transformer.fit(dataframe, df_is_class_n)
        scores = pd.DataFrame(fit.scores_)
        columns = pd.DataFrame(dataframe.columns)
        # concat 2 dataframes for better visualization
        featuresScore = pd.concat([columns, scores], axis=1)
        featuresScore.columns = ['Pixel', 'Score']
        pixels_features.append(
            featuresScore.nlargest(10, 'Score')['Pixel'].values)

    for features in pixels_features:
        write_first_n_features(2, features, name)
        write_first_n_features(5, features, name)
        write_first_n_features(10, features, name)
