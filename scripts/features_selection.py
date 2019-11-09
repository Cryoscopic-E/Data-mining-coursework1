import pandas as pd
import numpy as np
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

def create_full_table(dataframe, file_name):
    df_classes = data_operations.load_dataframe(constants.ORIGINAL_CLASSES)
    ranges = []
    start_indx = 0
    last_indx = 0
    for n in range(10):
        count = len(np.argwhere(df_classes.values.ravel() == n))
        last_indx = last_indx + count
        ranges.append((start_indx, last_indx))
        start_indx = last_indx + 1
    for n in [2, 5, 10]:
        df_pixels_feature = pd.DataFrame(0, index=np.arange(
            len(dataframe)), columns=dataframe.columns)
        with open(constants.N_FEATURES_PATH+str(n)+'_'+file_name+'.csv') as csv_file:
            csv_reader = csv.reader(csv_file)
            indx = 0
            for row in csv_reader:
                df_pixels_feature.loc[ranges[indx][0]: ranges[indx][1],
                                      row] = dataframe.loc[ranges[indx][0]: ranges[indx][1], row]
                indx = indx + 1

        with open(constants.FEATURES_N_SMPL_PATH+str(n)+'_'+file_name+'.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(dataframe.columns)
            for el in df_pixels_feature.values:
                csv_writer.writerow(el)

