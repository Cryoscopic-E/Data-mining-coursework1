import pandas as pd
import numpy as np
import csv

import constants
from features_selection import select_features
from data_operations import load_dataframe, slice_img, save_dataframe_csv


def create_full_table(dataframe, file_name):
    df_classes = load_dataframe(constants.ORIGINAL_CLASSES)
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
            pixels = []
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


if __name__ == '__main__':
    df_n = load_dataframe(constants.NORMALIZED_SMPL)
    df_s = load_dataframe(constants.NORMALIZED_SLICED_SMPL)

    select_features(df_n, "NORMALIZED")
    select_features(df_s, "SLICED")

    create_full_table(df_n, 'NORMALIZED')
    create_full_table(df_s, 'SLICED')
