import pandas as pd
import numpy as np
import csv


df_pixels_original = pd.read_csv('../data/x_train_gr_smpl.csv')

df_classes = pd.read_csv('../data/y_train_smpl.csv')

ranges = []
start_indx = 0
last_indx = 0
for n in range(10):
    count = len(np.argwhere(df_classes.values.ravel() == n))
    last_indx = last_indx + count
    ranges.append((start_indx , last_indx))
    start_indx = last_indx + 1


for n in [2,5,10]:
    df_pixels_feature = pd.DataFrame(0,index=np.arange(len(df_pixels_original)),columns=df_pixels_original.columns)
    with open('../data/'+str(n)+'_feats_train_smpl.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        pixels = []
        indx = 0
        for row in csv_reader:
            df_pixels_feature.loc[ranges[indx][0] : ranges[indx][1], row] = df_pixels_original.loc[ranges[indx][0] : ranges[indx][1], row]
            indx = indx + 1
    
    with open('../output/smpl_'+str(n)+'_features.csv', 'w',newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(df_pixels_original.columns)
        for el in df_pixels_feature.values:
            csv_writer.writerow(el)