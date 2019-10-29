import pandas as pd
import numpy as np
import csv

#Full dataset
df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')
df_all_classes = pd.read_csv('../data/y_train_smpl.csv')

#Reduced dataset
df_pixels_reduced = pd.read_csv('../output/reduced_x_train_gr_smpl.csv')

np.random.seed(100)
np.random.shuffle(df_pixels.values)
np.random.seed(100)
np.random.shuffle(df_pixels_reduced.values)
np.random.seed(100)
np.random.shuffle(df_all_classes.values)

with open('../output/shuffled_reduced_x_train_gr_smpl.csv', 'w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(df_pixels_reduced.columns)
    for el in df_pixels_reduced.values:
        csv_writer.writerow(el)

with open('../output/shuffled_x_train_gr_smpl.csv', 'w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(df_pixels.columns)
    for el in df_pixels.values:
        csv_writer.writerow(el)

with open('../output/shuffled_y_train_smpl.csv', 'w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(df_all_classes.columns)
    for el in df_all_classes.values:
        csv_writer.writerow(el)