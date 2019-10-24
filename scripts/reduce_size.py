import pandas as pd
import numpy as np
import csv

train_sample = pd.read_csv(filepath_or_buffer='./data/x_train_gr_smpl.csv', sep=',')
left = 9
right = 37

with open("./output/reduced_x_train_gr_smpl.csv", mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for flat_image in train_sample.values:
        re = np.reshape(flat_image, (48, 48))
        sub_matrix = re[9:37, 9:37]
        csv_writer.writerow(sub_matrix.flatten())


