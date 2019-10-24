import cv2
import pandas as pd
import numpy as np

img_numb = 8501

original_sample = pd.read_csv(filepath_or_buffer='./data/x_train_gr_smpl.csv', sep=',')
reduced_sample = pd.read_csv(filepath_or_buffer='./output/reduced_x_train_gr_smpl.csv', sep=',')

np_original = np.reshape(original_sample.values[img_numb],(48,48))
np_reduced = np.reshape(reduced_sample.values[img_numb],(28,28))
cv2.imwrite("./output/img/"+str(img_numb)+"_original.jpg",np_original)
cv2.imwrite("./output/img/"+str(img_numb)+"_reduced.jpg",np_reduced)