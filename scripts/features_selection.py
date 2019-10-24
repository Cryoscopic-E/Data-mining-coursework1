import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')

for n in range(10):
    df_is_class_n = df_is_class0 = pd.read_csv('../data/y_train_smpl_'+str(n)+'.csv')

    transformer = SelectKBest(score_func=chi2, k=10)

    #new_data = transformer.fit_transform(df_pixels.values, df_is_class0)
    fit = transformer.fit(df_pixels, df_is_class_n)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(df_pixels.columns)
    #concat 2 dataframes for better visualization
    featuresScore = pd.concat([columns,scores],axis=1)
    featuresScore.columns = ['Pixel','Score']
    
    print(featuresScore.nlargest(10,'Score')['Pixel'])