import pandas as pd
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def write_first_n_features(n,array):
    #with open('../data/'+str(n)+'_feats_train_smpl.csv', 'a',newline='') as csv_file:
    with open('../data/'+str(n)+'_feats_train_smpl_reduced.csv', 'a',newline='') as csv_file: 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(array[:n])
            

#full dataset
#df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')

#sliced dataset
df_pixels = pd.read_csv('../output/reduced_x_train_gr_smpl.csv')

pixels_features = []
for n in range(10):
    df_is_class_n = pd.read_csv('../data/y_train_smpl_'+str(n)+'.csv')

    transformer = SelectKBest(score_func=chi2, k=10)

    #new_data = transformer.fit_transform(df_pixels.values, df_is_class0)
    fit = transformer.fit(df_pixels, df_is_class_n)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(df_pixels.columns)
    #concat 2 dataframes for better visualization
    featuresScore = pd.concat([columns,scores],axis=1)
    featuresScore.columns = ['Pixel','Score']
    pixels_features.append(featuresScore.nlargest(10,'Score')['Pixel'].values)


for features in pixels_features:
    write_first_n_features(2,features)
    write_first_n_features(5,features)
    write_first_n_features(10,features)