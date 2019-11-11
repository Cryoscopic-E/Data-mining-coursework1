import numpy as np
import seaborn as sn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from matplotlib import pyplot
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


import data_operations
import constants


def naive_bayes(data, classes, split=70):
    
    train_s, train_s_class, test_s, test_s_class = data_operations.split_sets(
        data, classes, split)
    gnb = GaussianNB()
    gnb.fit(train_s, train_s_class.values.ravel())
    pred = gnb.predict(test_s)
    
    return metrics.accuracy_score(test_s_class, pred)
    
def select_features(dataframe, n_feature):

    pixels_features = []

    for n in range(10):
        df_is_class_n = pd.read_csv('../data/y_train_smpl_'+str(n)+'.csv')

        transformer = SelectKBest(score_func=chi2, k = 1)

        #new_data = transformer.fit_transform(df_pixels.values, df_is_class0)
        fit = transformer.fit(dataframe, df_is_class_n)
        scores = pd.DataFrame(fit.scores_)
        columns = pd.DataFrame(dataframe.columns)
        # concat 2 dataframes for better visualization
        featuresScore = pd.concat([columns, scores], axis=1)
        featuresScore.columns = ['Pixel', 'Score']
        pixels_features.append(
            featuresScore.nlargest(n_feature, 'Score')['Pixel'].values)
    
    #count ranges
    df_classes = data_operations.load_dataframe(constants.ORIGINAL_CLASSES)
    ranges = []
    start_indx = 0
    last_indx = 0
    for n in range(10):
        count = len(np.argwhere(df_classes.values.ravel() == n))
        last_indx = last_indx + count
        ranges.append((start_indx, last_indx))
        start_indx = last_indx + 1
    #empty dataframe
    df_pixels_feature = pd.DataFrame(0, index=np.arange(len(dataframe)), columns=dataframe.columns)
    indx = 0
    for features in pixels_features:
        df_pixels_feature.loc[ranges[indx][0]: ranges[indx][1],features] = dataframe.loc[ranges[indx][0]: ranges[indx][1], features]
        indx = indx + 1
    return df_pixels_feature

if __name__ == '__main__':
    sliced = data_operations.load_dataframe(constants.NORMALIZED_SLICED_SMPL)
    classes = data_operations.load_dataframe(constants.ORIGINAL_CLASSES)
    classes = data_operations.randomize_data(classes, constants.SEED)
    max_accuracy = 0.0
    current_accuracy = 0.0
    accuracy_list = []
    feat = 1
    for n in range(len(sliced.columns)):
        features_dataframe = select_features(sliced,feat)
        features_dataframe = data_operations.randomize_data(features_dataframe, constants.SEED)
        current_accuracy = naive_bayes(features_dataframe, classes)
        accuracy_list.append(current_accuracy)
        if max_accuracy < current_accuracy:
            max_accuracy = current_accuracy
        feat +=1

    pyplot.plot(range(len(accuracy_list)),accuracy_list)
    pyplot.title('Accuracy vs n Features')
    pyplot.xlabel('No of features') 
    pyplot.ylabel('Accuracy') 
    pyplot.show()
    
        
    
    
    
    
    
    
    
    