import numpy as np
import seaborn as sn
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt

import data_operations
import constants


def naive_bayes(data, classes, outfile_name, split=70):
    """
    The metod create an output file (output/ folder) containing:
        - classification report
        - confusion matric
        - accuracy score
    """

    train_s, train_s_class, test_s, test_s_class = data_operations.split_sets(
        data, classes, split)
    gnb = GaussianNB()
    gnb.fit(train_s, train_s_class.values.ravel())
    pred = gnb.predict(test_s)

    with open(constants.NAIVE_BAYES_REPORT_PATH+outfile_name+".txt", "w") as out_text:
        out_text.write(metrics.classification_report(test_s_class, pred))
        out_text.write('\n')
        cm = metrics.confusion_matrix(test_s_class, pred)
        out_text.write(np.array2string(cm))
        out_text.write('\n\nAccuracy score: ' +
                       str(metrics.accuracy_score(test_s_class, pred)))
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.show()
    pass


if __name__ == '__main__':
    #normalized_full, normalized_sliced = data_operations.load_normalized()
    classes = data_operations.load_dataframe(constants.ORIGINAL_CLASSES)
    classes = data_operations.randomize_data(classes, constants.SEED)

    f_5_full = data_operations.load_dataframe(
        constants.FEATURES_N_SMPL_PATH+'5_NORMALIZED.csv')
    f_5_full = data_operations.randomize_data(f_5_full, constants.SEED)
    naive_bayes(f_5_full, classes, 'report_5_features_normalized_full')
