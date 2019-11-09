import pandas as pd
import numpy as np
import csv
import cv2
from os import path
import constants
from progress.bar import Bar
from math import sqrt


def load_dataframe(path_csv):
    """
    Load a single dataframe from an csv file
    """
    return pd.read_csv(path_csv)
    pass


def load_normalized():
    """
    Returns all data sets we need:
        - Normalized_Full
        - Normalized_Sliced
        - Normalized_Full (rnd)
        - Normalized_Sliced (rnd)
    """

    # LOAD NORMALIZED DATA
    if not path.exists(constants.NORMALIZED_SMPL):
        print('Normalized samples file not found.')
        print('Creating Normalized Dataframe')
        original = load_dataframe(constants.ORIGINAL_X_SMPL)
        norm = normalize(original)
        save_dataframe_csv(norm, constants.NORMALIZED_SMPL)
    else:
        norm = load_dataframe(constants.NORMALIZED_SMPL)
    # LOAD NORMALIZED SLICED SAMPLE
    if not path.exists(constants.NORMALIZED_SLICED_SMPL):
        print('Normalized sliced samples file not found.')
        print('Creating Normalized Sliced Dataframe')
        norm_slc = slice_img(norm)
        save_dataframe_csv(norm_slc, constants.NORMALIZED_SLICED_SMPL)
    else:
        norm_slc = load_dataframe(constants.NORMALIZED_SLICED_SMPL)
    # LOAD NORMALIZED DATA RANDOM OR CREATE IT
    if not path.exists(constants.NORMALIZED_FULL_RND_SMPL):
        print('Normalized random samples file not found.')
        print('Creating Normalized Random Dataframe')
        norm_f_rnd = randomize_data(norm, constants.SEED)
        save_dataframe_csv(norm_f_rnd, constants.NORMALIZED_FULL_RND_SMPL)
    else:
        norm_f_rnd = load_dataframe(constants.NORMALIZED_FULL_RND_SMPL)

    # LOAD NORMALIZED SLICED DATA RANDOM OR CREATE IT
    if not path.exists(constants.NORMALIZED_SLICED_RND_SMPL):
        print('Normalized random samples (sliced) file not found.')
        print('Creating Normalized (sliced) Random Dataframe')
        norm_slc_rnd = randomize_data(norm_slc, constants.SEED)
        save_dataframe_csv(norm_slc_rnd, constants.NORMALIZED_SLICED_RND_SMPL)
    else:
        norm_slc_rnd = load_dataframe(constants.NORMALIZED_SLICED_RND_SMPL)

    return (norm, norm_slc, norm_f_rnd, norm_slc_rnd)


def split_sets(data_set, class_set, train_set_perc):
    """
    Split the input data in training sets and test sets
    train_set_perc is an integer number 0-100 defining the persentage of the input data
    that will treated as training data

    Return a python set containg in order:
        - training set
        - training set classes
        - test set 
        - test set classes
    """
    train_set_perc = max(min(train_set_perc, 100), 0)
    n_tr_set = int(train_set_perc/100 * len(data_set))
    n_ts_set = len(data_set) - n_tr_set

    return (data_set[:n_tr_set], class_set[:n_tr_set], data_set[n_ts_set:], class_set[n_ts_set:])


def randomize_data(dataframe, seed):
    """
    Returns a randomized version of the input data based on the seed
    It's possible to save the data in a new file setting the write_to_file flag to True
    And by providing the output file name

    Return panda's dataframe randomized
    """
    np.random.seed(seed)
    print("Randomizing..")
    copy = dataframe.copy()
    np.random.shuffle(copy.values)
    return copy


def slice_img(dataframe):
    """
    Return the dataframe with all the images reduced to 28x28 to eliminate the background
    """
    bar = Bar('Slicing', max=len(dataframe.values))
    data = []
    for image in dataframe.values:
        re = np.reshape(image, (48, 48))
        sub_matrix = re[9:37, 9:37]
        data.append(sub_matrix.flatten())
        bar.next()
    reduced = pd.DataFrame(data, columns=range(0, 28**2))
    bar.finish()
    return reduced


def normalize(dataframe):
    """
    Create the normalized version of the train_smpl
    The image's pixels are converted in a range [0-255]
    """
    normalized = []
    bar = Bar('Normalizing\t', max=len(dataframe.values))
    for pixels in dataframe.values:
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(pixels)
        new_row = []
        for pixel in pixels:
            pixel = int((pixel - minVal) * (255 / (maxVal - minVal)))
            new_row.append(pixel)
        normalized.append(new_row)
        bar.next()
    bar.finish()
    return pd.DataFrame(normalized, columns=dataframe.columns)


def save_dataframe_csv(dataframe, file_path):
    """
    Helper method to save a dataframe in the correct format
    """
    if file_path != "":
        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(dataframe.columns)
            bar = Bar('Saving csv\t', max=len(dataframe.values))
            for el in dataframe.values:
                csv_writer.writerow(el)
                bar.next()
            bar.finish


def save_img(dataframe, indx, type_name):
    """
    Saves an image of a data fram given its index
    """
    size = int(sqrt(dataframe.shape[1]))
    ocv_img = np.reshape(dataframe.values[indx], (size, size))
    cv2.imwrite(constants.IMG_PATH+str(indx)+type_name+".png", ocv_img)
    pass


# =============================================================================
# if __name__ == '__main__':
#     # CREATE FILES
#     # SAVING IMAGES
#     # n = 5741
#     # df_o = load_dataframe(constants.ORIGINAL_X_SMPL)
#     # df_o = randomize_data(df_o, constants.SEED)
#     # save_img(df_o, n, "ORIGINAL")
#     # df_n = load_dataframe(constants.NORMALIZED_FULL_RND_SMPL)
#     # save_img(df_n, n, "NORMALIZED")
#     # df_s = load_dataframe(constants.NORMALIZED_SLICED_RND_SMPL)
#     # save_img(df_s, n, "SLICED")
# =============================================================================
