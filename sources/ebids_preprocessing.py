#import libraries
from __future__ import print_function

import warnings
from ebids_columns import *

warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
os.environ['KERAS_BACKEND'] = "tensorflow"
import math
#from keras import backend as optimizers
from tensorflow.compat.v1.keras import backend as K



import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#from losses import binary_focal_loss
#from ensemble_factory import *

import pickle as pk

from sklearn.metrics import precision_recall_curve

#DNN
from tensorflow.keras.layers import Lambda, Concatenate, Dense, BatchNormalization, Dropout, Add, Multiply, Average, Input
from losses import *
from sklearn.compose import ColumnTransformer

# CREATE EXTENDED INPUT
def create_extended_input(raw_input_layer):
    extended_features = []
    range_values = [2, 4, 8, 16]

    extended_features.append(raw_input_layer)

    one_minus_i = Lambda(lambda x: 1 - K.clip(x, 0, 1))(raw_input_layer)
    extended_features.append(one_minus_i)

    # power
    for v in range_values:
        power_i = Lambda(lambda x: x ** v)(raw_input_layer)
        extended_features.append(power_i)

    # root
    for v in range_values:
        root_i = Lambda(lambda x: K.clip(x, 0, 1) ** (1 / v))(raw_input_layer)
        extended_features.append(root_i)

    # sin and 1-cos
    sin_i = Lambda(lambda x: K.sin(math.pi * K.clip(x, 0, 1)))(raw_input_layer)
    extended_features.append(sin_i)
    one_minus_cos_i = Lambda(lambda x: 1 - K.cos(math.pi * K.clip(x, 0, 1)))(raw_input_layer)
    extended_features.append(one_minus_cos_i)

    # other extensions
    log_i = Lambda(lambda x: K.log(K.clip(x, 0, 1) + 1) / math.log(2))(raw_input_layer)
    extended_features.append(log_i)
    one_minus_inv_log_i = Lambda(lambda x: 1 - K.log(K.clip(-x, 0, 1) + 2) / math.log(2))(raw_input_layer)
    extended_features.append(one_minus_inv_log_i)
    exp_i = Lambda(lambda x: K.exp(x - 1))(raw_input_layer)
    extended_features.append(exp_i)
    one_minus_exp_i = Lambda(lambda x: 1 - K.exp(-x))(raw_input_layer)
    extended_features.append(one_minus_exp_i)

    # improved input
    return Concatenate()(extended_features)



# CREATE DICTIONARY FOR CATEGORICAL ATTRIBUTE ENCODING
def create_dictionary(dataset_paths, delim, decimal, load_from_file, preprocesser_folder_path, suffix,
                      to_remove_list_parameter, categorical_feature_list_parameter, debug):
    if load_from_file:
        # DO SOMETHING
        # STORING LabelEncoderMap
        column_dict = pk.load(open(preprocesser_folder_path + "clm_dict_" + suffix + ".sav", 'rb'))

        # STORING ohe
        x_ohe = pk.load(open(preprocesser_folder_path + "ohe_" + suffix + ".sav", 'rb'))

        scaler_x = pk.load(open(preprocesser_folder_path + "scaler_" + suffix + ".sav", 'rb'))

        if debug:
            print("LOADED")

        return (column_dict, x_ohe, scaler_x)

    data_list = []
    for i in range(0, len(dataset_paths)):

        # --- LOADING DATASET ---
        data = readData(dataset_paths[i], delim, decimal)
        new_columns = get_dataset_columns()
        data = data[new_columns]

        # REMOVING ID
        to_remove = to_remove_list_parameter
        data = data.drop(to_remove, 1)

        # CLEAN STRING

        data_list.append(data)
        if debug:
            print("READ")

    categorical_feature_list = categorical_feature_list_parameter  # ["fc_proto"]

    data_list = pd.concat(data_list)
    data_list = data_list.drop("class", 1)
    # mapping
    column_dict = {}

    # CREATE LABEL ENCODER
    for column_name in categorical_feature_list:
        # ~ column_encoder = sklearn.preprocessing.label.LabelEncoder()
        column_encoder = sklearn.preprocessing.LabelEncoder()
        # column_encoder.fit(preprocessed_training[column_name])
        column_encoder.fit(data_list[column_name])
        column_dict[column_name] = column_encoder

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        data_list[column_name] = current_encoder.transform(data_list[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(data_list.columns.get_loc(v))

    if debug:
        print("Features indexes: ", indexes_to_encode)

    # x_ohe = OneHotEncoder(categorical_features=indexes_to_encode, sparse=False)
    # x_ohe = OneHotEncoder(categorical_features=indexes_to_encode, sparse=False)
    x_ohe = ColumnTransformer([('my_ohe', OneHotEncoder(), indexes_to_encode)], remainder='passthrough')

    x_ohe.fit(data_list)

    print(column_dict)
    print("-----------------------")
    print(data_list)

    # SCALING NUMERICAL FEATURES
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    to_scale = data_list.columns.difference(categorical_feature_list)
    # if learning
    scaler_x.fit(data_list[to_scale])

    # STORING Scaler

    pk.dump(scaler_x, open(preprocesser_folder_path + "scaler_" + suffix + ".sav", 'wb'), protocol=2)

    # STORING LabelEncoderMap
    pk.dump(column_dict, open(preprocesser_folder_path + "clm_dict_" + suffix + ".sav", 'wb'), protocol=2)

    # STORING ohe
    pk.dump(x_ohe, open(preprocesser_folder_path + "ohe_" + suffix + ".sav", 'wb'), protocol=2)

    return (column_dict, x_ohe, scaler_x)

def load_dictionary(preprocesser_folder_path, suffix, debug):

    # STORING LabelEncoderMap
    column_dict = pk.load(open(preprocesser_folder_path + "clm_dict_" + suffix + ".sav", 'rb'))

    # STORING ohe
    x_ohe = pk.load(open(preprocesser_folder_path + "ohe_" + suffix + ".sav", 'rb'))

    scaler_x = pk.load(open(preprocesser_folder_path + "scaler_" + suffix + ".sav", 'rb'))

    if debug:
        print("LOADED")

    return (column_dict, x_ohe, scaler_x)

# READ A SINGLE CHUNCK
def readData(path, delimiter, decimal):
    "READ SINGLE FILE"
    df = pd.read_csv(path, delimiter=delimiter, decimal=decimal, engine="c", header=0)
    return df


#
def extract_preprocessed_data(dataset_path, delim, decimal, train_perc, test_perc, column_dict, x_ohe,
                              to_remove_list_parameter, categorical_feature_list_parameter, seed, debug, scaler_x):
    # --- LOADING DATASET ---
    data = readData(dataset_path, delim, decimal)
    new_columns = get_dataset_columns()
    data = data[new_columns]

    if (debug):
        print("--- LOADED DATASET ---", dataset_path)

    # REMOVING id-like useless features
    to_remove = to_remove_list_parameter
    data = data.drop(to_remove, 1)

    if (debug):
        print("--- USELESS ATTRIBUTES REMOVED ---")

    # PRINT CLASS DISTRIBUTION
    if (debug):
        print(data["class"].value_counts())

    ## --- PREPROCESSING ---

    # STEP 1: SPLIT THE DATASET IN TRAINING SET AND TEST SET
    # STEP 2: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    # create X and y for training set and test set
    X = data.drop("class", 1)

    y = data["class"]

    # STEP 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc, test_size=test_perc,
                                                        random_state=seed)
    if debug:
        print("--- CREATED TRAINING AND TEST SET ---")

    # PRINT DISTRIBUTION
    if (debug):
        print("Training distribution")
        print(y_train.value_counts())
        print("Test distribution")
        print(y_test.value_counts())

    # categorical features conversion via one-hot encoding

    # categorical feature list
    categorical_feature_list = categorical_feature_list_parameter

    preprocessed_training = X_train

    # SCALING NUMERICAL FEATURES
    #scaler_x = MinMaxScaler(feature_range=(0, 1))

    #SAVE Scaler on disk


    to_scale = preprocessed_training.columns.difference(categorical_feature_list)

    preprocessed_training[to_scale] = scaler_x.transform(preprocessed_training[to_scale])

    if (debug):
        print("--- SCALING APPLIED ---")

    # CREATE AND APPLY OHE

    # CREATE LABEL ENCODER

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        preprocessed_training[column_name] = current_encoder.transform(preprocessed_training[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(preprocessed_training.columns.get_loc(v))

    preprocessed_training = x_ohe.transform(preprocessed_training)

    if (debug):
        print("--- CATEGORICAL FEATURE ENCODED ---")

    # preparing test

    # create copy
    preprocessed_test = X_test.copy()
    preprocessed_test[to_scale] = scaler_x.transform(preprocessed_test[to_scale])
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        preprocessed_test[column_name] = current_encoder.transform(preprocessed_test[column_name])
    preprocessed_test = x_ohe.transform(preprocessed_test)

    return (preprocessed_training, y_train, preprocessed_test, y_test)

def apply_data_preprocesser(dataset_path, delim, decimal, column_dict, x_ohe,
                              to_remove_list_parameter, categorical_feature_list_parameter, seed, debug, scaler_x):
    # --- LOADING DATASET ---
    data = readData(dataset_path, delim, decimal)
    new_columns = get_dataset_columns()
    data = data[new_columns]

    if (debug):
        print("--- LOADED DATASET ---", dataset_path)

    # REMOVING id-like useless features
    to_remove = to_remove_list_parameter
    data = data.drop(to_remove, 1)

    if (debug):
        print("--- USELESS ATTRIBUTES REMOVED ---")

    # PRINT CLASS DISTRIBUTION
    if (debug):
        print(data["class"].value_counts())

    # STEP: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    # create X and y for training set and test set
    X_test = data.drop("class", 1)
    y_test = data["class"]

    # categorical features conversion via one-hot encoding

    # categorical feature list
    categorical_feature_list = categorical_feature_list_parameter

    preprocessed_test = X_test

    # SCALING NUMERICAL FEATURES
    #scaler_x = MinMaxScaler(feature_range=(0, 1))

    #SAVE Scaler on disk


    to_scale = preprocessed_test.columns.difference(categorical_feature_list)

    #if learning
    #scaler_x.fit(preprocessed_training[to_scale])
    ##Salvare lo scaling
    #else deply
    ##read scaling from file


    preprocessed_test[to_scale] = scaler_x.transform(preprocessed_test[to_scale])

    if (debug):
        print("--- SCALING APPLIED ---")

    # CREATE AND APPLY OHE

    # CREATE LABEL ENCODER

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        preprocessed_test[column_name] = current_encoder.transform(preprocessed_test[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(preprocessed_test.columns.get_loc(v))

    preprocessed_test = x_ohe.transform(preprocessed_test)

    if (debug):
        print("--- CATEGORICAL FEATURE ENCODED ---")

    # preparing test



    return (preprocessed_test, y_test)


# utility method
def first_extract_preprocessed_data(dataset_path, delim, decimal, train_perc, test_perc, column_dict, x_ohe,
                                    out_dataset, to_remove_list_parameter, categorical_feature_list_parameter, seed, debug):
    # --- LOADING DATASET ---
    data = readData(dataset_path, delim, decimal)
    new_columns = get_dataset_columns()
    data = data[new_columns]

    if (debug):
        print("--- LOADED DATASET ---")

    # PRINT COLUMN NAMES
    if (debug):
        print(data.columns)

    # PRINT HEAD OF THE DATASET
    if (debug):
        print(data.head(3))

    # REMOVING ID
    # to_remove = ["fc_id","fc_tstamp","fc_src_port", "fc_dst_port"]
    to_remove = to_remove_list_parameter  # ["fc_id","fc_tstamp","fc_src_port", "fc_dst_port","fc_src_addr","fc_dst_addr", "lpi_category", "lpi_proto", "crl_group", "crl_name" ]
    data = data.drop(to_remove, 1)

    if (debug):
        print("--- USELESS ATTRIBUTES REMOVED ---")

    # PRINT COLUMN NAMES
    if (debug):
        print("FEATURES AFTER USELESSS ATTRIBUTES REMOVED:")
        print(data.columns)

    # PRINT HEAD OF THE DATASET
    if (debug):
        print(data.head(3))

    # PRINT CLASS DISTRIBUTION
    if (debug):
        print(data["class"].value_counts())

    ## --- PREPROCESSING ---

    # categorical features conversion via one-hot encoding

    categorical_feature_list = categorical_feature_list_parameter

    # SCALING NUMERICAL FEATURES
    scaler_x = MinMaxScaler(feature_range=(0, 1))

    to_scale = data.columns.difference(categorical_feature_list)

    scaler_x.fit(data[to_scale])

    data[to_scale] = scaler_x.transform(data[to_scale])

    if (debug):
        print("--- SCALING APPLIED ---")

    # CREATE AND APPLY OHE

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        data[column_name] = current_encoder.transform(data[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(data.columns.get_loc(v))

    if debug:
        print("Features indexes: ", indexes_to_encode)

    tmp_X = data.drop("class", 1)
    tmp_y = data["class"]
    tmpdata = x_ohe.transform(tmp_X)
    data = pd.DataFrame(tmpdata, columns=["Attr_" + str(int(i)) for i in range(tmpdata.shape[1])])
    data['class'] = tmp_y

    if (debug):
        print("--- CATEGORICAL FEATURE ENCODED ---")

    # preparing test

    # saving data
    data.to_pickle(out_dataset)

    # STEP 1: SPLIT THE DATASET IN TRAINING SET AND TEST SET
    # STEP 2: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    # CLEAN STRING
    # data["class"] = data["class"].apply(convert_string_to_float)

    # create X and y for training set and test set
    X = data.drop("class", 1)

    y = data["class"]

    # STEP 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc, test_size=test_perc,
                                                        random_state=seed)
    if debug:
        print("--- CREATED TRAINING AND TEST SET ---")

    # PRINT DISTRIBUTION
    if (debug):
        print("Training distribution")
        print(y_train.value_counts())
        print("Test distribution")
        print(y_test.value_counts())
        print("Type xtrain", type(X_train), " Type Y ", type(y_train))
    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))

# only first time load and preprocess all data

def eff_extract_preprocessed_data(out_dataset, delim, decimal, train_perc, test_perc, column_dict, x_ohe, debug, seed):
    # loading data
    data = pd.read_pickle(out_dataset)
    # STEP 1: SPLIT THE DATASET IN TRAINING SET AND TEST SET
    # STEP 2: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    # CLEAN STRING
    # data["class"] = data["class"].apply(convert_string_to_float)

    # create X and y for training set and test set
    X = data.drop("class", 1)

    y = data["class"]

    # STEP 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc, test_size=test_perc,
                                                        random_state=seed)
    if debug:
        print("--- CREATED TRAINING AND TEST SET ---")

    # PRINT DISTRIBUTION
    if (debug):
        print("Training distribution")
        print(y_train.value_counts())
        print("Test distribution")
        print(y_test.value_counts())

    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))