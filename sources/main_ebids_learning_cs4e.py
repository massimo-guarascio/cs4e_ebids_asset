
from ebids_learning import *

import os

if __name__ == "__main__":

    #--------- From JSON FILE Params ---------------
    dataset_path = "ebids_files/dataset"
    model_folder_path = "ebids_files/out/model/"
    preprocesser_folder_path = "ebids_files/out/preprocesser/"
    file_params = "ebids_files/default.ini"
    delim = ","
    decimal = "."
    batch_size = 512
    # -----------------------------------------------

    directories = os.listdir(dataset_path)
    data_names = []
    model_names = []
    dataset_names = []
    dataset_number = 0
    for file in directories:
        data_names.append(str(file))
        model_string = "model_"+ str(file)[0:len(str(file))-4]
        model_names.append(model_string)
        dataset_string = "dataset_"+str(file)[0:len(str(file))-4]
        dataset_names.append(dataset_string)
        dataset_number = dataset_number+1
        print(file)

    print(data_names)
    print(model_names)
    print(dataset_names)
    print(dataset_number)

    params = {"dataset_number" : dataset_number,
              "model_folder_path" : model_folder_path,
              "preprocesser_folder_path" : preprocesser_folder_path,
              "data_path" : "ebids_files/dataset",
              "data_names" : data_names,
              "model_names" : model_names,
              "dataset_names" : dataset_names,
              "file_params" : file_params,
              "delim": delim,
              "decimal": decimal,
              "batch_size": batch_size,
              }

    #Run Learning Method
    print('main')
    ebids_learning(params)
