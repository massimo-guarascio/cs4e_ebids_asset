#import libraries
from __future__ import print_function

import warnings
from configparser import ConfigParser
warnings.filterwarnings("ignore")

import os
import random as rn

os.environ['KERAS_BACKEND'] = "tensorflow"


from ebids_model import *

#SEED
seed = 56


def ebids_learning(params):

    seed = 3223
    model_folder_path = params["model_folder_path"]
    preprocesser_folder_path = params["preprocesser_folder_path"]

    data_path = params ["data_path"]
    data_names = params["data_names"]

    model_names = params["model_names"]
    dataset_names = params["dataset_names"]

    file_params = 'ebids_files/default.ini'
    delim = ","
    decimal = "."
    batch_size = 512

    # seed initialization
    os.environ['PYTHONHASHSEED'] = '0'
    # numpy seed
    np.random.seed(seed)
    # rn seed
    rn.seed(seed)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #K.set_session(sess)
    # tf seed
    tf.compat.v1.set_random_seed(seed)

    # suppose training-set (after splitting train and test) is composed of 1000 tuples, the new training set for training DNn will be training_growing_dnn=1000*(1-training_perc)
    # and for training dnn you use  training_grow_dnn * (1-growing_perc) and for growing dnn, you use growing_perc * training_growing_dnn

    # read params from file
    config = ConfigParser()
    config.read(file_params)

    # MANAGEMENT PARAMETERS read from argv[2] or 'default.in
    verbose_fit = config.getint('management', 'verbose_fit')
    verbose_model_check = config.getint('management', 'verbose_model_check')
    debug = config.getboolean('management', 'debug')
    load_preprocesser_from_file = False

    load_from_file = config.getboolean('management', 'load_from_file')
    load_datasets_from_file = config.getboolean('management', 'load_datasets_from_file')
    if debug:
        print("verbose fit", verbose_fit, " verbose model check", verbose_model_check, " debug ", debug)
        print("load_from_preprocesser", load_preprocesser_from_file, " load_from_file", load_from_file,
              " load_dataset_from_file ", load_datasets_from_file)
    # parameters running or not algorithms
    running_ensemble = config.getboolean('running', 'running_ensemble')
    running_competitor = config.getboolean('running', 'running_competitor')
    evaluate_base_models = config.getboolean('running', 'evaluate_base_models')
    if debug:
        print("running_ensemble", running_ensemble, " running_competitor", running_competitor, " evaluate_base_models",
              evaluate_base_models)
    train_perc = config.getfloat('running', 'train_perc')
    test_perc = config.getfloat('running', 'test_perc')
    growing_perc = config.getfloat('running', 'growing_perc')
    validation_perc = config.getfloat('running', 'validation_perc')
    if debug:
        print("Trainining ", train_perc, " Testing ", test_perc, " Growing ", growing_perc, " Validation ",
              validation_perc);

    # ENSEMBLE PARAMETERs
    ensemble_types = [ENSEMBLE_MOE]

    do_freeze_choose = [False]  # do_freeze_choose = [True, False] enable/disable weight freezing
    ens_factors = 64  # number for neurons for the ensemble layers

    # DNN ARCHITECTURE PARAMETERs

    factors = 32  # number for neurons for the base model layers

    # LOSSES PARAMETERs
    # available values: COST_SENSITIVE_LOSS, FOCAL_LOSS
    loss_type = config.get('architecture', 'loss_type')
    add_competitor_to_ensemble = config.getboolean('architecture', 'add_competitor')
    num_epoch = config.getint('architecture', 'num_epochs')
    dropout_pcg = config.getfloat('architecture', 'dropout_perc')

    if debug:
        print("Loss= ", loss_type)
    gamma_loss_parameter = 2.
    alpha_loss_parameter = 0.5
    fn_weight = 4.  # false negative weight
    fp_weight = 4.  # false positive weight

    if debug:
        print("loss type ", loss_type, " add competitor ", add_competitor_to_ensemble, " num epochs ", num_epoch,
              " dropout ", dropout_pcg)

    dataset_paths = [os.path.join(data_path, x) for x in data_names]
    if debug:
        print("datasets ", dataset_paths)

    # cs4e 2021 categorical encoder and useless attributes
    enc_suffix = "cs4e_2021"
    #Inserire nell'oggetto parametri
    to_remove_list_parameter = ["Destination_Port", "Flow_Bytes_s", "Flow_Packets_s"]
    categorical_feature_list_parameter = ["port_type"]

    # Variables
    base_model_list = []
    x_ensemble = []
    y_ensemble = []

    # base learner parameters
    base_learner_parameters = {}
    base_learner_parameters["loss_type"] = loss_type
    base_learner_parameters["factors"] = factors
    base_learner_parameters["dropout_pcg"] = dropout_pcg
    base_learner_parameters["gamma"] = gamma_loss_parameter
    base_learner_parameters["alpha"] = alpha_loss_parameter
    base_learner_parameters["fn_weight"] = fn_weight
    base_learner_parameters["fp_weight"] = fp_weight
    base_learner_parameters["embedding_size"] = 96
    base_learner_parameters["depth"] = 3

    # dataset dictionary creation
    cat_pre = create_dictionary(dataset_paths, delim, decimal, True, preprocesser_folder_path,
                                enc_suffix, to_remove_list_parameter, categorical_feature_list_parameter, debug)
    column_dict = cat_pre[0]
    x_ohe = cat_pre[1]
    scaler_x = cat_pre[2]

    if debug:
        print("LOAD/TRAIN BASE MODEL")

    for i in range(0, len(dataset_paths)):

        # EXTRACT DATASET
        preprocessed_data = apply_data_preprocesser(dataset_paths[i], delim, decimal, column_dict, x_ohe,
                                                         to_remove_list_parameter,
                                                         categorical_feature_list_parameter, seed, debug, scaler_x)

        # TEMPORARY VARs
        preprocessed_training = preprocessed_data[0]
        y_train = preprocessed_data[1]

        # MODELS
        if i == 0:
            old_model_name = None
        else:
            old_model_name = model_names[i - 1]
        new_model_name = model_names[i]

        # split the training set into 2 datasets also stratifying uniformly for each class    (x_growing would be used to build the base model and X_validation for the ensemble
        X_train_base, X_train_ensemble, y_train_base, y_train_ensemble = train_test_split(preprocessed_training,
                                                                                          y_train, stratify=y_train,
                                                                                          test_size=validation_perc,
                                                                                          random_state=seed)
        if debug:
            # print("Num dimensions:", len(X_growing[0]))
            print("Num dimensions:", X_train_base.shape[1])
        X_growing, X_validation, y_growing, y_validation = train_test_split(X_train_base, y_train_base,
                                                                            stratify=y_train_base,
                                                                            test_size=growing_perc, random_state=seed)
        # CREATING MODELs
        result = build_base_model(X_growing, X_validation, y_growing, y_validation, batch_size, num_epoch, verbose_fit,
                                  verbose_model_check, old_model_name, new_model_name, load_from_file,
                                  base_learner_parameters, model_folder_path, to_remove_list_parameter,
                                  categorical_feature_list_parameter, seed, loss_type, debug)

        base_model = result[0]

        x_ensemble.append(X_train_ensemble)
        y_ensemble.append(y_train_ensemble)

        if debug:
            print(dataset_names[i], " set for the training of the ensemble")
            print(y_train_ensemble.value_counts())

        total_time = result[2]

        # STORING FOR TESTING ENSEMBLE
        base_model_list.append(base_model)

    # create dataset for the validation of the ensemble and of the competitor
    ensemble_data_x = x_ensemble[0].copy()
    ensemble_data_y = y_ensemble[0].copy()
    for i in range(1, len(dataset_paths)):
        ensemble_data_x = np.vstack((ensemble_data_x, x_ensemble[i]))
        ensemble_data_y = np.concatenate((ensemble_data_y, y_ensemble[i]), axis=0)
        if debug:
            print("size ensemble data", len(ensemble_data_x))
            print("size ensemble data y", len(ensemble_data_y))

    if running_competitor:
        # Competitor Model

        # base learner parameters
        competitor_parameters = {}
        competitor_parameters["loss_type"] = loss_type
        competitor_parameters["factors"] = factors
        competitor_parameters["dropout_pcg"] = dropout_pcg
        competitor_parameters["gamma"] = gamma_loss_parameter
        competitor_parameters["alpha"] = alpha_loss_parameter
        competitor_parameters["fn_weight"] = fn_weight
        competitor_parameters["fp_weight"] = fp_weight
        competitor_parameters["embedding_size"] = 48
        competitor_parameters["depth"] = 3

        # DNN competitor

        # callback list
        competitor_model_path = ''.join([model_folder_path, "competitor_", loss_type, ".hdf5"])
        # save_weights_only = False,
        checkpoint = ModelCheckpoint(competitor_model_path, monitor='macro_f1', verbose=verbose_model_check,
                                     save_best_only=True,
                                     save_weights_only=True, mode='max')
        opt = ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr=1e-15, patience=3, factor=0.001, verbose=0)
        callbacks_list = [checkpoint, opt]

        # init competitor
        competitor_init = create_dnn_tf_func(len(ensemble_data_x[0]), competitor_parameters, seed, loss_type)

        # fit
        start = timer()
        competitor_init[0].fit(ensemble_data_x, ensemble_data_y, batch_size=batch_size, epochs=num_epoch,
                               callbacks=callbacks_list,
                               verbose=verbose_fit)
        end = timer();
        total_time = end - start

        # load best competitor weights
        competitor_init[0].load_weights(competitor_model_path)

    # adding competitor to the ensemble
    if add_competitor_to_ensemble:
        base_model_list.append(competitor_init[0])

    if running_ensemble:
        # ENSEMBLE CREATION AND EVALUATION

        for ensemble_type in ensemble_types:
            for do_freeze in do_freeze_choose:

                # creating parameters for ensemble
                parameters = {}
                parameters["freeze_base_models"] = True
                parameters["freeze_base_models_partly"] = False
                parameters["ensemble_type"] = ensemble_type
                parameters["factors"] = ens_factors
                parameters["dropout_pcg"] = dropout_pcg
                parameters["loss_type"] = COST_SENSITIVE_LOSS  # "OTHER"
                parameters["gamma"] = gamma_loss_parameter
                parameters["alpha"] = alpha_loss_parameter
                parameters["fn_weight"] = fn_weight
                parameters["fp_weight"] = fp_weight
                parameters["do_freeze"] = do_freeze

                # create the input model
                # print("Building model input...")

                input_shape = ensemble_data_x.shape[1]
                name_model_input = 'input_ensemble'

                ensemble_input = Input(shape=(input_shape,), name=name_model_input)

                # init ensemble model
                ensemble_model = create_ensemble(base_model_list, parameters, seed)

                # init callback
                # callback list
                if do_freeze:
                    parfreeze = "_Freeze"
                else:
                    parfreeze = "_NOFreeze"
                ensemble_model_path = ''.join(
                    [model_folder_path, parameters["ensemble_type"], "_", parameters["loss_type"], parfreeze, ".hdf5"])

                checkpoint = ModelCheckpoint(ensemble_model_path, monitor='macro_f1', verbose=verbose_model_check,
                                             save_best_only=True,
                                             save_weights_only=True, mode='max')
                opt = ReduceLROnPlateau(monitor='loss', mode='min', min_lr=1e-15, patience=3, factor=0.001, verbose=0)
                # lrm = LearningRateMonitor()
                callbacks_list = [checkpoint, opt]

                # fill input
                ensemble_data_inputs_x = [ensemble_data_x for i in range(0, len(base_model_list))]

                # fit
                start = timer()
                ensemble_model.fit(ensemble_data_inputs_x, ensemble_data_y, batch_size=batch_size, epochs=num_epoch,
                                   callbacks=callbacks_list, verbose=verbose_fit)
                end = timer();
                total_time = end - start
                # load weight
                ensemble_model.load_weights(ensemble_model_path)