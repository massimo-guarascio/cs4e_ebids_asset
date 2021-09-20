#import libraries
from __future__ import print_function

# WORK ONLY WITH sklearn <= 0.21
import warnings

warnings.filterwarnings("ignore")
import os
import random as rn
from configparser import ConfigParser

os.environ['KERAS_BACKEND'] = "tensorflow"
from ebids_model import *

#SEED
seed = 56

#----------------------- MAIN -------------------------------------


# MAIN

def ebids_application(params, debug):

    seed = 3223


    # if len (sys.argv) == 3:
    file_params = params["file_params"]

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

    # --- PARAMETERS ---

    # other parameters
    delim = ","
    decimal = "."

    # read params from file
    config = ConfigParser()
    config.read(file_params)

    # MANAGEMENT PARAMETERS read from argv[2] or 'default.in
    verbose_fit = config.getint('management', 'verbose_fit')
    verbose_model_check = config.getint('management', 'verbose_model_check')
    #debug = config.getboolean('management', 'debug')
    load_preprocesser_from_file = True
    evaluate_performance = False

    load_from_file = True
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



    # ENSEMBLE PARAMETERs
    ensemble_types = [ENSEMBLE_MOE]

    do_freeze_choose = [False]  # do_freeze_choose = [True, False] enable/disable weight freezing
    ens_factors = 64  # number for neurons for the ensemble layers

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



    # FIXED PARAMETERs

    model_folder_path = params["model_folder_path"]
    preprocesser_folder_path = params["preprocesser_folder_path"]

    #modello
    data_path = params["data_path"]
    #data_names = params["data_names"]
    testset_path = params["testset_path"]

    number_of_models = params["number_of_models"]

    # cs4e 2021 categorical encoder and useless attributes
    enc_suffix = "cs4e_2021"

    to_remove_list_parameter = ["Destination_Port", "Flow_Bytes_s", "Flow_Packets_s"]
    categorical_feature_list_parameter = ["port_type"]

    # Variables
    base_model_list = []

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

    cat_pre = load_dictionary(preprocesser_folder_path, enc_suffix, debug)

    column_dict = cat_pre[0]
    x_ohe = cat_pre[1]
    scaler_x = cat_pre[2]

    preprocessed_test_list = apply_data_preprocesser(testset_path, delim, decimal, column_dict, x_ohe,
                                                         to_remove_list_parameter,
                                                       categorical_feature_list_parameter, seed, debug, scaler_x)
    if debug:
        print(preprocessed_test_list)

    num_column = preprocessed_test_list[0].shape[1]

    if debug:
        print(num_column)

    for i in range(0, number_of_models):

        # CREATING MODELs
        model_init = create_dnn_tf_func(num_column, base_learner_parameters, seed, loss_type)

        # STORING FOR TESTING ENSEMBLE
        base_model_list.append(model_init[0])

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

        # init competitor
        competitor_init = create_dnn_tf_func(num_column, competitor_parameters, seed, loss_type)

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

                input_shape = num_column
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


                # load weight
                ensemble_model.load_weights(ensemble_model_path)

                if debug:
                    print("*** ENSEMBLE EVALUATION ***")

                # evaluate for all the test set
                current_test_x = preprocessed_test_list[0]
                current_test_y = preprocessed_test_list[1]

                input_test_list = [current_test_x for j in range(0, len(base_model_list))]

                ensemble_pred_y_prob = ensemble_model.predict(input_test_list, verbose=0)
                #ensemble_pred_y = np.around(ensemble_pred_y_prob, 0)
                return ensemble_pred_y_prob



