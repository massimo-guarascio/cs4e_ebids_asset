#import libraries
from __future__ import print_function

# WORK ONLY WITH sklearn <= 0.21
import warnings

from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings("ignore")

from timeit import default_timer as timer
import os

os.environ['KERAS_BACKEND'] = "tensorflow"

from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import optimizers
import sys


#from losses import binary_focal_loss
#from ensemble_factory import *

import pickle as pk

from sklearn.metrics import precision_recall_curve

#DNN

from tensorflow.compat.v1.keras.initializers import glorot_normal

from ebids_util import *
from ebids_constants import *


# METHOD FOR TRAINING A BASE MODEL
def build_base_model(X_growing, X_validation, y_growing, y_validation, batch_size, num_epoch, verbose_fit,
                     verbose_model_check, old_model_name, new_model_name, load_from_file, base_learner_parameters,
                     model_folder_path, to_remove_list_parameter, categorical_feature_list_parameter, seed, loss_type,
                     debug):


    model_init = create_dnn_tf_func(X_growing.shape[1], base_learner_parameters, seed, loss_type)

    model = model_init[0]
    model_input = model_init[1]
    model_output = model_init[2]
    model_features = model_init[3]

    if model == "not_init":
        print("Classifier not init, exit")
        exit(-1)

    if (debug):
        print("--- INIT COMPLETED ---")

    # building

    # callback list
    best_model_path = ''.join([model_folder_path, new_model_name, "_", loss_type, ".hdf5"])
    # save_weights_only = False,
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_macro_f1', verbose=verbose_model_check,
                                 save_best_only=True,
                                 save_weights_only=True, mode='max')
    opt = ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr=1e-15, patience=3, factor=0.001, verbose=0)
    # lrm = LearningRateMonitor()
    cb_list = [checkpoint, opt]

    # LOAD OLD MODEL IF EXIST
    if old_model_name is not None:
        # load old model
        old_model_path = ''.join([model_folder_path, old_model_name, "_", loss_type, ".hdf5"])
        # model = load_model(old_model_path, custom_objects={'binary_class_weighted_loss_tf': binary_class_weighted_loss_tf})

        # TRANSFER LEARNING
        # UNCOMMENT TO ENABLE TRANSFER LEARNING
        # model.load_weights(old_model_path)
        if debug:
            print("### OLD MODEL LOADED ###")

    # fit
    if not load_from_file:
        if debug:
            print("### FITTING ###")
        start = timer();
        model.fit(X_growing, y_growing, batch_size=batch_size, epochs=num_epoch, callbacks=cb_list,
                  validation_data=(X_validation, y_validation), verbose=verbose_fit)
        end = timer();
        total_time = end - start
    else:
        total_time = 0
        if (debug):
            print("no training phase: loaded model from file")

    if debug:
        print("base model created")

    # load best model
    model.load_weights(best_model_path)

    return (model, model_features, total_time)


# FACTORY FOR ENSEMBLE MODELs
def create_ensemble(models, parameters, seed):
    if parameters["ensemble_type"] == ENSEMBLE_MAX:
        return ensemble_max(models, parameters)

    if parameters["ensemble_type"] == ENSEMBLE_AVG:
        return ensemble_avg(models, parameters)

    if parameters["ensemble_type"] == ENSEMBLE_STACK:
        return ensemble_stacking(models, parameters, seed)

    if parameters["ensemble_type"] == ENSEMBLE_F_STACK:
        return ensemble_stacking_feature(models, parameters, seed)

    if parameters["ensemble_type"] == ENSEMBLE_F_STACK_V2:
        return ensemble_stacking_feature_V2(models, parameters, seed)

    if parameters["ensemble_type"] == ENSEMBLE_MOE:
        return ensemble_moe(models, parameters, seed)

    print("ERROR: No ensemble specified")
    sys.exit(-1)


# ENSEMBLE STATEGY: MAX SCORE (NO TRAINABLE)
def ensemble_max(models, parameters, ensemble_model_name=None):
    def compute_strongest_pred(x):
        thr = tf.fill(tf.shape(x), 0.5)
        x1 = x - thr
        pos = K.relu(x1)
        neg = K.relu(-x1)
        max_pos_abs = K.max(pos, axis=1)
        max_neg_abs = K.max(neg, axis=1)
        bool_idx = K.greater(max_pos_abs, max_neg_abs)
        float_idx = K.cast(bool_idx, dtype=K.floatx())
        thr1 = tf.fill(tf.shape(max_pos_abs), 0.5)
        mask = float_idx * max_pos_abs - (1 - float_idx) * max_neg_abs + thr1
        return K.reshape(mask, (-1, 1))

    def compute_strongest_pred_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] = 1
        return tuple(shape)

    freeze = parameters["do_freeze"]
    if freeze:
        for m in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(m, parameters["freeze_base_models_partly"])

    ensemble_input = [m.input for m in models]
    base_preds = [m.output for m in models]

    x = Concatenate()(base_preds)

    y = Lambda(compute_strongest_pred, output_shape=compute_strongest_pred_shape)(x)

    ensemble_model = Model(ensemble_input, y, name='ensembleMax')

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        ensemble_model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]),
                               optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            ensemble_model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]),
                                   optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss: using default")
            ensemble_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
            # sys.exit(-1)

    return ensemble_model


# ENSEMBLE STATEGY: AVERAGE SCORE (NO TRAINABLE)
def ensemble_avg(models, parameters):
    freeze = parameters["do_freeze"]
    if freeze:
        for model in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(model, parameters["freeze_base_models_partly"])
    inputs = [model.input for model in models]
    outputs = [model.output for model in models]
    y = Average()(outputs)
    model = Model(inputs, y, name='ensembleAvg')

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-14)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss")
            sys.exit(-1)

    return model


# ENSEMBLE STRATEGY: DEEP STACKING (TRAINABLE)
def ensemble_stacking(models, parameters, seed):
    freeze = parameters["do_freeze"]
    if freeze:
        for m in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(m, parameters["freeze_base_models_partly"])

    inputs = [m.input for m in models]
    outputs = [m.output for m in models]

    # ADD default features
    # outputs.append(models[0].input)

    x = Concatenate()(outputs)
    factors = parameters["factors"]

    err_x = Lambda(lambda v: abs(v - 0.5))(x)

    # concatenate , models[0].input
    x = Concatenate()([x, err_x])

    # path
    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x, name='ensembleStacking')

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss: using default")
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
            # sys.exit(-1)

    return model

# ENSEMBLE STRATEGY: MIXTURE OF EXPERTS(TRAINABLE)
def ensemble_moe(models, parameters, seed):
    freeze = parameters["do_freeze"]
    if freeze:
        for m in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(m, parameters["freeze_base_models_partly"])

    ensemble_input = [m.input for m in models]
    models_outputs = [m.output for m in models]

    # Gating network
    g = Dense(128, activation="tanh", kernel_initializer=glorot_normal(seed))(models[0].input)
    # g = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(inputs)
    # g = BatchNormalization()(g)
    # g = Dropout(0.2)(g)
    g = Dense(len(models_outputs), activation='softmax')(g)

    # Weighted combination
    p = Concatenate()(models_outputs)
    weighted_p = Multiply()([p, g])
    shape_list = models_outputs[0].get_shape().as_list()
    y = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=tuple(shape_list[1:]))(weighted_p)

    model = Model(ensemble_input, y, name='ensemble_moe')

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss: using default")
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
            # sys.exit(-1)

    return model


# ENSEMBLE STRATEGY: DEEP STACKING WITH HIGH LEVEL FEATURES EXTRACTED FROM BASE MODELS (TRAINABLE)
def ensemble_stacking_feature(models, parameters, seed):
    freeze = parameters["do_freeze"]
    if freeze:
        for model in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(model, parameters["freeze_base_models_partly"])

    factors = parameters["factors"]

    # input declaration
    w_ensemble_input = [model.input for model in models]

    pred_list = [model.output for model in models]
    # print("len:",len(pred_list))

    # print("base_preds_tensor:", base_preds_tensor.shape)
    context_list = [model.layers[-4].output for model in models]

    pred_tensor = Concatenate()(pred_list)
    err_tensor = Lambda(lambda v: abs(v - 0.5))(pred_tensor)
    context_tensor = Concatenate()(context_list)

    # ADD default features
    # pred_list.append(models[0].input)

    # merged_context = Concatenate()(pred_list+context_list)

    merged_context = Concatenate()([pred_tensor, err_tensor, context_tensor])

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(merged_context)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(w_ensemble_input, x, name='ensemble_stacking_feature')

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss")
            sys.exit(-1)

    return model


# ENSEMBLE STRATEGY: DEEP STACKING WITH HIGH LEVEL FEATURES EXTRACTED FROM BASE MODELS - variant 2 (TRAINABLE)
def ensemble_stacking_feature_V2(models, parameters, seed):
    freeze = parameters["do_freeze"]
    if freeze:
        for model in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(model, parameters["freeze_base_models_partly"])

    w_ensemble_input = [model.input for model in models]

    pred_list = [model.output for model in models]
    # print("len:",len(pred_list))

    # print("base_preds_tensor:", base_preds_tensor.shape)
    context_list = [model.layers[-4].output for model in models]

    # ADD default features
    # pred_list.append(models[0].input)

    add_context = Add()(context_list)
    pred_list.append(add_context)
    merged_context = Concatenate()(pred_list)

    factors = parameters["factors"]

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(merged_context)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(w_ensemble_input, x, name='ensemble_stacking_feature')

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-14)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss")
            sys.exit(-1)

    return model

# FREEZE THE WEIGHT OF A MODEL
def freeze_model(model, partly=False):
    if not partly:
        model.trainable = False
    for layer in model.layers:
        if not partly or ("features_0" not in layer.name) and ("features_1" not in layer.name):
            layer.trainable = False