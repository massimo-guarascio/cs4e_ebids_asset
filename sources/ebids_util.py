#import libraries
from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")

import os
os.environ['KERAS_BACKEND'] = "tensorflow"
#from keras import backend as optimizers
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras import optimizers
import sys

#DNN
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.initializers import glorot_normal
from ebids_preprocessing import *

# CREATE A SINGLE RESIDUAL BLOCK INCLUDING 2 BUILDING BLOCKs
def create_single_building_block(output, input, factors, dropout_pcg, seed):
    # building block
    l = Dense(factors, kernel_initializer=glorot_normal(seed), activation="tanh")(output)
    out = Concatenate()([input, l])
    out = BatchNormalization()(out)
    out = Dropout(dropout_pcg)(out)

    # res
    add = Add()([out, output])

    l = Dense(factors, kernel_initializer=glorot_normal(seed), activation="tanh")(add)
    out = Concatenate()([input, l])
    out = BatchNormalization()(out)
    out = Dropout(dropout_pcg)(out)

    return out


# CREATE A NUMBER OF RESIDUAL BLOCKs
def create_multiple_building_block(out, input, factors, dropout_pcg, depth, seed):
    current_out = create_single_building_block(out, input, factors, dropout_pcg, seed)
    for i in range(1, depth):
        current_out = create_single_building_block(current_out, input, factors, dropout_pcg, seed)
    return current_out


# CREATE THE BASE MODEL ARCHITECTURE
def create_dnn_tf_func(dimensions, base_learner_parameters, seed, loss_type):
    # parameters
    dropout_pcg = base_learner_parameters["dropout_pcg"]
    embedding_size = base_learner_parameters["embedding_size"]

    # latent factors
    factors = base_learner_parameters["factors"]

    # init input
    raw_input_layer = Input(shape=(dimensions,))

    # improved layer
    extended_input_layer = create_extended_input(raw_input_layer)

    # feature embedding
    input_layer = Dense(embedding_size, kernel_initializer=glorot_normal(seed), activation="tanh")(extended_input_layer)

    # depth and width factors
    depth = base_learner_parameters["depth"] - 1

    # l1
    l = Dense(factors, kernel_initializer=glorot_normal(seed), activation="tanh")(input_layer)
    out = Concatenate()([input_layer, l])
    out = BatchNormalization()(out)
    out = Dropout(dropout_pcg)(out)

    # add deep BB
    out = create_multiple_building_block(out, input_layer, factors, dropout_pcg, depth, seed)

    # l-1
    decision_layer = Dense(factors, kernel_initializer=glorot_normal(seed), activation="sigmoid")(out)
    out = BatchNormalization()(decision_layer)

    # lp
    out = Dense(1, kernel_initializer=glorot_normal(seed), activation="sigmoid")(out)

    model = Model(inputs=[raw_input_layer], outputs=out)

    opt = optimizers.RMSprop(lr=0.001, epsilon=1e-14)

    if loss_type == base_learner_parameters["loss_type"]:
        model.compile(
            loss=cost_sensitive_loss(base_learner_parameters["fn_weight"], base_learner_parameters["fp_weight"]),
            optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        # model.compile(loss="mse", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])

    else:
        if loss_type == base_learner_parameters["loss_type"]:
            model.compile(
                loss=binary_focal_loss(gamma=base_learner_parameters["gamma"], alpha=base_learner_parameters["alpha"]),
                optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("error: Unknown loss")
            sys.exit(-1)

    return model, raw_input_layer, out, decision_layer


# COST SENSITIVE LOSS
def cost_sensitive_loss(fn_weight=1., fp_weight=1.):
    def inner_cost_sensitive_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        mask_fn = K.clip(K.round(y_true - y_pred), 0, 1)
        w_fn = mask_fn * fn_weight
        mask_fp = K.clip(K.round(y_pred - y_true), 0, 1)
        w_fp = mask_fp * fp_weight
        mask_other = K.clip(1 - K.round(K.abs(y_true - y_pred)), 0, 1)
        w = w_fn + w_fp + mask_other
        return K.mean(K.square(y_pred - y_true) * w)

    return inner_cost_sensitive_loss


# COMPUTE MACRO RECALL FOR BINARY PROBLEMS
def macro_recall(y_true, y_pred):
    # HARDCODED
    num_classes = 2
    class_id = 0

    def rec(y_true, y_pred):
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        total_per_class = K.cast(K.equal(K.round(y_true), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        class_acc = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(total_per_class), 1),
                                                                         K.floatx())
        return class_acc

    v = 0
    for i in range(num_classes):
        v = v + rec(y_true, y_pred)
        class_id = i

    return v / num_classes


# COMPUTE MACRO PRECISION FOR BINARY PROBLEMS
def macro_precision(y_true, y_pred):
    # HARDCODED
    num_classes = 2
    class_id = 0

    def prec(y_true, y_pred):
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        class_acc = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(accuracy_mask), 1), K.floatx())
        return class_acc

    v = 0
    for i in range(num_classes):
        v = v + prec(y_true, y_pred)
        class_id = i

    return v / num_classes


# COMPUTE MACRO F1-SCORE FOR BINARY PROBLEMS
def macro_f1(y_true, y_pred):
    # HARDCODED
    num_classes = 2
    class_id = 0

    def f1(y_true, y_pred):
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        prec = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(accuracy_mask), 1), K.floatx())
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        total_per_class = K.cast(K.equal(K.round(y_true), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        rec = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(total_per_class), 1), K.floatx())
        return 2 * K.cast(K.sum(rec), K.floatx()) * K.cast(K.sum(prec), K.floatx()) / K.cast(
            K.maximum(K.sum(prec + rec), 1), K.floatx())

    v = 0
    for i in range(num_classes):
        v = v + f1(y_true, y_pred)
        class_id = i

    return v / num_classes



