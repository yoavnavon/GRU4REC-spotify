from keras.layers import Input, Dense, Dropout, CuDNNGRU, GRU, Embedding
from keras.losses import categorical_crossentropy
from keras.models import Model
import keras.backend as K
import keras
from keras import regularizers
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def bpr(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])  # flatten for gather (b,1) -> (b,)
    y_true = tf.cast(y_true, tf.int32)
    # get positive and negative scores
    gather = tf.gather(y_pred, y_true, axis=1)
    diag = tf.diag_part(gather)  # positive samples
    diag_exp = tf.expand_dims(diag, axis=0)  # expand dim to transpose
    trans = tf.transpose(diag_exp)
    diff = trans - gather  # diference between positive and all
    sig = tf.nn.sigmoid(diff)
    loss = tf.reduce_mean(-tf.log(sig))
    return loss


def softmax_neg(logits, batch_size):
    mask = tf.cast(1, tf.float32) - tf.eye(batch_size,
                                           batch_size, dtype=tf.float32)
    neg_scores = mask * logits
    diff = neg_scores - tf.reduce_max(neg_scores, axis=1)
    exp = tf.math.exp(diff) * mask
    softmaxed = exp / tf.reduce_sum(exp, axis=1)
    return softmaxed

# def bpr_max(bpr_reg, batch_size):


def _bpr_max(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_true = tf.cast(y_true, tf.int32)
    # get positive and negative scores, gather=yhat
    gather = tf.gather(y_pred, y_true, axis=1)
    y_softmax = softmax_neg(gather, 64)
    diag = tf.diag_part(gather)  # positive samples
    diag_exp = tf.expand_dims(diag, axis=0)  # expand dim to transpose
    trans = tf.transpose(diag_exp)
    diff = trans - gather
    sig = tf.nn.sigmoid(diff) * y_softmax
    reg = 0.0001 * tf.reduce_sum(((gather**2)*y_softmax), axis=1)
    loss = tf.reduce_mean(-tf.log(sig + 1e-24) + reg)
    return loss
    # return _bpr_max


class GRU4REC:

    def __init__(self, args, n_items):
        self.optimizer = keras.optimizers.Adam(
            lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=None, amsgrad=False)
        self.activation = args.activation
        self.batch_size = args.batch_size
        self.emb_size = args.emb_size
        self.dropout = args.dropout
        self.hidden_units = args.hidden_units
        self.n_items = n_items
        self.loss = self.set_loss(args.loss)
        self.regularizer = regularizers.l2(
            args.regularization) if args.regularization else None
        self.set_input(args.input_form)
        self.model = self.create_model()

    def create_model(self):
        inputs = self.first_layer
        gru, gru_states = CuDNNGRU(self.hidden_units, stateful=True,
                                   return_state=True, kernel_regularizer=self.regularizer)(inputs)
        drop2 = Dropout(self.dropout)(gru)
        predictions = Dense(self.n_items, activation=self.activation,
                            kernel_regularizer=self.regularizer)(drop2)
        model = Model(input=self.input, output=[predictions])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        return model

    def set_loss(self, loss):
        if loss == 'bpr':
            return bpr

        if loss == 'crossentropy':
            return categorical_crossentropy

        if loss == 'bpr-max':
            return _bpr_max  # (0.0001, self.batch_size)

    def set_input(self, input_form):
        if input_form == 'one-hot':
            self.input = Input(batch_shape=(self.batch_size, 1, self.n_items))
            self.first_layer = self.input
        if input_form == 'emb':
            self.input = Input(batch_shape=(self.batch_size, 1))
            self.first_layer = Embedding(
                self.n_items, self.emb_size)(self.input)
        if input_form == 'content':
            self.input = Input(batch_shape=(self.batch_size, 1, 28))
            self.first_layer = self.input


class MostPopular:

    def __init__(self, dataset_train):
        self.train = dataset_train.df
        self.most_popular = None
        self.get_most_popular()
        self.model = self

    def get_most_popular(self):
        grouped = self.train.groupby('item_idx').count()['session_id']
        items = grouped.sort_values(ascending=False)
        self.most_popular = np.zeros((len(items),))
        for i in range(len(items)):
            self.most_popular[i] = items[i]

    def predict(self, input_oh, batch_size=64):
        return np.repeat(self.most_popular[None], batch_size, axis=0)
