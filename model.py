from keras.layers import Input, Dense, Dropout, CuDNNGRU, Embedding
from keras.losses import categorical_crossentropy
from keras.models import Model
import keras.backend as K
import keras
import numpy as np

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def bpr(y_true, y_pred):
  #print(y_true, y_pred)
    y_true = tf.reshape(y_true, [-1])
    y_true = tf.cast(y_true, tf.int64)
    yhat = tf.gather(y_pred, y_true, axis=1)
    yhatT = tf.transpose(yhat)
    diag = tf.diag_part(yhat)
    sig = tf.nn.sigmoid(diag-yhatT)
    return tf.reduce_mean(-tf.log(sig))


class GRU4REC:

    def __init__(self, args, n_items):
        self.optimizer = keras.optimizers.Adam(
            lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=None, amsgrad=False)
        self.loss = self.set_loss(args.loss)
        self.activation = args.activation
        self.batch_size = args.batch_size
        self.emb_size = args.emb_size
        self.hidden_units = args.hidden_units
        self.dropout = args.dropout
        self.n_items = n_items
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(batch_shape=(self.batch_size, 1, self.n_items))
        gru, gru_states = CuDNNGRU(
            self.hidden_units, stateful=True, return_state=True)(inputs)
        drop2 = Dropout(self.dropout)(gru)
        predictions = Dense(self.n_items, activation=self.activation)(drop2)
        model = Model(input=inputs, output=[predictions])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def set_loss(self, loss):
        if loss == 'bpr':
            return bpr

        if loss == 'crossentropy':
            return categorical_crossentropy


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
