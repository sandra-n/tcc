from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, Dropout
from keras.models import Sequential
import tensorflow as tf
from treat_tweets import treating_tweet
from joblib import load
import pandas as pd
import numpy as np
import os

path = os.path.dirname(__file__)

def create_model():
    with tf.device("cpu:0"):
        model = Sequential()

        model.add(Input(shape=(200, 1)))
        model.add(Conv1D(100, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy",loss_weights=[1.5, 0.75], metrics=['mse', 'accuracy'])

    return model

def load_cnn_emb_model():
    model = create_model()
    model.load_weights(path + "/model/cnn_embedded/cp-0050.ckpt").expect_partial()
    return model

def tweets_tok(tweet):
    cntvct = load(path + '/model/cnn_embedded/cntvct.joblib')
    svd = load(path + '/model/cnn_embedded/svd.joblib')
    tweet_cntvct = cntvct.transform(pd.Series(tweet))
    tweet_svd = svd.transform(tweet_cntvct)
    return np.array(tweet_svd)

def cnn_emb_pre_process_tweet(tweet_treated):
    tweet_tok = tweets_tok(tweet_treated)
    return tweet_tok

