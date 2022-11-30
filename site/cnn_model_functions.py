import keras.api._v2.keras as keras
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Dropout
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
from treat_tweets import remove_stopwords, lower_tweet, splitPunctuation, removeLink, separateEmoji, removeMention
import streamlit as st

def create_model(num_words, max_length):
    with tf.device("cpu:0"):
        model = Sequential()

        model.add(Embedding(num_words, 100, input_length=max_length))
        model.add(Conv1D(max_length, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy",loss_weights=[1.5, 0.75], metrics=['mse', 'accuracy'])
    return model

def load_cnn_model():
    num_words = 5000
    max_length = 280
    model = create_model(num_words, max_length)
    model.load_weights("model/cnn/cp-0011.ckpt")
    return model

def cnn_pre_process_tweet(tweet):
    max_length = 280
    tweet_treated = lower_tweet(remove_stopwords(removeMention(removeLink(splitPunctuation(separateEmoji(tweet))))))
    file = open("model/cnn/tokenizer.pickle",'rb')
    tokenizer = pickle.load(file)
    tweet_tokenized = tokenizer.texts_to_sequences([tweet_treated])
    tweet_padded = pad_sequences(tweet_tokenized, maxlen=max_length, padding='post')
    return tweet_padded

