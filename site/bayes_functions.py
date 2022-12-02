import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from treat_tweets import remove_stopwords, lower_tweet, splitPunctuation, removeLink, separateEmoji, add_space, is_emoji, removeMention

def load_bayes_model():
    nb = load('./model/bayes/bayes_step2_nb.joblib')
    return nb

def bayes_predict(tweet, nb):
    vectorizer = load('./model/bayes/bayes_step1_vectorizer.joblib')
    tweet_treated = lower_tweet(remove_stopwords(removeMention(removeLink(splitPunctuation(separateEmoji(tweet))))))
    tweet_vect = vectorizer.transform(pd.Series(tweet_treated))
    pred = nb.predict(tweet_vect)
    return pred