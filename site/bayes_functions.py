import pandas as pd
from joblib import load
from treat_tweets import treating_tweet
import os

path = os.path.dirname(__file__)

def load_bayes_model():
    nb = load(path + '/model/bayes/bayes_step2_nb.joblib')
    return nb

def bayes_predict(tweet_treated, nb):
    vectorizer = load(path + '/model/bayes/bayes_step1_vectorizer.joblib')
    tweet_vect = vectorizer.transform(pd.Series(tweet_treated))
    pred = nb.predict(tweet_vect)
    return pred