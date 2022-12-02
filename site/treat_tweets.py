import streamlit as st
from emoji import UNICODE_EMOJI
import re
import nltk
from nltk.corpus import stopwords

@st.cache 
def download_stopwords():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.add('\u200d')
    stopwords_set.add(' ')
    return stopwords_set

regex = re.compile("(http://t\.co.{12})|(https://t\.co.{11})")

def removeMention(tweet):
    words = tweet.split()
    words = [word for word in words if "@" not in word ]
    newTweet = ' '.join(words)
    return newTweet

  # substituir menção por <USER>

# search your emoji
def is_emoji(s, language="en"):
    return s in UNICODE_EMOJI[language]

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char + ' ' if is_emoji(char) else char for char in text).strip()

def separateEmoji(tweet):
    return add_space(tweet)

def removeLink(tweet):
    #words = tweet.split()
    #links = [word for word in words if "t.co" in word]
    words = regex.sub('',tweet)
    return words

def splitPunctuation(tweet):
    tweet = tweet.replace(".", " . ").replace(",", " , ").replace(";", " ; ")\
        .replace("!", " ! ").replace("?", " ? ").replace(":", " : ")\
        .replace("(", " ( ").replace(")", " ) ")
    return tweet

def remove_stopwords(tweet):
    words = tweet.split()
    stopwords_set = download_stopwords()
    words = [word for word in words if not word in stopwords_set]
    newTweet = ' '.join(words)
    return newTweet

def lower_tweet(tweet):
    return tweet.lower()
