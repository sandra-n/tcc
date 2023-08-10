import tweepy
import logging
import time
import os
from os import getenv
from google.cloud import storage
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, Dropout
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
from emoji import UNICODE_EMOJI
import re
import nltk
from nltk.corpus import stopwords
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

stopwords_set= None
model = None

no_racism_responses = ["Yay! I didn't find any racist messages on the thread",
                       "The HEXA VEEEEEEM, there's no racism on this thread (if you are not Brazillian, I'm sorry for the random message)",
                       "On a galaxy far far away there are people making racist comments. This is not the case for this thread, good job :)",
                       "I couldn't find racism connotation on this thread :D"]

def download_stopwords():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    stopwords_set = set(stopwords.words('english'))
    return stopwords_set

regex = re.compile("(http://t\.co.{12})|(https://t\.co.{11})")

def removeMention(tweet):
    words = tweet.split()
    words = [word for word in words if "@" not in word ]
    newTweet = ' '.join(words)
    return newTweet

# search your emoji
def is_emoji(s, language="en"):
    return s in UNICODE_EMOJI[language]

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char + ' ' if is_emoji(char) else char for char in text).strip()

def separateEmoji(tweet):
    return add_space(tweet)

def removeLink(tweet):
    words = regex.sub('',tweet)
    return words

def splitPunctuation(tweet):
    tweet = tweet.replace(".", " . ").replace(",", " , ").replace(";", " ; ")\
        .replace("!", " ! ").replace("?", " ? ").replace(":", " : ")\
        .replace("(", " ( ").replace(")", " ) ")
    return tweet

def remove_stopwords(tweet):
    global stopwords_set
    words = tweet.split()

    if not stopwords_set:
        stopwords_set = download_stopwords()
    words = [word for word in words if not word in stopwords_set]
    newTweet = ' '.join(words)
    return newTweet

def lower_tweet(tweet):
    return tweet.lower()

def download_model_file(bucket):

    data_file = "cp-0011.ckpt.data-00000-of-00001"
    index_file = "cp-0011.ckpt.index"
    tokenizer_file = "tokenizer.pickle"

    data = bucket.blob(data_file)
    index = bucket.blob(index_file)
    tokenizer = bucket.blob(tokenizer_file)
    
    folder = "/tmp/"
    if not os.path.exists(folder):
      os.makedirs(folder)
    
    data.download_to_filename(folder + "cp-0011.ckpt.data-00000-of-00001")
    index.download_to_filename(folder + "cp-0011.ckpt.index")
    tokenizer.download_to_filename(folder + "tokenizer.pickle")

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

def pre_process_tweet(tweet, bucket):
    max_length = 280
    tweet_treated = lower_tweet(remove_stopwords(removeMention(removeLink(splitPunctuation(separateEmoji(tweet))))))
    file = open("/tmp/tokenizer.pickle",'rb')
    tokenizer = pickle.load(file)
    tweet_tokenized = tokenizer.texts_to_sequences([tweet_treated])
    tweet_padded = pad_sequences(tweet_tokenized, maxlen=max_length, padding='post')
    return tweet_padded

def predict(tweets, bucket):
    global model

    if not model:
        download_model_file(bucket)
        num_words = 5000
        max_length = 280
        
        model = create_model(num_words, max_length)
        model.load_weights("/tmp/cp-0011.ckpt")
        
    logger.info("got tweets: %s", tweets)

    pred_total = []
    if (tweets is not None):
        for message in tweets:
            frase_processada = pre_process_tweet(message, bucket)
            prediction = model(frase_processada)[0][0]
            pred_total.append(float(prediction))
        return pred_total
    else:
        return -1

# Authenticate to Twitter
def create_api():
    api_key = getenv("api_key")
    api_key_secret = getenv("api_key_secret")
    access_token = getenv("access_token")
    access_token_secret = getenv("access_token_secret") 

    auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    
    logger.info("API created successfully!")
    
    return tweepy.API(auth)
    
# given a client and a tweet, this function iterates over the thread to get all tweets form it
def searches_tweets_trough_conversation(client, original_tweet):
    responses_from_conversation = []
    for response in tweepy.Paginator(client.search_recent_tweets,query = 'conversation_id:' + str(original_tweet.conversation_id)):
        for tweet in response.data:
            responses_from_conversation.append(tweet.text)
    return responses_from_conversation

def follows_bot(api, bot, user):
    rel = api.get_friendship(source_screen_name = user, target_screen_name = bot)
    return rel[0].following

def searches_mentions_with_paginator(api, keywords, since_id, bot, bucket):
    client = tweepy.Client(getenv("bearer_token"))
    logger.info("Retrieving mentions")
    for response in tweepy.Paginator(client.get_users_mentions, id=bot.id, since_id=since_id, tweet_fields=["conversation_id", "author_id"]):
        tweets = response.data
        try:
            if tweets:
                for tweet in tweets:
                    since_id = max(since_id, tweet.id)
                    following = follows_bot(api, bot.screen_name, client.get_user(id = tweet.author_id).data.username)
                    if following:
                        if any(keyword.lower() in tweet.text.lower() for keyword in keywords):
                        
                            responses_from_conversations = searches_tweets_trough_conversation(client, tweet)
                            try:
                                responses_from_conversations.append(client.get_tweet(tweet.conversation_id).data.text)
                            except:
                                logger.info("Original tweet was deleted")
                            logger.info("Replying to tweet: %s", tweet.text)
                            
                            prediction = predict(responses_from_conversations, bucket)

                            logger.info(responses_from_conversations)
                            logger.info("Predictions")
                            logger.info(prediction)

                            if any(x < 0.5 for x in prediction):
                                nb_racist = 100*len([x for x in prediction if x < 0.3])//len(prediction)
                                nb_almost_racist = 100*len([x for x in prediction if (x < 0.5 and x >= 0.3)])//len(prediction)
                                reply_message = f"Analyzing all the messages on the thread, {nb_racist}% of them seems racist and {nb_almost_racist}% tend to be racist\n\nMy goal is to make people aware of how they talk. Some expressions can offend or make people uncomfortable, even if not intended. You can agree/disagree with me, it's just my opinion."

                            else:
                                reply_message = random.choice(no_racism_responses)
                        
                            api.update_status(status=reply_message, in_reply_to_status_id=tweet.id,  auto_populate_reply_metadata=True)
                            responses_from_conversations = [] #so that next tweet doesnt have previous tweets to judge whether or not it has racist comments

                    else:
                        api.update_status(status="Follow me if you want me to classify this thread :D", in_reply_to_status_id=tweet.id,  auto_populate_reply_metadata=True)

        except TypeError as e:
                logging.error("Tweets were not found, or some error ocurred while searching for them: {}".format(e))
        
        except tweepy.errors.NotFound as e:
                logging.error("User or tweet deleted: {}".format(e))

        except tweepy.errors.TweepyException as e:  
                logging.error("Tweepy error occured:{}".format(e))

    return since_id

def get_since_id(bucket, filename):
    last_tweet_id = bucket.blob(filename).download_as_text()
    return last_tweet_id

def update_file(new_since_id, bucket, filename):
    blob = bucket.blob(filename)
    blob.upload_from_string(new_since_id)

def main(request, context):
    logging.info("request: %s", request)
    logging.info("context: %s", context)

    api = create_api()
    bot = api.verify_credentials()

    storage_client = storage.Client()
    bucket = storage_client.bucket("tcc-tweet-bot")

    since_id = int(get_since_id(bucket, "last_tweet_id.txt"))
    since_id = searches_mentions_with_paginator(api, ["tell me"], since_id, bot, bucket)
    
    update_file(str(since_id), bucket, "last_tweet_id.txt")
    return str(since_id)