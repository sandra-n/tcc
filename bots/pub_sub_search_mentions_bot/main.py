import tweepy
import logging
import time
import os
from os import getenv
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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

def searches_mentions_with_paginator(api, keywords, since_id, bot_id):
    client = tweepy.Client(getenv("bearer_token"))
    logger.info("Retrieving mentions")
    for response in tweepy.Paginator(client.get_users_mentions, id=bot_id, since_id=since_id, tweet_fields=["conversation_id"]):
        tweets = response.data
        try:
            for tweet in tweets:
                since_id = max(since_id, tweet.id)
                if any(keyword.lower() in tweet.text.lower() for keyword in keywords):
                    responses_from_conversations = searches_tweets_trough_conversation(client, tweet)
                    responses_from_conversations.append(client.get_tweet(tweet.conversation_id).data.text)
                    logger.info("Replying to tweet: %s", tweet.text)
                    logger.info(responses_from_conversations)
        
                api.update_status(status="Some answer", in_reply_to_status_id=tweet.id,  auto_populate_reply_metadata=True)
                responses_from_conversations = [] #so that next tweet doesnt have previous tweets to judge whether or not it has racist comments

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
    bot_id = api.verify_credentials().id

    storage_client = storage.Client()
    bucket = storage_client.bucket("tcc-tweet-bot")

    since_id = int(get_since_id(bucket, "last_tweet_id.txt"))
    since_id = searches_mentions_with_paginator(api, ["help", "support", "calling"], since_id, bot_id)
    
    update_file(str(since_id), bucket, "last_tweet_id.txt")
    return str(since_id)