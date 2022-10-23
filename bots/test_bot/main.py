import tweepy
import logging
import twitter_keys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Authenticate to Twitter
def create_api():
    auth = tweepy.OAuth1UserHandler(twitter_keys.api_Key, twitter_keys.api_key_secret, twitter_keys.access_token, twitter_keys.access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    
    logger.info("API created successfully!")
    
    return tweepy.API(auth)

# Post a tweet given a message and/or an image
def tweet_message(api, message, image_path=None):
    if image_path:
        api.update_status_with_media(message, image_path)
    else:
        api.update_status(message)

    print('tweeted successfully :D')

# Searches mention from tweets and responds using cursor
def searches_mentions(api, keywords, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        if tweet.in_reply_to_status_id is None:
            continue
        if any(keyword in tweet.text.lower() for keyword in keywords):
            logger.info('Answering to %s', tweet.user.name)

            try:
                original_tweet_id = tweet.in_reply_to_status_id
                original_tweet = api.get_status(original_tweet_id)
                logger.info("tweet: %s", original_tweet.text)

                while original_tweet.in_reply_to_status_id is not None:
                    original_tweet_id = original_tweet.in_reply_to_status_id
                    original_tweet = api.get_status(original_tweet_id)
                    logger.info("tweet: %s", original_tweet.text)
    
                api.update_status(status="Some answer", in_reply_to_status_id=tweet.id,  auto_populate_reply_metadata=True)

            except tweepy.errors.NotFound as e:
                    logging.error("User or tweet deleted".format(e))

            except tweepy.errors.TweepyException as e:  
                logging.error("Tweepy error occured:{}".format(e))
            

    return new_since_id

# Searches mention from tweets using paginator
def searches_mentions_with_paginator(api, keywords, since_id, bot_id):
    client = tweepy.Client(twitter_keys.bearer_token)
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Paginator(client.get_users_mentions, id=bot_id, since_id=since_id):
        print(tweet)
        for info in tweet:
            print(info)
        if tweet.in_reply_to_status_id is None:
            continue
        if any(keyword.lower() in tweet.text.lower() for keyword in keywords):
            logger.info('Answering to %s', tweet.user.name)

            original_tweet_id = tweet.in_reply_to_status_id
            original_tweet = api.get_status(original_tweet_id)
            logger.info("Replying to tweet: %s", original_tweet.text)
    
            api.update_status(status="Some answer", in_reply_to_status_id=tweet.id,  auto_populate_reply_metadata=True)
            break
    return new_since_id

class IDPrinter(tweepy.StreamingClient):

    def on_tweet(self, tweet):
        print(tweet.id)

def main():
    api = create_api()
    bot_id = api.verify_credentials().id
    since_id = 1
    while True:
        since_id = searches_mentions(api, ["help", "support", "calling"], since_id)
        logger.info("Waiting...")
        time.sleep(60)
    #tweet_message(api, 'test')
    #printer = IDPrinter(twitter_keys.bearer_token)
    #printer.sample()

if __name__ == '__main__':
    main()