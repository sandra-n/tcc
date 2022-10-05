import tweepy
import logging
import twitter_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Authenticate to Twitter
def create_api():
    auth = tweepy.OAuthHandler(twitter_keys.api_Key, twitter_keys.api_key_secret)
    auth.set_access_token(twitter_keys.access_token, twitter_keys.access_token_secret)
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

# Searches mention from own tweets and responds
def searches_mentions(api, keywords, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        if tweet.in_reply_to_status_id is not None:
            continue
        if any(keyword in tweet.text.lower() for keyword in keywords):
            logger.info('Answering to %s', tweet.user.name)

            if not tweet.user.following:
                tweet.user.follow()

            api.update_status(
                status="Please reach us via DM",
                in_reply_to_status_id=tweet.id,
            )
    return new_since_id


def main():
    api = create_api()
    tweet_message(api, 'This is a test')


if __name__ == '__main__':
    main()