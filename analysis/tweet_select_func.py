import random

def tweet_select(tweets):
    '''
    Select a tweet based on a weighted random choice from a list of tweets. 
    Probability of selection will be adjusted according to the tweet engagement score.
    Parameters:
    tweets (list): A list of processed tweet dictionaries.
    '''
    if not tweets:
        return None

    # Calculate total engagement score
    for tweet in tweets:
        like_count = tweet['public_metrics']['like_count']
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        quote_count = tweet['public_metrics']['quote_count']
        engagement_score = like_count + retweet_count**2 + reply_count**2 + quote_count**2
        tweet['engagement_score'] = engagement_score

    total_engagement = sum(tweet.get('engagement_score', 0) for tweet in tweets)

    # If no engagement, return random tweet
    if total_engagement == 0:
        return random.choice(tweets)

    # Select tweet based on weighted random choice
    # Generate a random value between 0 and the total_engagement value
    rand = random.uniform(0, total_engagement)
    random.shuffle(tweets)  # Shuffle tweets to randomize selection order
    # Iterate through the tweets and subtract their engagement score from rand
    # If the value of rand drops below 0, that tweet is selected
    # Tweets with higher engagement scores have a higher chance of being selected, but ones with lower engagement could still be selected
    for tweet in tweets:
        rand -= tweet.get('engagement_score', 0)
        if rand <= 0:
            return tweet

    return tweets[-1]  # Fallback if previous logic fails