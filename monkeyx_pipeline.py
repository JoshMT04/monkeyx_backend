
from retrieval.retrieval_func import tag_activity, pull_tweets
from processing.tweet_processing import tweet_processing
from analysis.analysis_func import discourse_temp
from analysis.openai_api_func import beyond_binary_sentiment_analysis, secondary_sentiment_analysis
from twarc import expansions
import pandas as pd

hashtags = ["#BlackLivesMatter"]

tag_activity_dict = {}
for tag in hashtags:

    query = f"{tag} lang:en -is:retweet"
    activity_dict = tag_activity(query)
    tag_activity_dict[tag] = activity_dict
    
most_active_tag = max(tag_activity_dict, key=lambda k: sum(tag_activity_dict[k]['counts']))
print(f"The most active tag is: {most_active_tag}")

search_results = pull_tweets(f"{most_active_tag} lang:en -is:retweet", max_results=10)
first_page = next(search_results)
# The Twitter API v2 returns the Tweet information and the user, media etc. separately
# so we use expansions.flatten to get all the information in a single JSON
result = expansions.flatten(first_page)
processed_tweets = tweet_processing(result)

temp = discourse_temp(processed_tweets, tag_activity_dict[most_active_tag])
print(f"Discourse Temperature of {most_active_tag}: {temp}")

for tweet in processed_tweets:
    emotion = beyond_binary_sentiment_analysis(tweet['text'])
    secondary_emotion = secondary_sentiment_analysis(tweet['text'], emotion)
    print(f"Tweet ID: {tweet['id']}, Emotion: {emotion}, Secondary Emotion: {secondary_emotion}")

    tweet['bbs_emotion'] = emotion
    tweet['secondary_emotion'] = secondary_emotion

# Save processed tweets with emotions
df = pd.DataFrame(processed_tweets)
df.to_csv('tweet_log/140825_blm_processed_with_emotions.csv', index=False)