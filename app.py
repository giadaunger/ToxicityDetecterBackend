from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import praw
import tweepy
import os
import re
import emoji

load_dotenv()
app = Flask(__name__)
CORS(app) 

# Clean text
def clean_text(text):
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = emoji.replace_emoji(text, "")
    return text

# Reddit 
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

def fetch_reddit_thread(url):
    try:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)
        title = clean_text(submission.title)
        author = clean_text(submission.author.name) if submission.author else "Unknown"
        comments = [clean_text(comment.body) for comment in submission.comments.list()]
        return {
            "title": title,
            "author": author,
            "comments": comments
        }
    except Exception as e:
        print(f"Error fetching Reddit thread: {e}")
        return {"error": str(e)}


# Twitter
client = tweepy.Client(bearer_token=os.getenv("BEARER_TOKEN"))

def fetch_twitter_thread(tweet_id):
    try:
        tweet = client.get_tweet(tweet_id, tweet_fields=["conversation_id", "author_id"])
        conversation_id = tweet.data["conversation_id"]
        
        thread_tweets = client.search_recent_tweets(query=f"conversation_id:{conversation_id}", tweet_fields=["author_id", "text"])
        if not thread_tweets.data:
            return {"error": "No thread found for the given tweet"}

        thread = [tweet.text for tweet in thread_tweets.data]

        return {
            "main_tweet": tweet.data["text"],
            "thread": thread
        }
    except Exception as e:
        return {"error": str(e)}


def extract_tweet_id(tweet_url):
    tweet_id = tweet_url.split('/')[-1]
    print(f"Extracted tweet_id: {tweet_id}") 
    return tweet_id


@app.route('/submit', methods=['POST'])
def receive_input():
    data = request.json 
    link = data.get('link')
    comment = data.get('comment')

    if not data:
        return jsonify({'error': 'No data received'}), 400
    
    if "reddit.com" in link:
        reddit_data = fetch_reddit_thread(link)
        return jsonify(reddit_data), 200
    elif "x.com" in link:
        tweet_id = extract_tweet_id(link)
        print(tweet_id)
        twitter_thread = fetch_twitter_thread(tweet_id)
        return jsonify(twitter_thread), 200
    else:
        return jsonify({'error': 'Invalid link or unsupported platform'}), 400

if __name__ == '__main__':
    app.run(debug=True)
