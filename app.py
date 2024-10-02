from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import praw
import tweepy
import os

load_dotenv()
app = Flask(__name__)
CORS(app) 

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

def fetch_reddit_thread(url):
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)
    comments = [comment.body for comment in submission.comments.list()]
    return {
        "title": submission.title,
        "author": submission.author.name,
        "comments": comments
    }

client = tweepy.Client(bearer_token=os.getenv("TWITTER_TOKEN"))

@app.route('/submit', methods=['POST'])
def receive_input():
    data = request.json 
    link = data.get('link')
    comment = data.get('comment')

    if not data:
        return jsonify({'error': 'No data received'}), 400
    
    try:
        reddit_data = fetch_reddit_thread(link)
        return jsonify(reddit_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
