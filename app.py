from flask import Flask, request, jsonify
from flask_cors import CORS
import praw
import tweepy
import os

app = Flask(__name__)
CORS(app) 

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

client = tweepy.Client(bearer_token=os.getenv("TWITTER_TOKEN"))

@app.route('/submit', methods=['POST'])
def receive_input():
    data = request.json 
    link = data.get('link')
    comment = data.get('comment')

    if not data:
        return jsonify({'error': 'No data received'}), 400

    response = {
        'received_link': link,
        'received_comment': comment
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
