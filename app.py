from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/submit', methods=['POST'])
def receive_input():
    data = request.json 
    if not data:
        return jsonify({'error': 'No data received'}), 400

    link = data.get('link')
    comment = data.get('comment')

    response = {
        'received_link': link,
        'received_comment': comment
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
