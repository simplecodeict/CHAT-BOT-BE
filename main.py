from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import random
import string

from chatBot import ChatBot

app = Flask(__name__)
CORS(app)

# MongoDB setup
client = MongoClient(os.getenv('MONGO_URI'))
db = client.chat_database
sessions = db.sessions

bot = ChatBot()


def generate_random_string(length=12):
    """Generate a random alphanumeric string of given length."""
    characters = string.ascii_letters + string.digits  # Letters and digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_email = data.get('user_email')
    message = data.get('message')
    session_id = data.get('session_id')

    print(data)

    # Process the user_message with your chatbot
    result = bot.rag_chain.invoke(message)

    # Extract the answer from the result (assuming the answer is always prefixed by "Answer: ")
    answer = result.split("Answer:")[-1].strip()

    if not session_id:
        print(generate_random_string())
        session = {
            'user_email': user_email,
            'user_message': message,
            'user_respond': answer,
            'session_id': generate_random_string()
        }
        sessions.insert_one(session)
    else:
        session = {
            'user_email': user_email,
            'user_message': message,
            'user_respond': answer,
            'session_id': session_id
        }
        sessions.insert_one(session)

    return jsonify({'response': answer, 'session_id': session_id})


@app.route('/api/get_chat/<user_email>', methods=['GET'])
def get_chat(user_email):
    sessions_data = sessions.find({'user_email': user_email})
    chat_history = []

    for session in sessions_data:
        chat_history.append({
            'session_id': session['session_id'],
            'user_message': session['user_message'],
            'user_respond': session['user_respond'],
            'user_email': session['user_email']
        })

    return jsonify({'chat_history': chat_history})


@app.route('/api/get_single_chat/<session_id>', methods=['GET'])
def get_single_chat(session_id):
    sessions_data = sessions.find({'session_id': session_id})
    chat_history = []

    for session in sessions_data:
        chat_history.append({
            'session_id': session['session_id'],
            'user_message': session['user_message'],
            'user_respond': session['user_respond'],
            'user_email': session['user_email']
        })

    return jsonify({'chat_history': chat_history})


if __name__ == '__main__':
    app.run(debug=True)
