from flask import Flask
import tensorflow as tf
import pandas as pd
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# predict = predict_controller()

print ('loading model')
model = tf.keras.models.load_model('utils/model/bene.h5')
print('model loaded')

print('loading dataset')
dataset = pd.read_csv('utils/dataset/tweets_clean.csv')
print('dataset loaded')

@app.route("/")
def hello():
    return 'cek gais'


@socketio.on('connectd')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('client_event')
def handle_client_event(data):
    print('Received event from client:', data)
    print('cek 3')
    # predict.get_predicted_tweet()
    emit('server_response', 'Hello from the server', broadcast=True)
    
    
from controller import *


