from app import app
from model.predict_model import predict_model
from flask import request

new_obj = predict_model()

@app.route("/predict", methods=['POST'])
def predict_sentence_controller():
    return new_obj.predict_sentence(request.form)

@app.route("/predict/all")
def get_predicted_tweet():
    return new_obj.user_getall()

@app.route("/predict/grafik")
def get_groupby_predicted():
    return new_obj.user_groupby_date()
