import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from app import model, dataset, app
from flask import request
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///coba-bene.db'
db = SQLAlchemy(app)


class predict_model():
    def __init__(self, data = dataset, my_model = model):
        try:
            self.my_model = my_model
            data = data.drop(['date'], axis=1)
            data = data[data['label'] < 3]
            data['label'] = data['label'].replace([0.0, 1.0, 2.0], ['Not Related', 'Kebakaran', 'Pencegahan'])
            label = pd.get_dummies(data.label)
            data_baru = pd.concat([data, label], axis=1)
            data_baru = data_baru.drop(columns='label')
            tweet = data_baru['tweet'].values
            label = data_baru[['Kebakaran', 'Not Related', 'Pencegahan']].values
            X_train, X_test, y_train, y_test = train_test_split(tweet, label, test_size=0.2, random_state=123)
            max_word = 6000
            self.tokenizer = Tokenizer(num_words=max_word, oov_token='x')
            self.tokenizer.fit_on_texts(X_train)
            self.tokenizer.fit_on_texts(X_test)
            print('preprocess data success, ready to predict')
            with app.app_context():
                try:
                    db.create_all()
                    print('connected to database')

                except:
                    print('connection failed')
        except:
            print('failed to preprocess data')
    
    class Tweet(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        user_name = db.Column(db.String(50), nullable=False)
        user_img = db.Column(db.String(200), nullable=False)
        tweet = db.Column(db.String(250), nullable=False)
        url = db.Column(db.String(200), nullable=False)
        created_at = db.Column(db.String(100), nullable=False)
        predict = db.Column(db.String(100), nullable=False)
    
    def predict_sentence(self, data):
        
        with app.app_context():
            try:
                maxlen = 40
                teks_sequence = self.tokenizer.texts_to_sequences([data['tweet']])
                teks_padded = pad_sequences(teks_sequence, maxlen=maxlen, padding='post', truncating='post', value=0) 
                kategori = self.my_model.predict(teks_padded)
                label_encoder = np.argmax(kategori, axis=-1)
                label_encoder = np.vectorize({0: 'tidak kebakaran', 1: 'kebakaran', 2: 'penanganan'}.get)(label_encoder)
                data = request.form
                new_user = self.Tweet(user_name=data['user_name'], user_img=data['user_img'], tweet=data['tweet'], url=data['url'], created_at=data['created_at'], predict=label_encoder[0])
                db.session.add(new_user)
                db.session.commit()
                return {"data": [{"tweet": data['tweet']},{"predict":label_encoder[0]}]}
            except:
                return {'message': 'failed to add and predict tweet'}
        
    def user_getall(self):
        with app.app_context():
            try:
                users = self.Tweet.query.all()
                results = []
                for user in users:
                    user_data = {}
                    user_data['id'] = user.id
                    user_data['user_name'] = user.user_name
                    user_data['user_img'] = user.user_img
                    user_data['tweet'] = user.tweet
                    user_data['url'] = user.url
                    user_data['created_at'] = user.created_at
                    user_data['predict'] = user.predict
                    results.append(user_data)
                return {'data': results}
            except:
                return {'message': 'failed to get users'} 
            
    def user_groupby_date(self):
        with app.app_context():
            try:
                tweets = self.Tweet.query.all()
                results = []
                for tweet in tweets:
                    result_index = None
                    for i, result in enumerate(results):
                        if result['created_at'] == tweet.created_at:
                            result_index = i
                            break

                    if result_index is None:
                        # create new result object
                        result = {
                            'count': 1,
                            'created_at': tweet.created_at,
                            'predicted': {
                                'kebakaran': 0,
                                'tidak kebakaran': 0,
                                'penanganan': 0
                            }
                        }
                        results.append(result)
                    else:
                        # use existing result object
                        result = results[result_index]
                        result['count'] += 1

                    # increment count for predicted label
                    if tweet.predict in result['predicted']:
                        result['predicted'][tweet.predict] += 1

                response_data = [{'count': result['count'], 'created_at': result['created_at'], 'predicted': result['predicted']} for result in results]
                return {'data': response_data}
            except:
                return {'message': 'failed to add user'}
    
        # try:
            
        #     if(data['sentence'].strip() == '' or data['sentence'].strip() == None):
        #         return {'message':'sentence cannot be empty'}
        #     maxlen = 40
        #     teks_sequence = self.tokenizer.texts_to_sequences([data['sentence']])
        #     teks_padded = pad_sequences(teks_sequence, maxlen=maxlen, padding='post', truncating='post', value=0) 
        #     kategori = self.my_model.predict(teks_padded)
        #     label_encoder = np.argmax(kategori, axis=-1)
        #     label_encoder = np.vectorize({0: 'Tidak kebakaran', 1: 'Kebakaran', 2: 'Penanganan'}.get)(label_encoder)
            
        #     return {'message':'success','data': [{'sentence': data['sentence']}, {'prediction': label_encoder[0]}]}
        # except:
        #     return {'message': 'failed to predict sentence'}
        