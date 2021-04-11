from flask import Flask, render_template
from flask import request

# import relevent for ML model
import numpy as np
import string
import pandas as pd
import pickle
from nltk.corpus import stopwords

# loading model and tokens for the model
logi_model = pickle.load(open('finalized_model.sav', 'rb'))
logi_tokens = None
with open('tokens.txt', 'r') as f:
    logi_tokens = f.readline()
logi_tokens = logi_tokens.split()


# defining a class having relevent functions for the model
class ModelUtils:
    def __init__(self, model, tokens, test_str):
        self.model = model
        self.tokens = tokens
        self.review = test_str
        self.vecData = self.cust_vec()
#         print(self.vecData)
        self.pred = self.model.predict(self.vecData)

    def cleanData(self):
        ''' Cleans the input data string for vectorization '''
        #     removing punctuation
        self.review = ''.join([char for char in self.review if char not in string.punctuation])
#         print('after removing punct : ', self.review)
        #     tokenizing
        self.review = self.review.split(' ')
#         print('after tokenizing : ', self.review)
        #     remove stopwords
        stop = set(stopwords.words("english"))
        self.review = " ".join([word.lower() for word in self.review if word.lower() not in stop and word.lower() != 'br'])
#         print('after removing stopwords : ', self.review)

    def cust_vec(self):
        ''' Vectorizes the input data using the tokens we have '''
#         clean the review
        self.cleanData()
#         print(self.review)
#         vectorise the data
        vec = dict()
#         print(len(self.tokens))
        for t in self.tokens:
            vec[t] = [0]
        for word in self.review.split():
            try:
                vec[word][0] += 1
            except KeyError:
                continue
        return pd.DataFrame(data=vec)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    mood = ""
    if request.method == "POST":
        response = request.form["user_response"]
        # find the mood from the response got from the user
        m = ModelUtils(logi_model, logi_tokens, response)
        # use the keras model for sentiment analysis and then run it here
        mood = "Someone is jumping with joy today!" if m.pred[0] else "You seem to be in a bad mood!!"
    return render_template('index.html', mood=mood, resp=m.pred[0])


if __name__ == '__main__':
    app.run(debug=True)
