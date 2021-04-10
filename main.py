from flask import Flask, render_template
from flask import request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    mood = ""
    if request.method == "POST":
        response = request.form["user_response"]
        # find the mood from the response got from the user
        # use the keras model for sentiment analysis and then run it here
        mood = response
    return render_template('index.html', mood=mood)


if __name__ == '__main__':
    app.run(debug=True)
