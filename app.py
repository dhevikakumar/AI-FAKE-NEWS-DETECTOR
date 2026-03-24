from flask import Flask, render_template, request
from utils import predict_news

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    news = ""
    if request.method == 'POST':
        news = request.form['news']
        label, confidence, words = predict_news(news)
        result = {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'words': words
        }
    return render_template('index.html', result=result, news=news)

if __name__ == '__main__':
    app.run(debug=True)