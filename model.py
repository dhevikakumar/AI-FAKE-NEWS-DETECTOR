import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset (you can replace with real dataset)
data = {
    "text": [
        "Government launches new education policy",
        "Aliens found living on the moon",
        "Stock market reaches all time high",
        "Miracle cure for cancer discovered overnight"
    ],
    "label": [0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)
