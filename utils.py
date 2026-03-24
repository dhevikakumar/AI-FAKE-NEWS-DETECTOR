import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


def predict_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    label = "Fake News" if prediction == 1 else "Real News"
    confidence = max(prob)
    
    # Simple explanation (top keywords)
    feature_names = vectorizer.get_feature_names_out()
    weights = vec.toarray()[0]
    important_words = [feature_names[i] for i in weights.argsort()[-3:]]
    
    return label, confidence, important_words
