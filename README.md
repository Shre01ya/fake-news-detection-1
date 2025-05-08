# fake-news-detection-1
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

data = [
    ("The government announced a new economic policy that aims to reduce inflation.", "REAL"),
    ("Scientists have confirmed water on Mars through latest rover analysis.", "REAL"),
    ("Aliens landed in New York and shook hands with the President.", "FAKE"),
    ("Drinking bleach cures COVID-19 according to experts.", "FAKE"),
    ("NASA to launch new mission to explore Jupiter's moons.", "REAL"),
    ("Politician caught cloning themselves in secret lab.", "FAKE")
]

texts = [item[0] for item in data]
labels = [item[1] for item in data]

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

texts_cleaned = [clean_text(t) for t in texts]

X_train, X_test, y_train, y_test = train_test_split(texts_cleaned, labels, test_size=0.33, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return f"Prediction: {prediction}"


print(predict_news("President signs new bill into law to support education funding."))
print(predict_news("Scientists say dinosaurs are living secretly on an island."))
