import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download("stopwords")

# Load the trained model
model = load_model("model.h5")

# Load the tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Main prediction function
def predict_message(message: str) -> float:
    message = clean_text(message)
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)[0][0]
    return float(prediction) * 100  # Return as percentage
