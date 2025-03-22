import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Simple text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Load dataset
df = pd.read_csv("covid.csv")
df["headlines"] = df["headlines"].astype(str).apply(clean_text)

# Extract texts and labels
texts = df["headlines"].tolist()
labels = df["outcome"].tolist()

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# Save tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Build the CNN model
model = Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile and train the model using class weights
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    np.array(X_train),
    np.array(y_train),
    epochs=10,
    batch_size=32,
    validation_data=(np.array(X_test), np.array(y_test)),
    class_weight=class_weights
)

# Save the model
model.save("model.h5")
print("✅ Model and tokenizer saved.")
