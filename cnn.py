import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.legacy.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 1️⃣ הורדת משאבי NLTK (פעם אחת)
nltk.download("stopwords")
nltk.download("punkt")

# 2️⃣ פונקציה לניקוי טקסטים
def clean_text(text):
    text = text.lower()  # אותיות קטנות
    text = text.translate(str.maketrans("", "", string.punctuation))  # הסרת סימני פיסוק
    words = word_tokenize(text)  # חלוקה למילים
    words = [word for word in words if word not in stopwords.words("english")]  # הסרת מילים חסרות משמעות
    return " ".join(words)  # חיבור חזרה למשפט נקי

# 3️⃣ טעינת הדאטה
df = pd.read_csv("dataset.csv")  
df["text"] = df["text"].astype(str).apply(clean_text)  # ניקוי כל השורות בטקסט
texts = df["text"].tolist()
labels = df["label"].astype(int).tolist()  

# 4️⃣ Tokenization (כאן ממשיכים כרגיל)
max_words = 10000  
max_len = 100  

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

# 5️⃣ חלוקה לסט אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 6️⃣ בניית רשת CNN לטקסט
model = keras.Sequential([
    layers.Embedding(max_words, 128, input_length=max_len),  
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# 7️⃣ קומפילציה ואימון
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))

# הערכת המודל
loss, acc = model.evaluate(np.array(X_test), np.array(y_test))
print(f"Accuracy: {acc * 100:.2f}%")
