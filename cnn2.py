import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

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
df = pd.read_csv("covid.csv")  
df["headlines"] = df["headlines"].astype(str).apply(clean_text)  # ניקוי כל השורות בטקסט
texts = df["headlines"].tolist()
labels = df["outcome"].astype(int).tolist()  # אם יש בעיה, השתמש בLabelEncoder על הלייבל

# 4️⃣ Tokenization ויצירת מטריצה של מספרים עם CountVectorizer
vectorizer = CountVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray()  # ממיר את הטקסט למטריצה של מספרים

# 5️⃣ פדינג (אם נדרש)
max_len = 100
X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding="post")

# 6️⃣ חלוקה לסט אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X_padded, labels, test_size=0.2, random_state=42)

# 7️⃣ בניית רשת CNN לטקסט
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=max_len),  # שים לב שהחלק הזה מתאים למטריצה המפודדת
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, 5, activation="relu"),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# 8️⃣ קומפילציה ואימון
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))

# הערכת המודל
loss, acc = model.evaluate(np.array(X_test), np.array(y_test))
print(f"Accuracy: {acc * 100:.2f}%")
