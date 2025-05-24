import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.api.layers import TextVectorization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# הורדת משאבים
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ניקוי טקסט
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# שלב 1: טען את המודל
model = load_model("final_train.keras")

# שלב 2: טען את הדאטאסט החדש
df = pd.read_csv("final_eval_dataset.csv")
df["text"] = df["text"].astype(str).apply(clean_text)

# שלב 3: הכנת וקטוריזציה (אם לא נשמרה)
vectorization = TextVectorization(max_tokens=20000, output_sequence_length=150)
vectorization.adapt(df["text"])

# יצירת טנסור של טקסטים ותוויות
text_label_ds = tf.data.Dataset.from_tensor_slices((df["text"], df["label"])).batch(32)

# חישוב loss
loss, _ = model.evaluate(text_label_ds, verbose=0)
print(f"🧮 Loss: {loss:.4f}")

# יצירת טנסור של טקסטים לבד (עבור predict)
text_ds = tf.data.Dataset.from_tensor_slices(df["text"]).batch(32)

# שלב 5: ניבוי
probs = model.predict(text_ds)
df["score"] = probs.flatten()
df["predicted_label"] = (df["score"] >= 0.5).astype(int)

# שלב 6: חישוב ביצועים (אם יש תוויות אמת)
if "label" in df.columns:
    y_true = df["label"].astype(int).values
    y_pred = df["predicted_label"].values

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n📊 תוצאת המודל על דאטהסט חדש:")
    print(f"✅ True Positives: {TP}")
    print(f"❌ False Positives: {FP}")
    print(f"❌ False Negatives: {FN}")
    print(f"✅ True Negatives: {TN}")
    print(f"📌 Prediction real news (סה\"כ): {sum(y_pred)}")

    print("\n📈 מדדים:")
    print(f"🎯 Accuracy:  {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🔁 Recall:    {recall:.4f}")
    print(f"💡 F1 Score:  {f1:.4f}")
else:
    print("⚠️ לא נמצאה עמודת 'label' בדאטהסט — לא ניתן לחשב ביצועים.")
