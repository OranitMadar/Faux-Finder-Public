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
model = load_model("CNN_Models/kaggle_dataset_politics.keras")

# שלב 2: טען את הדאטאסט החדש
df = pd.read_csv("Datasets/combined_balanced_sport.csv")
df["text"] = df["text"].astype(str).apply(clean_text)

# שלב 3: הכנת וקטוריזציה
vectorization = TextVectorization(max_tokens=20000, output_sequence_length=150)
vectorization.adapt(df["text"])

# טנסורים
text_label_ds = tf.data.Dataset.from_tensor_slices((df["text"], df["label"])).batch(32)
text_ds = tf.data.Dataset.from_tensor_slices(df["text"]).batch(32)

# חישוב loss
loss, _ = model.evaluate(text_label_ds, verbose=0)
print(f"🧮 Loss: {loss:.4f}")

# שלב 5: ניבוי
probs = model.predict(text_ds)
df["score"] = probs.flatten()
df["predicted_label"] = (df["score"] >= 0.5).astype(int)

# שלב 6: חישוב ביצועים
if "label" in df.columns:
    y_true = 1 - df["label"].astype(int).values
    y_pred = 1 - df["predicted_label"].values

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # מיפוי לפלט תואם אקסל
    FN_excel = TP  # פייק שזוהו נכון
    TP_excel = FP  # אמיתיות שסווגו כפייק
    FP_excel = FN  # פייק שסווגו כאמיתיות
    TN_excel = TN  # אמיתיות שזוהו נכון

    # מדדים
    precision = TP_excel / (TP_excel + FP_excel) if (TP_excel + FP_excel) > 0 else 0
    recall = TP_excel / (TP_excel + FN_excel) if (TP_excel + FN_excel) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP_excel + TN_excel) / (TP_excel + TN_excel + FP_excel + FN_excel)

    # --- פלט תואם אקסל ---
    print("\n📊 תוצאה תואמת לטבלת האקסל שלך:")
    print(f"✅ True Positives (I): {TP_excel}")
    print(f"❌ False Positives (H): {FP_excel}")
    print(f"❌ False Negatives (J): {FN_excel}")
    print(f"✅ True Negatives (K): {TN_excel}")

    print("\n📈 מדדים (עמודות L–O):")
    print(f"📊 Accuracy (L):  {accuracy:.4f}")
    print(f"🎯 Precision (M): {precision:.4f}")
    print(f"🔁 Recall (N):    {recall:.4f}")
    print(f"💡 F1 Score (O):  {f1:.4f}")
else:
    print("⚠️ לא נמצאה עמודת 'label' בדאטהסט — לא ניתן לחשב ביצועים.")