import tensorflow as tf
import re

# --- ניקוי טקסט ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- טעינת המודל ---
print("📦 טוען מודל...")
model = tf.keras.models.load_model("faux_finder_model.keras")

# --- שליפת שכבת ה־Vectorization מתוך המודל
vectorization = model.layers[0]

# --- חיזוי ידני ---
def predict_manual_input():
    print("\n📝 בדיקת מודל על משפטים ידניים:")
    while True:
        text = input("הקלד משפט לבדיקה (או 'exit' ליציאה):\n> ")
        if text.lower() == 'exit':
            print("🚪 יציאה...")
            break

        cleaned_text = clean_text(text)
        print(cleaned_text)
        input_tensor = tf.convert_to_tensor([cleaned_text])
        pred_prob = model.predict(input_tensor)[0][0]

        pred_label = "REAL" if pred_prob >= 0.5 else "FAKE"
        confidence = round(pred_prob * 100, 2) if pred_label == "REAL" else round((1 - pred_prob) * 100, 2)

        print(f"\n🔎 סיווג: {pred_label} ({confidence}% ביטחון)\n")

# --- הרצת הבדיקה ---
predict_manual_input()
