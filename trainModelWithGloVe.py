import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix
from tensorflow.keras.layers import TextVectorization, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Input, Dropout
from keras import Model
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --- הורדת משאבים לניקוי טקסט ---
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- פונקציית ניקוי טקסט ---
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

# --- שלב 1: קריאה והכנה ---
df = pd.read_csv("Datasets/kaggle_dataset_politics.csv")

df["text"] = df["text"].astype(str).apply(clean_text)
df["label"] = df["label"].astype('float32')

x = df["text"].values
y = df["label"].values.astype('float32')
df_train, df_test, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, stratify=y)

train_ds = tf.data.Dataset.from_tensor_slices((df_train, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((df_test, Ytest))

MAX_VOCAB_SIZE = 20_000
vectorization = TextVectorization(max_tokens=MAX_VOCAB_SIZE, output_sequence_length=150)
vectorization.adapt(train_ds.map(lambda x, y: x))

train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# --- שלב 2: טעינת GloVe ---
print("\U0001F4E5 טוען GloVe...")
embedding_index = {}
with open("glove.6B.100d.txt", encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# --- שלב 3: בניית מטריצת embedding ---
embedding_dim = 100
vocab = vectorization.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# --- שלב 4: בניית המודל ---
D = embedding_dim
V = len(vocab)

vectorization2 = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_sequence_length=150,
    vocabulary=vocab,
)

i = Input(shape=(), dtype=tf.string)
x = vectorization2(i)
x = Embedding(V, D, weights=[embedding_matrix], trainable=True)(x)

# בדיוק כמו במאמר: 2 convolution blocks בלבד
x = Conv1D(64, 2, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, 2, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = GlobalMaxPooling1D()(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

r = model.fit(train_ds, validation_data=test_ds, epochs=30, callbacks=[early_stop])

# --- גרף של loss לאורך epochs ---
plt.figure(figsize=(8, 5))
plt.plot(r.history['loss'], label='Train')
plt.plot(r.history['val_loss'], label='Validation')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- שמירת המודל והוקטוריזציה ---
model.save("CNN_Models/kaggle_dataset_politics.keras", save_format="keras")
print("\n✅ המודל והוקטוריזציה נשמרו בהצלחה!")


# --- הערכת ביצועי המודל על סט הבדיקה ---
texts = []
labels = []

for text, label in test_ds.unbatch():
    texts.append(text.numpy().decode('utf-8'))
    labels.append(int(label.numpy()))

texts_tensor = tf.convert_to_tensor(texts)
pred_probs = model.predict(texts_tensor)
pred_labels = (pred_probs.flatten() >= 0.5).astype(int)

# --- היפוך תוויות כדי שה-TP בפלט יהיה עבור פייק (0) ---
labels = 1 - np.array(labels)
pred_labels = 1 - pred_labels

# חישוב מטריצת בלבול
cm = confusion_matrix(labels, pred_labels)
TN, FP, FN, TP = cm.ravel()

# --- חישוב מדדים ---
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
loss, _ = model.evaluate(test_ds, verbose=0)

print("\n📊 תוצאות המודל (בהתייחס לחדשות מזויפות):")
print(f"✅ True Positives (פייק שזוהו נכון): {TP}")
print(f"❌ False Positives (ריל שזוהו כפייק): {FP}")
print(f"❌ False Negatives (פייק שזוהו כריל): {FN}")
print(f"✅ True Negatives (ריל שזוהו נכון): {TN}")
print(f"📌 Prediction fake news (סה\"כ): {sum(pred_labels)}")

print("\n📈 מדדים:")
print(f"🎯 Accuracy (כללי):  {accuracy:.4f}")
print(f"🎯 Precision (פייק): {precision:.4f}")
print(f"🔁 Recall (פייק):    {recall:.4f}")
print(f"💡 F1 Score (פייק):  {f1:.4f}")
print(f"🧮 Loss:              {loss:.4f}")