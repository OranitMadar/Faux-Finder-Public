import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import Model
from keras.src.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Input
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix

# --- שלב 1: קריאה והכנה ---
df = pd.read_csv("final_balanced_600_dataset.csv")
# df = df.drop_duplicates(subset="text")  # להסיר דופליקייטים

x = df["text"].values
y = df["label"].values
df_train, df_test, Ytrain, Ytest = train_test_split(x, y, test_size=0.33, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((df_train, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((df_test, Ytest))

MAX_VOCAB_SIZE = 20_000
vectorization = TextVectorization(max_tokens=MAX_VOCAB_SIZE, output_sequence_length=100)
vectorization.adapt(train_ds.map(lambda x, y: x))

train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# --- שלב 2: טעינת GloVe ---
print("📥 טוען GloVe...")
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

# --- שלב 4: בניית המודל עם GloVe ---
D = embedding_dim
V = len(vocab)

vectorization2 = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_sequence_length=100,
    vocabulary=vocab,
)

i = Input(shape=(), dtype=tf.string)
x = vectorization2(i)
x = Embedding(
    input_dim=V,
    output_dim=D,
    weights=[embedding_matrix],
    trainable=True  # אפשר לשנות ל־True אם רוצים fine-tuning
)(x)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)

# --- שלב 5: קומפילציה ואימון ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
r = model.fit(train_ds, validation_data=test_ds, epochs=5)

# --- שלב 6: הערכה ---
texts = []
labels = []

for text, label in test_ds.unbatch():
    texts.append(text.numpy().decode('utf-8'))
    labels.append(int(label.numpy()))

texts_tensor = tf.convert_to_tensor(texts)
pred_probs = model.predict(texts_tensor)
pred_labels = (pred_probs.flatten() >= 0.5).astype(int)

cm = confusion_matrix(labels, pred_labels)
TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(labels, pred_labels)
precision = precision_score(labels, pred_labels)
recall = recall_score(labels, pred_labels)
f1 = f1_score(labels, pred_labels)
loss, _ = model.evaluate(test_ds, verbose=0)

print("\n📊 תוצאות המודל:")
print(f"✅ True Positives: {TP}")
print(f"❌ False Positives: {FP}")
print(f"❌ False Negatives: {FN}")
print(f"✅ True Negatives: {TN}")
print(f"📌 Prediction real news (סה\"כ): {sum(pred_labels)}")

print("\n📈 מדדים:")
print(f"🎯 Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔁 Recall:    {recall:.4f}")
print(f"💡 F1 Score:  {f1:.4f}")
print(f"🧮 Loss:       {loss:.4f}")

print(f"Distribution in Ytest:\n{pd.Series(Ytest).value_counts()}")

# plt.plot(r.history['accuracy'], label='acc')
# plt.plot(r.history['val_accuracy'], label='val_acc')
# plt.legend()
# plt.show()
# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

