import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from keras import Model
from keras.src.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Embedding, Dense
from tensorflow.keras.layers import Input
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Dropout



df = pd.read_csv("balanced_dataset.csv")

x = df["headlines"].values      # עמודת המשפטים
y = df["outcome"].values     # עמודת התוויות (0 או 1)
df_train, df_test, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, stratify=y)
label_counts = pd.Series(Ytest).value_counts()



# create tf datasets
train_ds = tf.data.Dataset.from_tensor_slices((df_train, Ytrain))
test_ds = tf.data.Dataset.from_tensor_slices((df_test, Ytest))

# convert sentences to sequences
MAX_VOCAB_SIZE = 20_000
vectorization = TextVectorization(max_tokens = MAX_VOCAB_SIZE)
vectorization.adapt(train_ds.map(lambda x, y: x))


# Shuffle and batch the dataset
train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

V = len(vectorization.get_vocabulary())

input_sequences_train = vectorization(df_train)
input_sequences_test = vectorization(df_test)

# print(input_sequences_test.shape)
T = input_sequences_train.shape[1]

vectorization2 = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_sequence_length=T,
    vocabulary=vectorization.get_vocabulary(),
)

# Create the model

D = 20


i = Input(shape=(), dtype=tf.string)
x = vectorization2(i)
x = Embedding(V, D)(x)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)  # 🔸 הוספנו פה את Dropout
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
# x = Dropout(0.3)(x)  # 🔸 אפשר גם אחרי ה־Pooling הסופי
x = Dense(1, activation='sigmoid')(x)


model = Model(i, x)


# Compile and fit
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']

)

r = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
)

# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()
#
# plt.plot(r.history['accuracy'], label='acc')
# plt.plot(r.history['val_accuracy'], label='val_acc')
# plt.legend()
# plt.show()

# print(f1_score(Ytrain, model.predict(df_train) >0.5))
# print(f1_score(Ytest, model.predict(df_test) >0.5))

final_val_loss = r.history['val_loss'][-1]
print(f"Final validation loss: {final_val_loss:.4f}")

# שלב 1: הוצאת טקסטים ותוויות מתוך test_ds
texts = []
labels = []

for text, label in test_ds.unbatch():
    texts.append(text.numpy().decode('utf-8'))
    labels.append(int(label.numpy()))

# שלב 2: תחזיות
texts_tensor = tf.convert_to_tensor(texts)
pred_probs = model.predict(texts_tensor)
pred_labels = (pred_probs.flatten() >= 0.5).astype(int)

# שלב 3: Confusion Matrix
cm = confusion_matrix(labels, pred_labels)
TN, FP, FN, TP = cm.ravel()

# שלב 4: מדדים נוספים
accuracy = accuracy_score(labels, pred_labels)
precision = precision_score(labels, pred_labels)
recall = recall_score(labels, pred_labels)
f1 = f1_score(labels, pred_labels)

# הדפסה
print("\n📊 תוצאות המודל:")
print(f"✅ True Positives: {TP}")
print(f"❌ False Positives: {FP}")
print(f"❌ False Negatives: {FN}")
print(f"✅ True Negatives: {TN}")
print(f"📌 Prediction real news (סך הכל): {sum(pred_labels)}")

print("\n📈 מדדים:")
print(f"🎯 Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔁 Recall:    {recall:.4f}")
print(f"💡 F1 Score:  {f1:.4f}")

