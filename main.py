import tensorflow as tf

# Define the path to the CSV file
csv_file_path = "D:\FakeNews\covid.csv"

dataset = tf.data.experimental.make_csv_dataset(
    csv_file_path,
    batch_size=1,     # Use a batch size of 1 to inspect row by row
    label_name="outcome",  # Replace with the actual label column name
    num_epochs=1,     # Do not repeat the dataset
    shuffle=False     # Do not shuffle for easier inspection
)

# Count the number of rows
num_rows = sum(1 for _ in dataset)

print(f"The file has {num_rows} rows.")

# # Inspect the data
# for features, label in dataset.take(10):  # View the first 5 rows
#     print("Features:", features)
#     print("Label:", label)