import pandas as pd
import pickle as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import datetime

# Load dataset
df = pd.read_csv(r'E:\Streamlit\files\heart_disease.csv')

# Optional: Convert all column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Label Encoding for 'sex'
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])

# Split data
X = df.drop('target', axis=1)
Y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save LabelEncoder and Scaler objects (✅ Fixed)
with open('label_encode_sex2.pkl', 'wb') as file:
    pl.dump(label_encoder, file)

with open('scaler_heart_disease2.pkl', 'wb') as file:
    pl.dump(scaler, file)

# Build ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display summary
model.summary()

# TensorBoard logging
log_dir = "logs/heart_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, early_stopping]
)

# Save model
model.save('heartmodel5.h5')
