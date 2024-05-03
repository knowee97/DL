import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

df = pd.read_csv(r"/content/Tetuan City power consumption.csv")

df.info()

df.isnull().sum()

df.head(10)

df['DateTime'] = pd.to_datetime(df.DateTime)
df.sort_values(by='DateTime', ascending=True, inplace=True)

df['DateTime']

chronological_order = df['DateTime'].is_monotonic_increasing

chronological_order

time_diffs = df['DateTime'].diff()
equidistant_timestamps = time_diffs.nunique() == 1

equidistant_timestamps

df.set_index('DateTime', inplace=True)
df.sort_values(by="DateTime", inplace=True)

chronological_order

df.head(10)

df.iloc[:, 0:6]

df.describe()

for zone in ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption', ]:
  df[zone].plot(kind = 'line', figsize = (14, 6), title = 'Zone Power Consumption')
  plt.legend(['Zone 1', 'Zone 2', 'Zone 3'])
  plt.gca().spines[['top', 'right']].set_visible(False)

df["Zone 1 Power Consumption"].plot(kind = 'line', figsize = (14, 4), title = 'Zone 1 Power Consumption')
plt.legend(['Zone 1'])

df["Zone 2  Power Consumption"].plot(kind = 'line', figsize = (14, 4), title = 'Zone 2 Power Consumption', color='orange')
plt.legend(['Zone 2'])

df["Zone 3  Power Consumption"].plot(kind = 'line', figsize = (14, 4), title = 'Zone 3 Power Consumption', color='green')
plt.legend(['Zone 3'])

# Resample the data for more meaningful time series analysis (e.g., daily, weekly)
daily_resampled = df.resample('D').mean()

# Plot daily Power Consumption for each zone
plt.figure(figsize=(12, 6))
sb.lineplot(data=daily_resampled[['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']])
plt.xlabel('Date')
plt.ylabel('Average Power Consumption')
plt.title('Average Daily Power Consumption')
plt.legend(labels=['Zone 1', 'Zone 2', 'Zone 3'])
plt.show()

# Initialize StandardScaler for y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df)

X = data_normalized[:, 0:5] # Feature columns
y = data_normalized[:, 5:] # Target Variable Columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Reshape input to be 3D (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(X_test.shape)

from keras import optimizers
epochs = 30
batch = 256
lr = 0.001
adam = optimizers.Adam(lr)

X_train.shape

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# LSTM model
model = Sequential()

model.add(LSTM(50, activation='relu', return_sequences = True, input_shape=(X_train_set.shape[1], X_train_set.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(50, activation='relu'))
model.add(Dense(25))

model.add(Dense(3))

model.compile(loss= 'mse', optimizer = adam, metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=2)

# test_predict = model.predict(X_test_set)
test_predict = model.predict(X_test)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Inverse scaling for predicted values
test_predict_O = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[1]), test_predict), axis=1))[:, -3:]

# Inverse scaling for actual values
y_test_O = scaler.inverse_transform(np.concatenate((X_test.reshape(X_test.shape[0], X_test.shape[1]), y_test), axis=1))[:, -3:]


# Plot predictions vs actuals for Zone 1 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(y_test_O[:, 0], label='Actual')
plt.plot(test_predict_O[:, 0], label='Predicted')
plt.title('Zone 1 Power Consumption Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

# Plot predictions vs actuals for Zone 2 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(y_test_O[:, 1], label='Actual')
plt.plot(test_predict_O[:, 1], label='Predicted')
plt.title('Zone 2 Power Consumption Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

# Plot predictions vs actuals for Zone 3 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(y_test_O[:, 2], label='Actual')
plt.plot(test_predict_O[:, 2], label='Predicted')
plt.title('Zone 3 Power Consumption Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test, test_predict)
mae = mean_absolute_error(y_test, test_predict)

mse

mae

plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend(loc='best')
plt.title('LSTM')
plt.xlabel('Epochs')
plt.ylabel('MSE')

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='best')
plt.title('LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

from sklearn.metrics import r2_score
r2 = r2_score(y_test, test_predict)

from sklearn.metrics import accuracy_score
# Plot Accuracy
y_test = y_test.argmax(axis=1)
test_predict = test_predict.argmax(axis=1)

acc = accuracy_score(y_test, test_predict)

acc

 r2
