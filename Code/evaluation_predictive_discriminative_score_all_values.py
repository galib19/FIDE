real_data = real_data.cpu().numpy()
samples_ddpm = samples_ddpm.cpu().numpy()
real_data.shape, samples_ddpm.shape

from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError

#First implement a simple RNN model for prediction
def RNN_regression(units):
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential()
    model.add(GRU(units=units,
                  name=f'RNN_1'))
    model.add(Dense(units=1,
                    activation='sigmoid',
                    name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model
#Prepare the dataset for the regression model

train_test_split = 0.9
# real_data=real_data
synth_data = samples_ddpm[:len(real_data)]
n_events = len(real_data)

#Split data on train and test
idx = np.arange(n_events)
n_train = int(train_test_split*n_events)
train_idx = idx[:n_train]
test_idx = idx[n_train:]

#Define the X for synthetic and real data
X_real_train = real_data[train_idx, :seq_len-1, :]
X_synth_train = synth_data[train_idx, :seq_len-1, :]

X_real_test = real_data[test_idx, :seq_len-1, :]
y_real_test = real_data[test_idx, -1, :]

#Define the y for synthetic and real datasets
y_real_train = real_data[train_idx, -1, :]
y_synth_train = synth_data[train_idx, -1, :]

print('Synthetic X train: {}'.format(X_synth_train.shape))
print('Real X train: {}'.format(X_real_train.shape))

print('Synthetic y train: {}'.format(y_synth_train.shape))
print('Real y train: {}'.format(y_real_train.shape))

print('Real X test: {}'.format(X_real_test.shape))
print('Real y test: {}'.format(y_real_test.shape))

#Training the model with the real train data
ts_real = RNN_regression(12)
early_stopping = EarlyStopping(monitor='val_loss')

real_train = ts_real.fit(x=X_real_train,
                          y=y_real_train,
                          validation_data=(X_real_test, y_real_test),
                          epochs=200,
                          batch_size=128,
                          callbacks=[early_stopping])

#Training the model with the synthetic data
ts_synth = RNN_regression(12)
synth_train = ts_synth.fit(x=X_synth_train,
                          y=y_synth_train,
                          validation_data=(X_real_test, y_real_test),
                          epochs=200,
                          batch_size=128,
                          callbacks=[early_stopping])

#Summarize the metrics here as a pandas dataframe
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
real_predictions = ts_real.predict(X_real_test)
synth_predictions = ts_synth.predict(X_real_test)

metrics_dict = {'r2': [r2_score(y_real_test, real_predictions),
                       r2_score(y_real_test, synth_predictions)],
                'MAE': [mean_absolute_error(y_real_test, real_predictions),
                        mean_absolute_error(y_real_test, synth_predictions)]
                # ,
                # 'MRLE': [mean_squared_log_error(y_real_test, real_predictions),
                #          mean_squared_log_error(y_real_test, synth_predictions)]
                }

results = pd.DataFrame(metrics_dict, index=['Trained with Real', 'Trained with Synthetic'])

print(f"\n Predictive Score (MAE): {mean_absolute_error(y_real_test, synth_predictions)}\n")


n_series = real_data.shape[0]
idx = np.arange(n_series)
n_train = int(train_test_split*n_series)
train_idx = idx[:n_train]
test_idx = idx[n_train:]
train_data = np.vstack((real_data[train_idx],
                        synth_data[train_idx]))
test_data = np.vstack((real_data[test_idx],
                       synth_data[test_idx]))
n_train, n_test = len(train_idx), len(test_idx)
train_labels = np.concatenate((np.ones(n_train),
                               np.zeros(n_train)))
test_labels = np.concatenate((np.ones(n_test),
                              np.zeros(n_test)))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

ts_classifier = Sequential([LSTM(2, input_shape=(seq_len, n_seq), name='LSTM'),
                            Dense(1, activation='sigmoid', name='OUT')],
                           name='Time_Series_Classifier')
ts_classifier.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[AUC(name='AUC'), 'accuracy'])
ts_classifier.summary()

result = ts_classifier.fit(x=train_data,
                           y=train_labels,
                           validation_data=(test_data, test_labels),
                           epochs=100,
                           batch_size=128,
                           callbacks=[early_stopping],
                           verbose=0)

evaluations= ts_classifier.evaluate(x=test_data, y=test_labels)
print(f"Discriminative Score: {np.abs(0.5-evaluations[2])}")