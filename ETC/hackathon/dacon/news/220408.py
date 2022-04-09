import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
import tensorflow as tf
from tensorflow import keras

root_dir = "D:\\_data\\dacon\\news\\"
nltk.download('stopwords')
nltk.download("punkt")

df_train = pd.read_csv(os.path.join(root_dir, "train.csv"), index_col="id")
df_test = pd.read_csv(os.path.join(root_dir, "test.csv"), index_col="id")
df_submission = pd.read_csv(os.path.join(root_dir, "sample_submission.csv"))

df_train["length"] = df_train.text.map(len)
df_test["length"] = df_test.text.map(len)

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


STOP_WORDS = stopwords.words('english')
print("stop words", STOP_WORDS)

# 불용어 제거
def remove_stop_words(s):
  return " ".join([x for x in word_tokenize(s)])

def preprocess_data(df_train, df_test, 
                    no_number=False,
                    no_stopwords=False,
                    no_punctuation=False,
                    min_len=0,
                    lowercase=False):
  df_train, df_test = df_train.copy(), df_test.copy()

  for df in [df_train, df_test]:
    # 띄어쓰기나 공백이 연속된 경우 공백 하나로 바꿈
    df.text = df.text.str.replace(r"\s+", " ", regex=True)

    if lowercase: # 소문자로 변경
      df.text = df.text.str.lower()
    if no_number: # 숫자 제거
      df.text = df.text.str.replace(r"\d+", "", regex=True)
    if no_punctuation: # punctuation 제거
      df.text = df.text.str.translate(str.maketrans('', '', string.punctuation))
    if no_stopwords: # 불용어 제거
      df.text = df.text.map(remove_stop_words)
    
    df["length"] = df.text.map(len)
    
    # 길이가 min_len 미만인 문자열은 학습 데이터에서 제거한다
    if min_len > 0 and "target" in df.columns:
      df.drop(df[df.length < min_len].index, inplace=True)

  return df_train, df_test

def tokenize_data(df_train, df_test, vocab_size, max_len):
    
  """
    Keras Tokenizer를 이용해 토큰화한 뒤 pad_sequences를 이용해 패딩을 추가함.
  """
  
  tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
  tokenizer.fit_on_texts(df_train.text)

  for df in [df_train, df_test]:
    df["encoded"] = pad_sequences(
         tokenizer.texts_to_sequences(df.text),
         maxlen=max_len,
         padding="post"
         ).tolist()

from keras import layers


def create_simple_model(vocab_size, embedding_size=256, hidden_size=512, dropout_rate=0.25, activation='relu'):
  model = keras.Sequential()
  model.add(layers.Embedding(vocab_size, embedding_size, input_shape=(None,)))
  model.add(layers.GlobalAveragePooling1D())
  model.add(layers.Dense(hidden_size, activation=activation))
  model.add(layers.Dropout(dropout_rate))
  model.add(layers.Dense(20, activation='softmax'))

  return model

def create_lstm_model(vocab_size, embedding_size=256, hidden_size=512, dropout_rate=0.25):
  model = keras.Sequential()
  model.add(layers.Embedding(vocab_size, embedding_size))
  model.add(layers.LSTM(hidden_size))
  model.add(layers.Dropout(dropout_rate))
  model.add(layers.Dense(20, activation="softmax"))
  return model

def create_cnn_model(vocab_size, max_len, embedding_size=64, hidden_size=256, dropout_rate=0.25):
  model = keras.Sequential()
  model.add(layers.Embedding(vocab_size, embedding_size))
  model.add(layers.Reshape((max_len, embedding_size, 1)))

  model.add(layers.Conv2D(3, (3, 3), activation='relu'))
  model.add(layers.Conv2D(8, (3, 3), activation='relu'))
  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=2))
  model.add(layers.Dropout(dropout_rate))

  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.Conv2D(32, (3, 3), activation='relu'))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=2))
  model.add(layers.Dropout(dropout_rate))

  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.Dense(hidden_size, activation='relu'))
  model.add(layers.Dropout(dropout_rate))
  model.add(layers.Dense(20, activation="relu"))
  model.add(layers.Dense(20, activation="softmax"))
  return model


def train_keras(model, training_params, X_train, y_train, X_test):
  learning_rate = training_params.get("learning_rate", 1e-2)
  batch_size = training_params.get("batch_size", 32)
  plot = training_params.get("plot_history", True)
  validation_split = training_params.get("validation_split", 0.2)
  verbose = training_params.get("verbose", 0)

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
  lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

  history = model.fit(X_train,
                      y_train, 
                      validation_split=validation_split,
                      epochs=1000, 
                      batch_size=batch_size,
                      callbacks=[early_stopping, lr_scheduler], verbose=verbose)
  
  if plot:
    plt.plot(history.history['loss'][3:], label='train_loss')
    plt.plot(history.history['val_loss'][3:], label='val_loss')
    plt.plot(history.history['accuracy'][3:], label='train_accuracy')
    plt.plot(history.history['val_accuracy'][3:], label='val_accuracy')
    plt.legend()
    plt.show()
  
  min_val_loss = history.history['val_loss'][-30]
  min_val_acc = history.history['val_accuracy'][-30]
  print("min val_loss", min_val_loss, "max val_accuracy", min_val_acc)

  y_test_pred = model.predict(X_test)
  return min_val_acc, y_test_pred

vocab_size = 10000
max_len = 512
preprocess_params = {
  "no_number" : True,
  "no_stopwords" : True,
  "no_punctuation" : True,
  "lowercase" : True,
  "min_len" : 30
}
tokenization_params = {
    "vocab_size": vocab_size,
    "max_len": max_len
}

train, test = preprocess_data(
    df_train, 
    df_test,
    **preprocess_params
    )
tokenize_data(train, test, **tokenization_params)

train.target.plot.hist(bins=20, title="target distribution")
plt.show()

lengths = train.length.copy()
print(lengths.describe())
seq_max_len = 5000

lengths[lengths > seq_max_len] = seq_max_len
lengths.plot.hist(bins=500, title="text length distribution")
plt.show()

model_params = {
    "vocab_size": vocab_size,
    "embedding_size": 64,
    "hidden_size": 128
}
training_params = {
    "verbose": 1,
    "learning_rate": 1e-3,
    "batch_size": 256
}
model = create_simple_model(**model_params)

min_val_acc, y_test_pred = train_keras(model,
            training_params,
            np.array(train.encoded.to_list()), 
            train.target.to_numpy(),
            np.array(test.encoded.to_list())
            )

df_submission.target = y_test_pred.argmax(axis=-1)
df_submission.to_csv(f"submission_{min_val_acc:.4f}.csv", index=False)



