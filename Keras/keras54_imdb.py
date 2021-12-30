from tensorflow.keras.datasets import imdb
import numpy as np


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(np.unique(y_train))

print(x_train[0], y_train[0])



#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=200, truncating='pre')
print(x_train.shape) 
x_test = pad_sequences(x_train, padding='pre', maxlen=20, truncating='pre')
print(x_test.shape) 


# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)



#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()

model.add(Embedding(input_dim=10000, output_dim=68, input_length=200)) #텐서플로에서 유일하게 인룻딤이 아웃풋이 아니다.
#model.add(Embedding(28, 10, input_length=5))
#model.add(Embedding(30, 10)) #단어사전갯수 이상으로 입력해야한다
model.add(LSTM(48))
model.add(Dense(24, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3. COMPILE, TRAIN
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=100, validation_split=0.2)


#4. EVALUATION 
acc= model.evaluate(x_train, y_test)[1]
print('acc:', acc)





#################################################
'''
words_to_index = imdb.get_word_index()
import operator
print(sorted(words_to_index.items(), key=operator.itemgetter(1)))

index_to_word =
for key, value in words_to_index.items():
    index_to_word[value+3] = key
    
for index, token in enumerate(("<pad","<sos>","<unk>")):
    index_to_word[index] = token
    
print(' '.join([index_to_word[index] for index in x_train[0]]))
'''
