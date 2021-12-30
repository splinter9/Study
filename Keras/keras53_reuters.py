from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(x_train, len(x_train), len(x_test))  #8982, 2246
print(y_train[0]) #3
print(np.unique(y_train)) #[ 0  1 ~ 44 45] 46개의 뉴스카테고리 데이터셋

print(type(x_train), type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape) #(8982,) (8982,)

print(len(x_train[0]), len(x_train[1])) #87 56
print(type(x_train[0]), type(x_train[1])) #<class 'list'> <class 'list'>

#print("뉴스기사의 최대길이:", max(len(x_train))) #error
print("뉴스기사의 최대길이:", max(len(i) for i in x_train))           #뉴스기사의 최대길이: 2376
print("뉴스기사의 평균길이:", sum(map(len, x_train))/(len(x_train)))  #뉴스기사의 평균길이: 145.5398574927633


#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
print(x_train.shape) # (8982, 2376) -> (8982, 100)
x_test = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
print(x_test.shape) # (2246, 100)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)   # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)     # (2246, 100) (2246, 46)

print(type(x_train), type(y_train))    
print(type(x_test), type(y_test)) 


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=68, input_length=100)) #텐서플로에서 유일하게 인룻딤이 아웃풋이 아니다.
#model.add(Embedding(28, 10, input_length=5))
#model.add(Embedding(100, 10)) #단어사전갯수 이상으로 입력해야한다
model.add(LSTM(48, activation='relu'))
model.add(Dense(46))
model.add(Dense(21))
model.add(Dense(46, activation='softmax'))
model.summary()


#3. COMPILE, TRAIN
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)


#4. EVALUATION 
acc= model.evaluate(x_train, y_test)[1]
print('acc:', acc)




'''
#################################################

words_to_index = reuters.get_word_index()
import operator
print(sorted(words_to_index.items(), key=operator.itemgetter(1)))

index_to_word = {}
for key, value in words_to_index.items():
    index_to_word[value+3] = key
    
for index, token in enumerate(("<pad","<sos>","<unk>")):
    index_to_word[index] = token
    
print(' '.join([index_to_word[index] for index in x_train[0]]))
'''
