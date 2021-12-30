from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.eager.monitoring import Metric

#1. DATA
docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요',
       '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요',
       '글쎄요','별로에요','생각보다 지루해요','연기가 어색해요',
       '재미없어요','너무 재미없다','참 잼있네요','예람이가 잘 생기긴 했어요']

#긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각
# 보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '잼있네요': 24, '예람이가': 25, '생기긴': 26, '했어요': 27}

x = token.texts_to_sequences(docs)
print(x) #[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
# 13개의 리스트가 생성됨 => 13개의 x가 생김
#어절이 5개로 제일 많은 '한 번 더 보고 싶네요[11, 12, 13, 14, 15]'를 기준으로 공백을 채워 x값으로 만든다.
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre',maxlen=5)
print(pad_x) #post로 정렬하면 값이 앞으로, pre로 정렬하면 값이 뒤로 LSTM 시계열 분석 쌉가능!
print(pad_x.shape) #(13, 5) 이걸 어떻게 모델링 하지?

#post정렬
# [[ 2  4  0  0  0]  너무 재밌어요
#  [ 1  5  0  0  0]  참 최고에요
#  [ 1  3  6  7  0] 
#  [ 8  9 10  0  0] 
#  [11 12 13 14 15] 
#  [16  0  0  0  0] 
#  [17  0  0  0  0] 
#  [18 19  0  0  0] 
#  [20 21  0  0  0] 
#  [22  0  0  0  0] 
#  [ 2 23  0  0  0] 
#  [ 1 24  0  0  0] 
#  [25  3 26 27  0]]  

#pre정렬
# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]

word_size = len(token.word_index)
print('word_size:', word_size) # word_size: 27
print(np.unique(pad_x)) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

#원핫인코딩하면?
#대용량 데이터를 인코딩하면 문제는?


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

#2. MODEL
model = Sequential()
#                                                 인풋은 (13, 5)
#                   단어사전의 개수                단어수, 길이        28 * 10 = 280
#model.add(Embedding(input_dim=28, output_dim=10, input_length=5)) #텐서플로에서 유일하게 인룻딤이 아웃풋이 아니다.
#model.add(Embedding(28, 10, input_length=5))
model.add(Embedding(30, 10)) #단어사전갯수 이상으로 입력해야한다
model.add(LSTM(32))
model.add(Dense(20, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


#3. COMPILE, TRAIN
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=100, batch_size=32)


#4. EVALUATION 
acc = model.evaluate(pad_x, labels)[1]
print('acc:', acc)


####################### 실습 : 소스 완성시키기(결과는 긍정인지, 부정인지) ######################
x_predict = '나는 반장이 정말 재미없다 정말'
# print(type(x_predict))
x_predict = [x_predict]
# print(docs_x_predict)   # ['나는 반장이 정말 재미없다 정말']

x_predict2 = token.texts_to_sequences(x_predict)

pad_x2 = pad_sequences(x_predict2, padding='pre', maxlen=5)  # maxlen은 맞출(가장 긴)문장길이, padding='post'는 뒤쪽에 0을 채우겠다. padding='pre'는 앞쪽에 0을 채우겠다.
# print(pad_x2)   # [[ 0  0  0  0 23]]
# print(pad_x2.shape)   # (1, 5)

predict = model.predict(pad_x2)

print(predict)


#결과는?? 긍정일까 부정일까?

#[[0.20118253]]
