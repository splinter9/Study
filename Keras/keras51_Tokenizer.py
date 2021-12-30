from tensorflow.keras.preprocessing.text import Tokenizer

text = "나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다"

token = Tokenizer() #어절(띄어쓰기단위)로 자르기
token.fit_on_texts([text])
print(token.word_index) #반복횟수, 문장앞 순서대로 인덱싱
#{'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
x = token.texts_to_sequences([text])
print(x) #[[3, 1, 4, 5, 6, 1, 2, 2, 7]] 
#수치화 할때 주의점: 숫자와 의미값은 상관이 없지만 모델은 구분을 못한다. 그래서 원핫인코딩

from tensorflow.keras.utils import to_categorical
word_size=len(token.word_index)
print("word_size: ", word_size) #word_size:  7

x = to_categorical(x)
print(x)
print(x.shape) 
# [[[0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1.]]]
#(1, 9, 8) (1개 문장, 9개 어절, 7개 단어 + to categorical 0 1개 = 8 )
