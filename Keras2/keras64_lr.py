x = 0.1
y = 0.9
w = 0.4 
lr = 0.0001
epochs = 300

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2 #MSE
    
    # 가중치도 넣어서 아래 PRINT 수정
    print("Loss : ", round(loss, 4), '\t predict :', round(predict, 4))

up_predict = x * (w + lr)
up_loss = (y - up_predict) ** 2

down_predict = x * (w - lr)
down_loss = (y - down_predict) ** 2

if(up_loss > down_loss):
    w = w - lr
else:
    w = w + lr
    
print()
# print("Loss : ", round(loss, 4), '\t predict :', round(predict, 4))
# 2차함수 그래프에서 가중치를 줄여가면서 최소 로스를 찾는다.
