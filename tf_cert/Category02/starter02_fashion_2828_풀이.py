# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
    # print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)
    
    x_train = x_train/255.
    x_test = x_test/255.

    #2. 모델링
    model = Sequential()
    model.add(Conv1D(100, kernel_size = 2, padding = 'same', input_shape = (28,28)))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,4))
    model.add(Conv1D(128,4))
    model.add(Dropout(0.2))
    model.add(Conv1D(256,2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    
    #3. 컴파일, 훈련
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    es = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 15, verbose = 1)
    model.fit(x_train, y_train, epochs = 100, batch_size=100, validation_split = 0.2, callbacks = [es])
    
    
    #4. 평가, 예측
    
    loss = model.evaluate(x_test, y_test)
    print('loss:', loss)
    # loss: [0.4725392460823059, 0.8335999846458435]
    
    
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./tf_cert/Category02/mymodel.h5")
