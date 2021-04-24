# Fashion-MNIST 데이터 분류하기(1) : Keras MLP + dropout

```
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import elice_utils
elice_utils = elice_utils.EliceUtils()

np.random.seed(100)
tf.random.set_seed(100)

'''
1. 다층 퍼셉트론 분류 모델을 만들고, 학습 방법을 설정해 
   학습시킨 모델을 반환하는 MLP 함수를 구현하세요.
   
   Step01. 다층 퍼셉트론 분류 모델을 생성합니다. 
           여러 층의 레이어를 쌓아 모델을 구성해보세요.
           
   Step02. 모델의 손실 함수, 최적화 방법, 평가 방법을 설정합니다.
   
   Step03. 모델을 학습시킵니다. epochs를 자유롭게 설정해보세요.
'''

def MLP(x_train, y_train):
    
    model = tf.keras.models.Sequential([
        Dense(128, activation = 'relu'),
        Dense(256, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dropout(0.2),
        Dense(64, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(32, activation = 'relu'),
        Dense(10, activation = 'softmax'),
    ])
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # loss 함수와 출력층 활성화 함수를 통해 데이터를 파악할 수 있다. 'softmax + sparse_categorical_crossentropy'의 경우 원-핫 인코딩이 안된 데이터이다.
    # 'softmax + categorical_crossentropy'의 경우 원-핫 인코딩이 된 데이터이다.
    return model

def main():
    
    x_train = np.loadtxt('./data/train_images.csv', delimiter =',', dtype = np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter =',', dtype = np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter =',', dtype = np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter =',', dtype = np.float32)
    
    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = MLP(x_train,y_train)
    
    model.fit(x_train, y_train, epochs = 40, batch_size = 50, validation_data = (x_test, y_test), verbose = 2)
    
    # 학습한 모델을 test 데이터를 활용하여 평가합니다.
    loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print('\nTEST 정확도 :', test_acc)
    
    # 임의의 3가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
    predictions = model.predict(x_test)
    rand_n = np.random.randint(100, size=3)
    
    for i in rand_n:
        img = x_test[i].reshape(28,28)
        plt.imshow(img,cmap="gray")
        plt.show()
        plt.savefig("test.png")
        elice_utils.send_image("test.png")
        
        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))

if __name__ == "__main__":
    main()
```

# Fashion-MNIST 데이터 분류하기(2) : 전통적인 sklearn방식 이용하기(SVC, DecisionTreeClassifier, RandomForestClassifier)

- 결과
```
SVC : Test 데이터에 대한 정확도 : 0.81600
DecisionTreeClassifier : Test 데이터에 대한 정확도 : 1.00000
RandomForestClassifier : Test 데이터에 대한 정확도 : 0.99100
```

```
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # 의사결정구조 모델
from sklearn.ensemble import RandomForestClassifier # 앙상블 모델

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import elice_utils
elice_utils = elice_utils.EliceUtils()

np.random.seed(100)
tf.random.set_seed(100)

def main():
    
    x_train = np.loadtxt('./data/train_images.csv', delimiter =',', dtype = np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter =',', dtype = np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter =',', dtype = np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter =',', dtype = np.float32)
    
    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # SVM 사용
    svc_model = SVC()
    # 의사결정트리 분류기 사용
    dtc_model = DecisionTreeClassifier()
    # 앙상블 - 랜덤포레스트 분류기 사용
    rfc_model = RandomForestClassifier()
    
    svc_model.fit(x_train, y_train)
    dtc_model.fit(x_train, y_train)
    rfc_model.fit(x_train, y_train)
    
    svc_pred = svc_model.predict(x_test)
    dtc_pred = dtc_model.predict(x_test)
    rfc_pred = rfc_model.predict(x_test)
    
    svc_accuracy = accuracy_score(svc_pred, y_test)
    dtc_accuracy = accuracy_score(dtc_pred, y_test)
    rfc_accuracy = accuracy_score(rfc_pred, y_test)
    
    print("SVC : Test 데이터에 대한 정확도 : %0.5f" % svc_accuracy)
    print("DecisionTreeClassifier : Test 데이터에 대한 정확도 : %0.5f" % dtc_accuracy)
    print("RandomForestClassifier : Test 데이터에 대한 정확도 : %0.5f" % rfc_accuracy)

if __name__ == "__main__":
    main()
```

# Fashion-MNIST 데이터 분류하기(3) : CNN 이용하기

- 결과  
`TEST 정확도 : 1.0`

```
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import elice_utils
elice_utils = elice_utils.EliceUtils()

np.random.seed(100)
tf.random.set_seed(100)


'''
input_dim은 2차원배열에만 사용할 수 있다. input_shape을 사용해야한다.
합성곱 연산층에 필요한 파라미터 : 필터, 커널 사이즈, 인풋형태(처음에만), 활성화함수
필터는 필터 내에 0과 1의 위치가 각기 다른데, 이런 다른 종류를 16개 사용하겠다는 의미이다. 
https://en.wikipedia.org/wiki/Kernel_(image_processing)
필터는 원본 이미지에서 서로 다른 특징을 찾아낸다. 필터는 보통 2의 거듭제곱 개수를 사용한다.
합성곱 연산은 정보가 많이 늘어나기 때문에, 보통 convolusion 다음에 polling을 진행한다(쌍으로 사용한다)
여기까지의 과정을 특징을 추출한다해서 feature extraction 이라고 한다.
특징을 추출하고 Flatten 층을 통해 1차원으로 바꾸어서 Dense 층을 거쳐 결과를 낸다.
큰 이미지의 경우 stride옵션을 주어서 움직이는 칸의 개수를 조절할 수 있으며
conv2D층에서 이미지가 (2,2,)씩 줄어드는 것을 방지하고 싶으면 padding = 'same' 옵션을 사용하면 된다.
'''
def CNN(x_train, y_train): # 합성공 신경망 모델 생성, 컴파일, 학습
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.fit(x_train, y_train, epochs = 50, batch_size = 32, verbose = 2)

    return model

def main():
    
    x_train = np.loadtxt('./data/train_images.csv', delimiter =',', dtype = np.float32)
    y_train = np.loadtxt('./data/train_labels.csv', delimiter =',', dtype = np.float32)
    x_test = np.loadtxt('./data/test_images.csv', delimiter =',', dtype = np.float32)
    y_test = np.loadtxt('./data/test_labels.csv', delimiter =',', dtype = np.float32)
    
    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # 원본 이미지 형태로 변환하는 과정
    x_train = x_train.reshape(-1, 28, 28, 1) # 흑백 이미지(rgb 채널값이 1(한개)) 색이 있는 이미지면 (-1, 28, 28, 3), -1을 지정해두면 자동으로 데이터 숫자에 따라 지정된다.(50000장이면 50000,)
    x_test = x_test.reshape(-1, 28, 28, 1) 
    
    model = CNN(x_train,y_train)
    
    # 학습한 모델을 test 데이터를 활용하여 평가합니다.
    loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print('\nTEST 정확도 :', test_acc)
    
    # 임의의 3가지 test data의 이미지와 레이블값을 출력하고 예측된 레이블값 출력
    predictions = model.predict(x_test)
    rand_n = np.random.randint(100, size=3)
    
    for i in rand_n:
        img = x_test[i].reshape(28,28)
        plt.imshow(img,cmap="gray")
        plt.show()
        plt.savefig("test.png")
        elice_utils.send_image("test.png")
        
        print("Label: ", y_test[i])
        print("Prediction: ", np.argmax(predictions[i]))
    
    

if __name__ == "__main__":
    main()
```

# 컬러 이미지(10가지 클래스) CNN으로 분류하기

```
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import elice_utils
elice_utils = elice_utils.EliceUtils()

np.random.seed(100)
tf.random.set_seed(100)

def CNN(x_train, y_train): # 합성공 신경망 모델 생성, 컴파일, 학습
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding = 'same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), padding = 'same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), padding = 'same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model.summary()

    return model

def main():

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
    print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)
    
    # 이미지 데이터를 0~1범위의 값으로 바꾸어 줍니다.(픽셀을 0~1로 압축시키는 것)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = CNN(x_train,y_train)
    
    model.fit(x_train, y_train, epochs = 15, batch_size = 16, verbose = 2, validation_data = (x_test, y_test))
    
    # 학습한 모델을 test 데이터를 활용하여 평가합니다.
    loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    
    print('\nTEST 정확도 :', test_acc)
    

if __name__ == "__main__":
    main()
```

## 내 질문

1. 전이 학습에서 사전 학습된 모델 중 어떤 것을 가져와서 쓸지 어떻게 정하나요?  
개/고양이 분류문제에서 사전 학습된 모델을 쓰려면 그냥 이미지 분류 모델에서 성능이 좋은 모델 아무거나 가져와서 써도 되는건가요? 아니면 개/고양이 분류모델을 학습한 특정 모델을 골라야하는건가요? 

> 모델의 예측력과, 모델의 처리시간이 모두 다르다. 이미지 분류의 경우 imagenet을 학습한 어떤 모델이라도 상관없지만 예측력과 처리시간을 고려하라

- 예를 들어 안면인식, 지문인식의 경우 사람이 느낄 정도로 처리시간이 오래걸리면 안된다. 모델에 따라 처리속도(inference time), 예측력 측면에서 trade-off가 있고 특정한 데이터셋에 대해 그러한 지표를 분석해놓은 정보도 있다. 

2. 합성곱 신경망에서 필터 내의 0이 아닌 숫자들이 가중치라고 생각해도 되나요?
합성곱의 의미가 필터와 겹치는 데이터가 필터의 숫자와 곱해진 다음 모두 더해지는 것인가요?

> 필터는 사진에서 유래되었다. blur, constast 등의 다양한 필터가 생겼다. 수학자들이 이러한 이미지를 연구해서 나온 필터의 값들이 딥러닝과 만난 것이다. 딥러닝의 합성곱 연산에서 사용해서 이미지 처리 분야에서 폭발적으로 발전한 것. 
