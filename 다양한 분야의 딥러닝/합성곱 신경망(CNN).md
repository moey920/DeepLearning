# 합성곱 신경망(CNN) : Convolutional Nu

- 합성곱 신경망의 구조
    - Convolution Layer -> Pooling Layer Fully -> Connected Layer
    - 입력이미지의 특징을 추출, 분류하는 과정으로 동작

- Convolution Layer : 입력 이미지 * 필터(커널) = 피쳐맵
    - 이미지에서 어떠한 특징이 있는지를 구하는 과정
    - 필터가 이미지를 이동하며 새로운 이미지 (피쳐맵)를 생성
```
# 입력 이미지
I(0,0) I(0,1) I(0,2) I(0,3) I(0,4) I(0,5) I(0,6)
I(1,0) i(1,1) i(1,2) i(1,3) I(1,4) I(1,5) I(1,6)
I(2,0) I(2,1) I(2,2) I(2,2) …
…
✕
# 필터(커널)
H(0,0) H(0,1) H(0,2)
H(1,0) H(1,1) H(1,2)
H(2,0) H(2,1) H(2,2)
＝
# 피쳐맵
# 피쳐맵
O(0,0)
```

## 피쳐맵의 크기 변형 : Padding, Striding

- Padding : 원본 이미지의 상하좌우에 한 줄씩 추가
- Striding : 필터를 이동시키는 거리(Stride) 설정

## Pooling Layer
- 이미지의 왜곡의 영향(노이즈)를 축소하는 과정
- Max Pooling(Padding 내에서 가장 큰 값만 가져온다)
- Average Pooling(Padding 내의 평균 값을 가져온다)

## Fully Connected Layer(뉴럴 네트워크)

- 추출된 특징을 사용하여 이미지를 분류
- 분류를 위한 Softmax 활성화 함수 : 마지막 계층에 Softmax 활성화 함수 사용(모든 확률 중에서 가장 큰 확률이 무엇인지 결정)

## 정리 : 합성곱 - 풀링 - 활성함수

> Convolution Layer 는 특징을 찾아내고, Pooling Layer 는 처리할 맵(이미지) 크기를 줄여준다. 이를 N 번 반복한다.  
반복할 때마다 줄어든 영역에서의 특징을 찾게 되고, 영역의 크기는 작아졌기 때문에 빠른 학습이 가능해진다.

- 합성곱 신경망 기반 다양한 이미지 처리 기술
    - Object detection & segmentation : 무엇이 어디에 있다(Bounding box)
    - Super resolution (SR) : 고화질로 변환

### Keras로 CNN 구현하기

이전 실습과 마찬가지로 이번 실습에서도 MNIST 데이터를 사용합니다. 이번엔 테스트용 MNIST 데이터를 95% 이상의 정확도로 분류하는 CNN 모델을 만들고 학습 시켜보겠습니다.

**CNN in Keras**

일반적으로 CNN 모델은 **Convolution 레이어 - MaxPooling 레이어** 순서를 반복해 층을 쌓다가, 마지막 MaxPooling 레이어 다음에 **Flatten 레이어**를 하나 쌓고, 이후 몇 개의 **Dense 레이어**를 더 쌓아 완성합니다.

**CNN을 위한 데이터 전처리**

MNIST 데이터는 이미지 데이터이지만 가로 길이와 세로 길이만 존재하는 2차원 데이터입니다. **CNN 모델은 채널(RGB 혹은 흑백)까지 고려한 3차원 데이터를 입력**으로 받기에 **채널 차원을 추가해 데이터의 모양(shape)을 바꿔줍니다**. 결과는 아래와 같습니다.
`[데이터 수, 가로 길이, 세로 길이] -> [데이터 수, 가로 길이, 세로 길이, 채널 수]`

- Keras에서 CNN 모델을 만들기 위해 필요한 함수/라이브러리
    - `tf.keras.layers.Conv2D(filters, kernel_size, activation, padding)`  
    : 입력 이미지의 특징, 즉 처리할 특징 맵(map)을 추출하는 레이어입니다.
        - filters : 필터(커널) 개수
        - kernel_size : 필터(커널)의 크기
        - activation : 활성화 함수
        - padding : 이미지가 필터를 거칠 때 그 크기가 줄어드는 것을 방지하기 위해서 가장자리에 0의 값을 가지는 픽셀을 넣을 것인지 말 것인지를 결정하는 변수. SAME 또는 VALID
    - `tf.keras.layers.MaxPool2D(padding)`  
    : 처리할 특징 맵(map)의 크기를 줄여주는 레이어입니다.
        - padding : SAME 또는 VALID

    - `tf.keras.layers.Flatten()`  
    : Convolution layer 또는 MaxPooling layer의 결과는 N차원의 텐서 형태입니다. 이를 1차원으로 평평하게 만들어줍니다.
    - `tf.keras.layers.Dense(node, activation)`
        - node : 노드(뉴런) 개수
        - activation : 활성화 함수
    - `np.expand_dims(data, axis)`  
    : Numpy 배열 데이터에서 마지막 축(axis)에 해당하는 곳에 차원 하나를 추가할 수 있는 코드입니다.

- 결과
```
Test Loss : 0.1323 | Test Accuracy : 0.9539999961853027
예측한 Test Data 클래스 :  [7 2 1 0 4 1 4 9 5 9]
```

```
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from visual import *
from plotter import *

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

'''
1. MNIST 데이테 셋을 전처리하는 'preprocess' 함수를 완성합니다.
   
   Step01과 Step03은 이전 실습과 동일한 코드를 사용할 수 있습니다.

   Step01. MNIST 데이터 이미지를 0~1 사이 값으로 정규화해줍니다.
           원본은 0~255 사이의 값입니다.
           
   Step02. MNIST 데이터의 채널 차원을 추가해줍니다.
           
   Step03. 0~9 사이 값인 레이블을 클래스화 하기 위해 원-핫 인코딩을 진행합니다.
'''

def preprocess():
    
    # MNIST 데이터 세트를 불러옵니다.
    mnist = tf.keras.datasets.mnist
    
    # MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()    
    
    # Train 데이터 5000개와 Test 데이터 1000개를 사용합니다.
    train_images, train_labels = train_images[:5000], train_labels[:5000]
    test_images, test_labels = test_images[:1000], test_labels[:1000]
    
    train_images = train_images / 255.
    test_images = test_images / 255.
    
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    return train_images, test_images, train_labels, test_labels

'''
2. CNN 모델을 생성합니다.
'''
def CNN():
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME', input_shape = (28,28,1)))
    model.add(tf.keras.layers.MaxPool2D(padding = 'SAME'))
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME', input_shape = (28,28,1)))
    model.add(tf.keras.layers.MaxPool2D(padding = 'SAME'))
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'SAME', input_shape = (28,28,1)))
    model.add(tf.keras.layers.MaxPool2D(padding = 'SAME'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    
    return model
    
'''
3. 모델을 불러온 후 학습시키고 테스트 데이터에 대해 평가합니다.

   Step01. CNN 함수를 통해 모델을 불러옵니다.
   
   Step02. 모델의 손실 함수, 최적화 알고리즘, 평가 방법을 설정합니다.
   
   Step03. 모델의 구조를 확인하는 코드를 작성합니다.
   
   Step04. 모델을 학습시킵니다. 검증용 데이터도 설정하세요.
           'epochs'와 'batch_size'도 자유롭게 설정하세요.
           단, 'epochs'이 클수록, 'batch_size'는 작을수록 학습 속도가 느립니다.
   
   Step05. 모델을 테스트하고 손실(loss)값과 Test Accuracy 값 및 예측 클래스, 
           손실 함수값 그래프를 출력합니다. 모델의 성능을 확인해보고,
           목표값을 달성해보세요.
'''

def main():
    
    train_images, test_images, train_labels, test_labels = preprocess()
    
    model = CNN()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    history = model.fit(train_images, train_labels, epochs = 20, batch_size = 500, validation_data = (test_images, test_labels))
    
    loss, test_acc = model.evaluate(test_images, test_labels)
    
    print('\nTest Loss : {:.4f} | Test Accuracy : {}'.format(loss, test_acc))
    print('예측한 Test Data 클래스 : ',model.predict_classes(test_images)[:10])
    
    Visulaize([('CNN', history)], 'loss')
    
    Plotter(test_images, model)
    
    return history
    
if __name__ == "__main__":
    main()
```
