# 텐서플로우(TensorFlow)

- 딥러닝 모델 구현을 위해 학습해야 할 분야 : 딥러닝 모델(파이썬) + 좋은 연산 장치(하드웨어) + 연산 장치 제어(C/C++) 등등 배울 것이 너무 많음
    - 데이터가 엄청 크면 분산 시스템 등의 엔지니어링 이슈도 있다. 주어진 시간과 메모리 안에서 해결하려면 단순 파이썬을 아는 것으로는 안된다.
    - 하드웨어와 소통하기 위해 하드웨어와 가까운 언어(C) 등도 필요하다.
    - 이러한 어려운 과정을 모든 연구자가 계속 구현할 수 없기 때문에 프레임워크를 통한 딥러닝 모델 구현이 탄생했다.

- 프레임워크를 통한 딥러닝 모델 구현
    - 딥러닝 모델의 학습과 추론을 위한 프로그램
    - 딥러닝 모델을 쉽게 구현, 사용가능
    - TensorFlow, Pytorch, Keras, Theano 등

- TensorFlow
    - 유연하고, 효율적이며, 확장성 있는 딥러닝 프레임워크
    - 대형 클러스터 컴퓨터부터 스마트폰까지 다양한 디바이스에서 동작

- 텐서(Tensor)

> Tensor = Multidimensional Arrays = Data : 딥러닝에서 텐서는 다차원 배열로 나타내는 데이터를 의미

- 플로우(Flow)
    - 플로우는 데이터의 흐름을 의미
    - 텐서플로우에서 계산은 데이터 플로우 그래프로 수행
    - 그래프를 따라 데이터가 노드를 거쳐 흘러가면서 계산

- 텐서+플로우 : 딥러닝에서 데이터를 의미하는 텐서(tensor)와 데이터 플로우 그래프를 따라 연산이 수행되는 형태(Flow)의 합

- 텐서플로우 version.1 : 이전 텐서플로우 1.X 에서는 계산을 위해 그래프 및 세션(Session) 생성 필요

- 직관적인 인터페이스의 텐서플로우 version.2 : 2019년 9월부터 배포된 2.0 버전 이후부터는 즉시 실행(Eager Execution)
    - 기능을 통해 계산 그래프, 세션 생성 없이 실행 가능

## 텐서플로우 버전 비교하기

```
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


"""""
텐서플로우 1.x 버전
"""""

def tf1():
    
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    
    # 상수
    a = tf.constant(5)
    b = tf.constant(3)
    
    # 계산 정의
    add_op = a + b
    
    # 세션 시작
    sess = tf.Session()
    result_tf1 = sess.run(add_op)
    
    return a, b, result_tf1

"""""
텐서플로우 2.0 버전
"""""

def tf2():
    
    import tensorflow as tf
    tf.compat.v1.enable_v2_behavior()
    
    # 상수
    a = tf.constant(5)
    b = tf.constant(3)
    
    # 즉시 실행 연산
    result_tf2 = tf.add(a, b)
    
    return a, b, result_tf2.numpy()

def main():
    
    tf_2, tf_1 = tf2()[2], tf1()[2]
    
    print('result_tf1:', tf_1)
    print('result_tf2:', tf_2)
    
if __name__ == "__main__":
    main()
```

## 텐서플로우 기초 사용법

- 텐서 다뤄보기 : 다양한 타입의 기본 텐서 알아보기
    - 상수 텐서(Constant Tensor)
    - 시퀀스 텐서(Sequence Tensor)
    - 변수 텐서(Variable Tensor)

- 상수 텐서(Constant Tensor)
    - value : 반환되는 상수값 
    - shape : Tensor의 차원
    - dtype : 반환되는 Tensor 
    - 타입 name : 텐서 이름
```
import tensorflow as tf
# 상수형 텐서 생성
tensor_a = tf.constant(value, dtype=None, shape=None, name=None)
```

```
import tensorflow as tf

# 모든 원소 값이 0인 텐서 생성
tensor_b = tf.zeros(shape, dtype=tf.float32, name=None)
# 모든 원소 값이 1인 텐서 생성
tensor_c = tf.ones(shape, dtype=tf.float32, name=None)
```

- 시퀀스 텐서(Sequence Tensor)
    - linspace의 경우(구간을 데이터 개수로 나눔)
        - start : 시작 값 
        - stop : 끝 값
        - num : 생성할 데이터 개수 
        - name : 텐서의 이름
    - range의 경우(일정한 증가량)
        - start : 시작 값 
        - limit : 끝 값
        - delta : 증가량 
        - name : 텐서의 이름

```
import tensorflow as tf
# start에서 stop까지 증가하는 num 개수 데이터를 가진 텐서 생성
tensor_d = tf.linspace(start, stop, num, name=None)
```
    
```
import tensorflow as tf
# start에서 limit까지 delta씩 증가하는 데이터를 가진 텐서 생성
tensor_e = tf.range(start, limit, delta, name=None)
```

- 변수 텐서(Variable Tensor)
    - initial_value : 초기 값
    - dtype : 반환되는 Tensor 타입 
    - name : 텐서의 이름

```
import tensorflow as tf
# 변수형 텐서 생성
tensor_f = tf.Variable(initial_value=None, dtype= None, name= None )
```

- 상수 텐서 생성 및 수식 정의

```
import tensorflow as tf
# 상수 텐서 생성
a = tf.constant([1,0],dtype=tf.float32)

# 수식 정의
def forward(x):
    return W * x + b

# 텐서 계산 및 출력
output = forward(a)
print(output)

'''result :
tf.Tensor(
        [[1. 0.]
        [1. 0.]], shape=(2, 2), dtype=float32)
'''
```

### 텐서 데이터 생성(상수, 시퀀스, 변수)

```
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 상수 텐서를 생성하는 constant_tensors 함수를 완성하세요.

   Step01. 5의 값을 가지는 (1,1) shape의 8-bit integer 텐서를 만드세요.
   
   Step02. 모든 원소의 값이 0인 (3,5) shape의 16-bit integer 텐서를 만드세요.
   
   Step03. 모든 원소의 값이 1인 (4,3) shape의 8-bit integer 텐서를 만드세요.
'''

def constant_tensors():
    
    t1 = tf.constant(5, shape = (1, 1), dtype=tf.int8)
    
    t2 = tf.zeros(shape = (3, 5), dtype = tf.int16)
    
    t3 = tf.ones(shape = (4, 3), dtype = tf.int8)
    
    return t1, t2, t3

'''
2. 시퀀스 텐서를 생성하는 sequence_tensors 함수를 완성하세요. 

   Step01. 1.5에서 10.5까지 증가하는 3개의 텐서를 만드세요.
   
   Step02. 2.5에서 20.5까지 증가하는 5개의 텐서를 만드세요. 
'''

def sequence_tensors():
    
    seq_t1 = tf.linspace(1.5, 10.5, 3)
    
    seq_t2 = tf.range(2.5, 21, 4.5)
    
    return seq_t1, seq_t2

'''
3. 변수를 생성하는 variable_tensor 함수를 완성하세요.

   Step01. 값이 100인 변수 텐서를 만드세요.
   
   Step02. 모든 원소의 값이 1인 (2,2) shape의 변수 텐서를 만드세요.
           이름도 'W'로 지정합니다.
   
   Step03. 모든 원소의 값이 0인 (2,) shape의 변수 텐서를 만드세요.
           이름도 'b'로 지정합니다.
'''

def variable_tensor():
    
    var_tensor = tf.Variable(initial_value = 100)
    
    W = tf.Variable(tf.ones(shape = (2, 2), name = 'W'))
    
    b = tf.Variable(tf.zeros(shape = (2,), name = 'B'))
    
    return var_tensor, W, b

def main():
    
    t1, t2, t3 = constant_tensors()
    
    seq_t1,seq_t2 = sequence_tensors()
    
    var_tensor, W, b = variable_tensor()
    
    constant_dict = {'t1':t1, 't2':t2, 't3':t3}
    
    sequence_dict = {'seq_t1':seq_t1, 'seq_t2':seq_t2}
    
    variable_dict = {'var_tensor':var_tensor, 'W':W, 'b':b}
    
    for key, value in constant_dict.items():
        print(key, ' :', value.numpy())
    
    print()
    
    for key, value in sequence_dict.items():
        print(key, ' :', value.numpy())
        
    print()
    
    for key, value in variable_dict.items():
        print(key, ' :', value.numpy())

if __name__ == "__main__":
    main()
```

### 텐서 연산

```
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 이항 연산자를 사용해 사칙 연산을 수행하여 각 변수에 저장하세요.

   Step01. 텐서 'a'와 'b'를 더해 'add'에 저장하세요.
   
   Step02. 텐서 'a'에서 'b'를 빼 'sub'에 저장하세요.
   
   Step03. 텐서 'a'와 'b'를 곱해 'mul'에 저장하세요.
   
   Step04. 텐서 'a'에서 'b'를 나눠 'div'에 저장하세요.
'''

def main():
    
    a = tf.constant(10, dtype = tf.int32)
    b = tf.constant(3, dtype = tf.int32)
    
    add = tf.add(a, b)
    sub = tf.subtract(a, b)
    mul = tf.multiply(a, b)
    div = tf.truediv(a, b)
    
    tensor_dict = {'add':add, 'sub':sub, 'mul':mul, 'div':div}
    
    for key, value in tensor_dict.items():
        print(key, ' :', value.numpy(), '\n')
    
    return add, sub, mul, div

if __name__ == "__main__":
    main()
```

### 텐서플로우로 사칙연산 계산기 구현하기

```
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
1. 두 실수와 연산 종류를 입력받는 함수입니다. 코드를 살펴보세요.
'''

def insert():
    
    x = float(input('정수 또는 실수를 입력하세요. x : '))
    y = float(input('정수 또는 실수를 입력하세요. y : '))
    cal = input('어떤 연산을 할것인지 입력하세요. (+, -, *, /)')
    
    return x, y, cal

'''
2. 입력받는 연산의 종류 cal에 따라 연산을 수행하고
   결과를 반환하는 calcul() 함수를 완성하세요.
'''

def calcul(x,y,cal):

    result = 0
    
    if cal == '+' :
        result = tf.add(x, y)
    
    elif cal == '-' :
        result = tf.subtract(x, y)
    
    elif cal == '*' :
        result = tf.multiply(x, y)
    
    elif cal == '/' :
        result = tf.truediv(x, y)
    
    return result.numpy()

'''
3. 두 실수와 연산 종류를 입력받는 insert 함수를 호출합니다. 그 다음
   calcul 함수를 호출해 실수 사칙연산을 수행하고 결과를 출력합니다.
'''

def main():
    
    x, y, cal = insert()
    
    print(calcul(x,y,cal))

if __name__ == "__main__":
    main()
```

# 텐서플로우로 딥러닝 모델 구현하기

딥러닝 모델 구현 순서   
1. 데이터셋 준비하기
2. 딥러닝 모델 구축하기
3. 모델 학습시키기
4. 평가 및 예측하기


- 1. 데이터셋 준비하기 : Epoch와 Batch
Epoch: 한 번의 epoch는 전체 데이터셋에 대해 한 번 학습을 완료한 상태
Batch: 나눠진 데이터셋 (보통 mini-batch라고 표현)
iteration는 epoch를 나누어서 실행하는 횟수를 의미
Batch size
04
Epoch와 Batch 예시
Data set
1 Epoch
Ex) 총 데이터가 1000개, Batch size = 100
• 1 iteration = 100개 데이터에 대해서 학습
• 1 epoch = 100/Batch size = 10 iteration
텐서플로우로 딥러닝 모델 구현하기
04
Example
data = np.random.sample((100,2))
labels = np.random.sample((100,1))
# numpy array로부터 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
Dataset API를 사용하여 딥러닝 모델용 데이터셋을 생성
데이터셋 준비하기 코드 예시
텐서플로우로 딥러닝 모델 구현하기
04 텐서플로우로 딥러닝 모델 구현하기
2. 딥러닝 모델 구축하기 : 고수준 API 활용
텐서플로우의 패키지로 제공되는 고수준 API
딥러닝 모델을 간단하고 빠르게 구현 가능
04 텐서플로우로 딥러닝 모델 구현하기
딥러닝 모델 구축을 위한 Keras 메소드(1)
모델 클래스 객체 생성
모델의 각 Layer 구성
• units : 레이어 안의 Node의 수
• activation : 적용할 activation 함수 설정
tf.keras.models.Sequential()
tf.keras.layers.Dense(units, activation)
04 텐서플로우로 딥러닝 모델 구현하기
Input Layer의 입력 형태 지정하기
…
Input Layer
…
1 Hidden Layer
…
N Hidden Layer Output Layer
… …
…
첫 번째 즉, Input Layer는 입력 형태에 대한 정보를 필요로 함
input_shape / input_dim 인자 설정하기
04 텐서플로우로 딥러닝 모델 구현하기
Example
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(10, input_dim=2, activation=‘sigmoid’),
tf.keras.layers.Dense(10, activation=‘sigmoid'),
tf.keras.layers.Dense(1, activation='sigmoid'),
])
모델 구축하기 코드 예시(1)
[model].add(tf.keras.layers.Dense(units, activation))
04 텐서플로우로 딥러닝 모델 구현하기
딥러닝 모델 구축을 위한 Keras 메소드(2)
모델에 Layer 추가하기
• units : 레이어 안의 Node의 수
• activation : 적용할 activation 함수 설정
04 텐서플로우로 딥러닝 모델 구현하기
Example
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=2, activation=‘sigmoid’))
model.add(tf.keras.layers.Dense(10, activation=‘sigmoid’))
model.add(tf.keras.layers.Dense(1, activation='sigmoid’))
모델 구축하기 코드 예시(2)
04 텐서플로우로 딥러닝 모델 구현하기
3. 딥러닝 모델 학습시키기 : Keras 메소드
모델 학습 방식을 설정하기 위한 함수
[model].compile(optimizer, loss)
• optimizer : 모델 학습 최적화 방법
• loss : 손실 함수 설정
모델을 학습시키기 위한 함수
[model].fit(x, y)
• x : 학습 데이터
• y : 학습 데이터의 label
04 텐서플로우로 딥러닝 모델 구현하기
Example
model.compile(loss='mean_squared_error’, optimizer=‘SGD')
model.fit(dataset, epochs=100)
딥러닝 모델 학습시키기 코드 예시
04 텐서플로우로 딥러닝 모델 구현하기
4. 평가 및 예측하기 : Keras 메소드
모델을 평가하기 위한 메소드
[model].evaluate(x, y)
• x : 테스트 데이터
• y : 테스트 데이터의 label
모델로 예측을 수행하기 위한 함수
[model].predict(x)
• x : 예측하고자 하는 데이터
04 텐서플로우로 딥러닝 모델 구현하기
Example
# 테스트 데이터 준비하기
dataset_test = tf.data.Dataset.from_tensor_slices((data_test, labels_test))
dataset_test = dataset.batch(32)
# 모델 평가 및 예측하기
model.evaluate(dataset_test)
predicted_labels_test = model.predict(data_test)

