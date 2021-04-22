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

## 텐서플로우 기초 사용법
03
03
텐서 다뤄보기
다양한 타입의 기본 텐서 알아보기
텐서플로우 기초 사용법
Constant
Tensor
Sequence
Tensor
상수 텐서 시퀀스 텐서
Variable
Tensor
변수 텐서
상수 텐서(Constant Tensor)
03
Example
import tensorflow as tf
tf.constant(value, dtype= None, shape= None, name= ‘Const’)
텐서플로우 기초 사용법
import tensorflow as tf
# 상수형 텐서 생성
tensor_a = tf.constant(value, dtype=None, shape=None, name=None)
value : 반환되는 상수값 shape : Tensor의 차원
dtype : 반환되는 Tensor 타입 name : 텐서 이름
다양한 상수 텐서 생성하기
03
Example
import tensorflow as tf
tf.constant(value, dtype= None, shape= None, name= ‘Const’)
텐서플로우 기초 사용법
import tensorflow as tf
# 모든 원소 값이 0인 텐서 생성
tensor_b = tf.zeros(shape, dtype=tf.float32, name=None)
# 모든 원소 값이 1인 텐서 생성
tensor_c = tf.ones(shape, dtype=tf.float32, name=None)
시퀀스 텐서(Sequence Tensor)
03
Example
import tensorflow as tf
tf.constant(value, dtype= None, shape= None, name= ‘Const’)
텐서플로우 기초 사용법
import tensorflow as tf
# start에서 stop까지 증가하는 num 개수 데이터를 가진 텐서 생성
tensor_d = tf.linspace(start, stop, num, name=None)
start : 시작 값 stop : 끝 값
num : 생성할 데이터 개수 name : 텐서의 이름
시퀀스 텐서(Sequence Tensor)
03
Example
import tensorflow as tf
tf.constant(value, dtype= None, shape= None, name= ‘Const’)
텐서플로우 기초 사용법
import tensorflow as tf
# start에서 limit까지 delta씩 증가하는 데이터를 가진 텐서 생성
tensor_e = tf.range(start, limit, delta, name=None)
start : 시작 값 limit : 끝 값
delta : 증가량 name : 텐서의 이름
변수 텐서(Variable Tensor)
03
Example
import tensorflow as tf
tf.constant(value, dtype= None, shape= None, name= ‘Const’)
텐서플로우 기초 사용법
import tensorflow as tf
# 변수형 텐서 생성
tensor_f = tf.Variable(initial_value=None, dtype= None, name= None )
initial_value : 초기 값
dtype : 반환되는 Tensor 타입 name : 텐서의 이름
상수 텐서 생성 및 수식 정의
03
Example
import tensorflow as tf
tf.constant(value, dtype= None, shape= None, name= ‘Const’)
텐서플로우 기초 사용법
import tensorflow as tf
# 상수 텐서 생성
a = tf.constant([1,0],dtype=tf.float32)
# 수식 정의
def forward(x):
return W * x + b
정의된 수식을 활용한 연산
03
Example
텐서플로우 기초 사용법
tf.Tensor(
[[1. 0.]
[1. 0.]], shape=(2, 2), dtype=float32)
Result
# 텐서 계산 및 출력
output = forward(a)
print(output)
Confidential all right reserved
텐서플로우로 딥러닝 모델 구현하기
04
04 텐서플로우로 딥러닝 모델 구현하기
딥러닝 모델 구현 순서
1. 데이터셋 준비하기
2. 딥러닝 모델 구축하기
3. 모델 학습시키기
4. 평가 및 예측하기
04 텐서플로우로 딥러닝 모델 구현하기
1. 데이터셋 준비하기 : Epoch와 Batch
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

