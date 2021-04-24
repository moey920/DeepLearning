# 퍼셉트론 선형 분류기

- 논리 회로 역할을 수행하는 퍼셉트론
    - 사람의 신경계 : 뉴런 - 신경망 - 지능
    - 딥러닝 : 퍼셉트론 - 인공 신경망 - 인공지능

- AND, OR, NAND, NOR 등 논리회로는 선형 분류로 해결할 수 있다.(싱글 레이어 퍼셉트론으로 해결할 수 있다.)
- 단층 퍼셉트론(Single Layer Perceptron)
    - Input Layer에서 Output Layer로 바로 결과를 도출할 수 있다.(은닉층이 없다) == Linear Classfier

## 논리 회로의 정의

> 일정한 논리 연산에 의해 출력을 얻는 회로를 의미

1. AND gate

| A/B | C |
|:---:|:---:|
| 𝟎/𝟎 | 0 |
| 𝟏/𝟎 | 0 |
| 𝟎/𝟏 | 0 |
| **𝟏/𝟏** | **1** |

`Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛(1 ∗ 𝐴 + 1 ∗ 𝐵 − 1.5)`


2. OR gate

| A/B | C |
|:---:|:---:|
| **𝟎/𝟎** | **0** |
| 𝟏/𝟎 | 1 |
| 𝟎/𝟏 | 1 |
| 𝟏/𝟏 | 1 |

`Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛(1 ∗ 𝐴 + 1 ∗ 𝐵 − 0.5)`

3. NAND gate

| A/B | C |
|:---:|:---:|
| 𝟎/𝟎 | 1 |
| 𝟏/𝟎 | 1 |
| 𝟎/𝟏 | 1 |
| **𝟏/𝟏** | **0** |

`Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛((−1) ∗ 𝐴 + (−1) ∗ 𝐵 + 1.5)`

4. NOR gate

| A/B | C |
|:---:|:---:|
| **𝟎/𝟎** | **1** |
| 𝟏/𝟎 | 0 |
| 𝟎/𝟏 | 0 |
| 𝟏/𝟏 | 0 |

`Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛((−1) ∗ 𝐴 + (−1) ∗ 𝐵 + 0.5)`

- 입력층(Input Layer) : 외부로부터 데이터를 입력 받는 신경망 입구의 Layer
- 출력층(Output Layer) : 모델의 최종 연산 결과를 내보내는 신경망 출구의 Layer

### 퍼셉트론을 활용한 선형 분류기

0, 1 데이터를 계산하던 퍼셉트론 논리 회로에서 확장
- 선형 분류기로써 데이터 분류 가능 : 2차원 공간에서 강아지와 고양이를 분류

#### AND gate와 OR gate 구현

```
import numpy as np

'''
1. AND_gate 함수를 완성하세요.

   Step01. 입력값 x1과 x2에 각각 곱해줄 가중치는
           0.5, 0.5로 설정되어 있습니다.
           
   Step02. AND_gate를 만족하는 Bias 값을
           설정합니다. 여러 가지 값을 대입해보며
           적절한 Bias 값을 찾아보세요.
   
   Step03. 가중치, 입력값, Bias를 이용하여 
           신호의 총합을 구합니다.
           
   Step04. Step Function 함수를 호출하여 
           AND_gate의 출력값을 반환합니다.
'''

def AND_gate(x1, x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([0.5,0.5])
    
    bias = -0.7
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y)
    
'''
2. OR_gate 함수를 완성하세요.

   Step01. 입력값 x1과 x2에 각각 곱해줄 가중치는
           0.5, 0.5로 설정되어 있습니다.
           
   Step02. OR_gate를 만족하는 Bias 값을
           설정합니다. 여러 가지 값을 대입해보며
           적절한 Bias 값을 찾아보세요.
   
   Step03. 가중치, 입력값, Bias를 이용하여 
           신호의 총합을 구합니다.
           
   Step04. Step Function 함수를 호출하여 
           OR_gate의 출력값을 반환합니다.
'''

def OR_gate(x1, x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([0.5,0.5])
    
    bias = -0.3
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y)

'''
3. 설명을 보고 Step Function을 완성합니다.

   Step01. 0 미만의 값이 들어오면 0을,
           0 이상의 값이 들어오면 1을
           출력하는 함수를 구현하면 됩니다.
'''
def Step_Function(y):
    
    return 1 if y >= 0 else 0
    
def main():
    
    # AND Gate와 OR Gate에 넣어줄 Input
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # AND Gate를 만족하는지 출력하여 확인
    print('AND Gate 출력')
    
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ',AND_gate(x1, x2))
    
    # OR Gate를 만족하는지 출력하여 확인
    print('\nOR Gate 출력')
    
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ',OR_gate(x1, x2))

if __name__ == "__main__":
    main()
```

#### NAND gate와 NOR gate 구현

```
import numpy as np

'''
1. NAND_gate 함수를 완성하세요.

   Step01. 이전 실습을 참고하여 입력값 x1과 x2를
           Numpy array 형식으로 정의한 후, x1과 x2에
           각각 곱해줄 가중치도 Numpy array 형식으로 
           적절히 설정해주세요.
           
   Step02. NAND_gate를 만족하는 Bias 값을
           적절히 설정해주세요.
           
   Step03. 가중치, 입력값, Bias를 이용하여 
           가중 신호의 총합을 구합니다.
           
   Step04. Step Function 함수를 호출하여 
           NAND_gate의 출력값을 반환합니다.
'''

def NAND_gate(x1, x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([-0.5, -0.5])
    
    bias = 0.7
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y)

'''
2. NOR_gate 함수를 완성하세요.

   Step01. 마찬가지로 입력값 x1과 x2를 Numpy array 
           형식으로 정의한 후, x1과 x2에 각각 곱해줄
           가중치도 Numpy array 형식으로 적절히 설정해주세요.
           
   Step02. NOR_gate를 만족하는 Bias 값을
           적절히 설정해주세요.
           
   Step03. 가중치, 입력값, Bias를 이용하여 
           가중 신호의 총합을 구합니다.
           
   Step04. Step Function 함수를 호출하여 
           NOR_gate의 출력값을 반환합니다.
'''

def NOR_gate(x1, x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([-0.5, -0.5])
    
    bias = 0.3
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y) 

'''
3. 설명을 보고 Step Function을 완성합니다.
   앞 실습에서 구현한 함수를 그대로 
   사용할 수 있습니다.

   Step01. 0 미만의 값이 들어오면 0을,
           0 이상의 값이 들어오면 1을
           출력하는 함수를 구현하면 됩니다.
'''

def Step_Function(y):
    
    return 1 if y >=0 else 0  

def main():
    
    # NAND와 NOR Gate에 넣어줄 Input
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # NAND Gate를 만족하는지 출력하여 확인
    print('NAND Gate 출력')
    
    for x1, x2 in array:
        print('Input: ',x1, x2, ' Output: ',NAND_gate(x1, x2))
    
    # NOR Gate를 만족하는지 출력하여 확인
    print('\nNOR Gate 출력')
    
    for x1, x2 in array:
        print('Input: ',x1, x2, ' Output: ',NOR_gate(x1, x2))

if __name__ == "__main__":
    main()
```

### 퍼셉트론 선형 분류기를 이용해 붓꽃 데이터 분류하기(1) : sklearn Perceptron 활용

```
import numpy as np
import pandas as pd

# sklearn 모듈들
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from sklearn.svm import SVC

from elice_utils import EliceUtils
elice_utils = EliceUtils()

np.random.seed(100)

'''
1. iris 데이터를 불러오고, 
   불러온 데이터를 학습용, 테스트용 데이터로 
   분리하여 반환하는 함수를 구현합니다.
   
   Step01. 불러온 데이터를 학습용 데이터 80%, 
           테스트용 데이터 20%로 분리합니다.
           
           일관된 결과 확인을 위해 random_state를 
           0으로 설정합니다.        
'''

def load_data():
    
    iris = load_iris()
    
    X = iris.data[:,2:4]
    Y = iris.target
    
    # random_state를 고정하지 않으면 split을 할 때마다 데이터가 random하게 잘린다.
    # train_test_split 함수는 의미를 이해하고 써야한다. 시계열 데이터는 먼저 슬라이싱이 필요하다. 순서가 바뀌기 때문에.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
    
    return X_train, X_test, Y_train, Y_test
    
'''
2. 사이킷런의 Perceptron 클래스를 사용하여 
   Perceptron 모델을 정의하고,
   학습용 데이터에 대해 학습시킵니다.
   
   Step01. 앞서 완성한 함수를 통해 데이터를 불러옵니다.
   
   Step02. Perceptron 모델을 정의합니다.
           max_iter와 eta0를 자유롭게 설정해보세요.
   
   Step03. 학습용 데이터에 대해 모델을 학습시킵니다.
   
   Step04. 테스트 데이터에 대한 모델 예측을 수행합니다. 
'''

def main(): 

    X_train, X_test, Y_train, Y_test = load_data()
    
    '''
    sklearn으로 해결하기
    '''
    
    perceptron = Perceptron(max_iter = 3000, eta0 = 0.2)
    
    # fit 함수는 항상 train 데이터를 인풋으로 받아 관계를 찾아낸다.
    perceptron.fit(X_train, Y_train)
    
    pred = perceptron.predict(X_test)
    
    accuracy = accuracy_score(pred, Y_test)
    
    print("Test 데이터에 대한 정확도 : %0.5f" % accuracy)
    
    return X_train, X_test, Y_train, Y_test, pred

if __name__ == "__main__":
    main()
```

### 퍼셉트론 선형 분류기를 이용해 붓꽃 데이터 분류하기(2) : pandas DataFrame, sklearn SVM 활용

```
import numpy as np
import pandas as pd

# sklearn 모듈들
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

from sklearn.svm import SVC

from elice_utils import EliceUtils
elice_utils = EliceUtils()

np.random.seed(100)

def main():   
    
    '''
    sklearn은 딕셔너리 형태로 제공된다. 데이터를 확인하는 부분
    iris = load_iris()
    print(iris.keys())
    print(iris.target[:5]) #[0 0 0 0 0] 같은 데이터가 반복되기 때문에 랜덤하게 섞어주는 것이 좋겠다.
    print(iris.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # print(iris.DESCR) # 데이터 제작자가 제작한 데이터 설명문을 보여준다.
    print(iris.data.shape) # numpy의 shape 속성 이용 (150, 4)
    print(iris.target.shape) # (150,)
    '''
    
    '''
    numpy 배열을 pandas df로 변환하기
    '''
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns = iris.feature_names)
    df_iris.columns = ['SL', 'SW', 'PL','PW'] # sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    # print(df_iris.head())
    '''
           sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0                5.1               3.5                1.4               0.2
    1                4.9               3.0                1.4               0.2
    2                4.7               3.2                1.3               0.2
    3                4.6               3.1                1.5               0.2
    4                5.0               3.6                1.4               0.2
    '''
    
    # 예측해야하는 y값도 열에 추가하기
    df_iris['Y'] = iris.target
    # print(df_iris.tail())
    '''
          SL   SW   PL   PW  Y
    145  6.7  3.0  5.2  2.3  2
    146  6.3  2.5  5.0  1.9  2
    147  6.5  3.0  5.2  2.0  2
    148  6.2  3.4  5.4  2.3  2
    149  5.9  3.0  5.1  1.8  2
    '''
    
    # EDA 과정
    # print(df_iris.info())
    '''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
    SL    150 non-null float64
    SW    150 non-null float64
    PL    150 non-null float64
    PW    150 non-null float64
    Y     150 non-null int64
    dtypes: float64(4), int64(1)
    '''
    
    # EDA(2)
    # print(df_iris.describe())
    ''' Feature Scaling : 0~1 범위로 맞추는 개념이 필요하다는 것을 알 수 있다
                   SL          SW          PL          PW           Y
    count  150.000000  150.000000  150.000000  150.000000  150.000000(각 열의 데이터)
    mean     5.843333    3.057333    3.758000    1.199333    1.000000(평균)
    std      0.828066    0.435866    1.765298    0.762238    0.819232(표준편차)
    min      4.300000    2.000000    1.000000    0.100000    0.000000(최소값)
    25%      5.100000    2.800000    1.600000    0.300000    0.000000(1사분위)
    50%      5.800000    3.000000    4.350000    1.300000    1.000000(중앙값)
    75%      6.400000    3.300000    5.100000    1.800000    2.000000(3사분위)
    max      7.900000    4.400000    6.900000    2.500000    2.000000(최대값)
    '''
    
    # 결측치가 존재할 경우 결측치의 개수 세기
    # print( df_iris.isnull().sum())
    '''
    SL    0
    SW    0
    PL    0
    PW    0
    Y     0
    dtype: int64
    '''
    
    # 중복값 찾기 : 해당 데이터가 두 배의 영향력을 가지기 때문에 모델의 성능을 떨어뜨린다.
    # print(df_iris.duplicated().sum()) # duplicated() :중복 데이터가 있을 경우 True를 반환한다
    # 1 
    
    # 중복값 제거하기
    df_iris = df_iris.drop_duplicates()
    # print(df_iris.duplicated().sum()) # 0
    # print(df_iris.shape) # (149, 5)
    
    # 상관계수 확인하기(corr이 각 열의 상관계수를 계산해준다)
    # print(df_iris.corr())
    '''
              SL        SW        PL        PW         Y
    SL  1.000000 -0.118129  0.873738  0.820620  0.786971
    SW -0.118129  1.000000 -0.426028 -0.362894 -0.422987
    PL  0.873738 -0.426028  1.000000  0.962772  0.949402
    PW  0.820620 -0.362894  0.962772  1.000000  0.956514
    Y   0.786971 -0.422987  0.949402  0.956514  1.000000
    '''
    
    # Feature Engineering : 파생 변수 사용
    df_iris['S_ratio'] = df_iris['SL'] / df_iris['SW']
    df_iris['P_ratio'] = df_iris['PL'] / df_iris['PW']
    # print(df_iris.head())
    '''
        SL   SW   PL   PW  Y   S_ratio  P_ratio
    0  5.1  3.5  1.4  0.2  0  1.457143      7.0
    1  4.9  3.0  1.4  0.2  0  1.633333      7.0
    2  4.7  3.2  1.3  0.2  0  1.468750      6.5
    3  4.6  3.1  1.5  0.2  0  1.483871      7.5
    4  5.0  3.6  1.4  0.2  0  1.388889      7.0
    '''
    
    # 데이터프레임을 이용해서 X, Y 정의하기
    X = df_iris.loc[:, 'S_ratio':'P_ratio'] # 모든 행을 선택하고, 열 중에는 SL, SW, PL, PW를 선택한다, 어떤 feature를 이용해 분석할 것인지도 분석가의 몫이다.
    Y = df_iris.loc[:, 'Y'] # 모든 행을 선택하고 열은 Y를 선택한다
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True, random_state = 2021) # random_state도 학습률에 영향을 주는 파라미터이다.
    
    # print(X_train.shape, Y_train.shape) # (119, 4) (119,)
    # print(X_test.shape, Y_test.shape) # (30, 4) (30,)
    
    # SVM 사용
    svc_model = SVC()
    # 의사결정트리 사용
    dtc_model = DecisionTreeClassifier()
    # 앙상블 - 랜덤포레스트 모델 사용
    rfc_model = RandomForestClassifier()
    
    svc_model.fit(X_train, Y_train)
    dtc_model.fit(X_train, Y_train)
    rfc_model.fit(X_train, Y_train)
    
    svc_pred = svc_model.predict(X_test)
    dtc_pred = dtc_model.predict(X_test)
    rfc_pred = rfc_model.predict(X_test)
    
    svc_accuracy = accuracy_score(svc_pred, Y_test)
    dtc_accuracy = accuracy_score(dtc_pred, Y_test)
    rfc_accuracy = accuracy_score(rfc_pred, Y_test)
    
    print("SVC Test 데이터에 대한 정확도 : %0.5f" % svc_accuracy)
    print("DecisionTreeClassifier Test 데이터에 대한 정확도 : %0.5f" % dtc_accuracy)
    print("RandomForestClassifier Test 데이터에 대한 정확도 : %0.5f" % rfc_accuracy)


if __name__ == "__main__":
    main()
```


```
    '''
    keras로 해결하기
    '''
#     perceptron = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(128, activation = 'relu'),
#         tf.keras.layers.Dense(1, activation = 'sigmoid')
#     ])
    
#     perceptron.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
#     hist = perceptron.fit(X_train, Y_train, epochs = 20, batch_size = 500, verbose = 2, validation_data = (X_test, Y_test))
    
#     pred = perceptron.evaluate(X_test, Y_test)
    
#     accuracy = accuracy_score(pred, Y_test)
    
#     print("Test 데이터에 대한 정확도 : %0.5f" % accuracy)
    
#     return X_train, X_test, Y_train, Y_test, pred
```

