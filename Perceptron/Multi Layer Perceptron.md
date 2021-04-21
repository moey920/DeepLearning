# 다층 퍼셉트론(Multi Layer Perceptron)

- 단층 퍼셉트론은 입력층과 출력층만 존재
- 단층 퍼셉트론을 여러 층으로 쌓아보기 : 다층 퍼셉트론
- XOR 연산은 하나의 레이어를 사용하여 표현하는 것은 불가능
    - 하지만, NAND와 OR 연산을 함께 사용할 시 표현 가능
    - Hidden Layer에 NAND와 OR 게이트를 넣어준다. => XOR 게이트를 표현할 수 있다.
- Hidden Layer : 입력층과 출력층 사이의 모든 Layer(층)
- **Hidden Layer가 3층 이상**일 시 깊은 신경망이라는 뜻에서 **Deep Learning**이라고 한다.
    - 모델의 complexity가 올라간다. 모델이 파워풀한 representation이 가능하다. 
    - `y = 2x + 3` => `y = 2x^4 + 3x^2 + 3`
- 1 Hidden Layer : 선형 분류
- 2 Hidden Layer : 선이 2개여서, 구역을 분류
- n Hidden Layer : 수많은 표현이 가능(variety Classification Boundary)

## 다층 퍼셉트론으로 XOR gate 구현하기

```
import numpy as np

'''
1. AND_gate 함수를 완성하세요. 
'''

def AND_gate(x1,x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([0.5, 0.5])
    
    bias = -0.7
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y) 

'''
2. OR_gate 함수를 완성하세요.
'''

def OR_gate(x1,x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([0.5, 0.5])
    
    bias = -0.3
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y) 

'''
3. NAND_gate 함수를 완성하세요.
'''

def NAND_gate(x1,x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([-0.5, -0.5])
    
    bias = 0.7
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y) 

'''
4. Step_Function 함수를 완성하세요.
'''

def Step_Function(y):
    
    return 1 if y >= 0 else 0

'''
5. AND_gate, OR_gate, NAND_gate 함수들을
   활용하여 XOR_gate 함수를 완성하세요. 앞서 만든
   함수를 활용하여 반환되는 값을 정의하세요.
'''

def XOR_gate(x1, x2):
    
    nand_out = NAND_gate(x1, x2)
    or_out = OR_gate(x1, x2)

    return AND_gate(nand_out, or_out)
    
def main():
    
    # XOR gate에 넣어줄 Input
    array = np.array([[0,0], [0,1], [1,0], [1,1]])
    
    # XOR gate를 만족하는지 출력하여 확인
    print('XOR Gate 출력')
    
    for x1, x2 in array:
        print('Input: ',x1, x2, ', Output: ', XOR_gate(x1, x2))

if __name__ == "__main__":
    main()
```

### 다층 퍼셉트론(MLP) 모델로 2D 데이터 분류하기

```
import numpy as np
from visual import *
from sklearn.neural_network import MLPClassifier

from elice_utils import EliceUtils
elice_utils = EliceUtils()

import warnings
warnings.filterwarnings(action='ignore')

np.random.seed(100)
    
# 데이터를 읽어오는 함수입니다.
    
def read_data(filename):
    
    X = []
    Y = []
    
    with open(filename) as fp:
        N, M = fp.readline().split()
        N = int(N)
        M = int(M)
        
        for i in range(N):
            line = fp.readline().split()
            for j in range(M):
                X.append([i, j])
                Y.append(int(line[j]))
    
    X = np.array(X)
    Y = np.array(Y)
    
    return (X, Y)

'''
1. MLPClassifier를 정의하고 hidden_layer_sizes를
   조정해 hidden layer의 크기 및 레이어의 개수를
   바꿔본 후, 학습을 시킵니다.
'''

def train_MLP_classifier(X, Y):
    
    clf = MLPClassifier(hidden_layer_sizes=(100, 100))
    
    clf.fit(X, Y)
    
    return clf

'''
2. 테스트 데이터에 대한 정확도를 출력하는 
   함수를 완성합니다. 설명을 보고 score의 코드를
   작성해보세요.
'''

def report_clf_stats(clf, X, Y):
    
    hit = 0
    miss = 0
    
    for x, y in zip(X, Y):
        if clf.predict([x])[0] == y:
            hit += 1
        else:
            miss += 1
    
    score = (hit / len(X)) * 100
    
    print("Accuracy: %.1lf%% (%d hit / %d miss)" % (score, hit, miss))

'''
3. main 함수를 완성합니다.

   Step01. 학습용 데이터인 X_train, Y_train과
           테스트용 데이터인 X_test, Y_test를 각각
           read_data에서 반환한 값으로 정의합니다. 
           
           우리가 사용할 train.txt 데이터셋과
           test.txt 데이터셋은 data 폴더에 위치합니다.
           
   Step02. 앞에서 학습시킨 다층 퍼셉트론 분류 
           모델을 'clf'로 정의합니다. 'clf'의 변수로는
           X_train과 Y_train으로 설정합니다.
           
   Step03. 앞에서 완성한 정확도 출력 함수를
           'score'로 정의합니다. 'score'의 변수로는
           X_test와 Y_test로 설정합니다.
'''

def main():
    
    X_train, Y_train = read_data('data/train.txt')
    
    X_test, Y_test = read_data('data/test.txt')
    
    clf = train_MLP_classifier(X_train, Y_train)
    
    score = report_clf_stats(clf, X_test, Y_test)
    
    visualize(clf, X_test, Y_test)

if __name__ == "__main__":
    main()
```
