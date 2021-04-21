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


𝐴
𝐵
Confidential all right reserved
비선형적인 문제
04
하나의 선으로 분류할 수 없는 문제의 등장
04 비선형적인 문제
vs
𝐴
𝐵
𝐴
𝐵
?
비선형적 논리 게이트, XOR gate
04 비선형적인 문제
C
A
B
XOR gate A/B C
𝟎/𝟎 0
𝟏/𝟎 1
𝟎/𝟏 1
𝟏/𝟏 0
단층 퍼셉트론으로는 해결 불가능한 XOR gate
04 비선형적인 문제
04 비선형적인 문제
1958
1969년 첫 번째 AI 겨울
1969
1986
1990 2020
비선형적 접근 방법의 필요성
04 비선형적인 문제
Confidential all right reserved
다층 퍼셉트론
05
05 다층 퍼셉트론
1958
1986년 첫 번째 빙하기의 끝
1969
1986
1990 2020
비선형적인 문제 해결
05 다층 퍼셉트론
단층 퍼셉트론은 입력층과 출력층만 존재
Input Layer Output Layer
𝑥1
Σ
𝑤1
𝑤2
𝑏𝑖𝑎𝑠
𝑥1
𝑥2 𝑦
𝑤0
단층 퍼셉트론을 여러 층으로 쌓아보기
05 다층 퍼셉트론
Input Layer Output Layer
𝑥1
Σ
𝑤1
𝑤2
𝑏𝑖𝑎𝑠
?
𝑥1
𝑥2
𝑤0
05 다층 퍼셉트론
XOR 연산은 하나의 레이어를 사용하여 표현하는 것은 불가능
하지만, NAND와 OR 연산을 함께 사용할 시 표현 가능
XOR gate 예시
Input Layer Hidden Layer
𝑥1
Σ
𝑥1
𝑥2
Output Layer
06 다층 퍼셉트론
다층 퍼셉트론(Multi Layer Perceptron)
이렇게 단층 퍼셉트론을 여러 개 쌓은 것을
다층 퍼셉트론(Multi Layer Perceptron)이라고 부름
Input Layer Hidden Layer
𝑥1
Σ
𝑥1
𝑥2
Output Layer
05 다층 퍼셉트론
히든층(Hidden Layer)
입력층과 출력층 사이의 모든 Layer
Input Layer Hidden Layer
𝑥1
Σ
𝑥1
𝑥2
Output Layer
05 다층 퍼셉트론
히든층의 개수와 딥러닝
히든층이 3층 이상일 시
깊은 신경망이라는 의미의 Deep Learning 단어 사용
𝑥1
𝑥𝑛
…
Input Layer
…
1 Hidden Layer
…
N Hidden Layer Input Layer
… …
…
05 다층 퍼셉트론
다층 퍼셉트론이 결정할 수 있는 영역
1 Hidden Layer 2 Hidden Layers N Hidden Layers
