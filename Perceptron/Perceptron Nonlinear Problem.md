# Nonlinear Classifier(비선형적인 문제)

- 하나의 선으로 분류할 수 없는 문제의 등장
- XOR 문제의 경우는 지금까지 만든 AND, OR, NAND, NOR gate처럼 선형 분류기 하나로 문제를 해결할 수 없다. 즉 이는 **로지스틱 회귀(Logistic Regression)**로 문제를 해결할 수 없다는 뜻과 같다.
    - **로지스틱 회귀는 범주형 변수를 회귀 예측하는 알고리즘**을 말한다. XOR gate를 포함한 AND, OR, NAND, NOR gate는 0과 1의 입력쌍을 통해 0 또는 1, 즉 두 종류의 변수를 예측한다. 따라서 지금까지 배운 gate 알고리즘은 모두 로지스틱 회귀 알고리즘이다.
- 첫번째 AI 빙하기는 단층 퍼셉트론이 비선형 문제를 풀 수 없기 때문에 찾아왔다.

1. XOR gate

| A/B | C |
|:---:|:---:|
| 𝟎/𝟎 | 0 |
| **𝟏/𝟎** | **1** |
| **𝟎/𝟏** | **1** |
| 𝟏/𝟏 | 0 |

## XOR 게이트 구현 : 선형 회귀로 XOR 게이트를 완벽하게 구현할 수 없다. Accuracy를 75%까지 맞추기

```
import numpy as np

'''
1. XOR_gate 함수를 최대한 완성해보세요.

   Step01. 이전 실습을 참고하여 입력값 x1과 x2를
           Numpy array 형식으로 정의한 후, x1과 x2에
           각각 곱해줄 가중치도 Numpy array 형식으로 
           적절히 설정해주세요.
           
   Step02. XOR_gate를 만족하는 Bias 값을
           적절히 설정해주세요.
           
   Step03. 가중치, 입력값, Bias를 이용하여 
           가중 신호의 총합을 구합니다.
           
   Step04. Step Function 함수를 호출하여 
           XOR_gate 출력값을 반환합니다.
'''

def XOR_gate(x1, x2):
    
    x = np.array([x1, x2])
    
    weight = np.array([0.5, 0.5])
    
    bias = -0.3
    
    y = np.matmul(x, weight) + bias
    
    return Step_Function(y)

'''
2. 설명을 보고 Step Function을 완성합니다.
   앞 실습에서 구현한 함수를 그대로 
   사용할 수 있습니다.

   Step01. 0 미만의 값이 들어오면 0을,
           0 이상의 값이 들어오면 1을
           출력하는 함수를 구현하면 됩니다.
'''

def Step_Function(y):
    
    return 1 if y >= 0 else 0

def main():
    
    # XOR Gate에 넣어줄 Input과 그에 따른 Output
    Input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])        
    Output = np.array([[0], [1], [1], [0]])
    
    # XOR Gate를 만족하는지 출력하여 확인
    print('XOR Gate 출력')
    
    XOR_list = []
    
    for x1, x2 in Input:
        print('Input: ',x1, x2, ' Output: ', XOR_gate(x1, x2))
        XOR_list.append(XOR_gate(x1, x2))
    
    hit = 0
    for i in range(len(Output)):
        if XOR_list[i] == Output[i]:
            hit += 1
    
    acc = float(hit/4)*100
    
    print('Accuracy: %.1lf%%' % (acc))

if __name__ == "__main__":
    main()
```
