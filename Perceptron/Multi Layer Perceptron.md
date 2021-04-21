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
