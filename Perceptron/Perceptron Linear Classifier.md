퍼셉트론 선형 분류기
뉴런
사람의 신경계
신경망
지능
퍼셉트론
딥러닝
인공 신경망
인공지능
논리 회로의 정의
03 퍼셉트론 선형 분류기
일정한 논리 연산에 의해 출력을 얻는 회로를 의미
AND gate
03 퍼셉트론 선형 분류기
A
B
C
AND gate
A/B C
𝟎/𝟎 0
𝟏/𝟎 0
𝟎/𝟏 0
𝟏/𝟏 1
퍼셉트론과 논리 회로 – AND gate
03 퍼셉트론 선형 분류기
Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛(1 ∗ 𝐴 + 1 ∗ 𝐵 − 1.5)
OR gate
03 퍼셉트론 선형 분류기
A
B
C
OR gate A/B C
𝟎/𝟎 0
𝟏/𝟎 1
𝟎/𝟏 1
𝟏/𝟏 1
퍼셉트론과 논리 회로 – OR gate
03 퍼셉트론 선형 분류기
Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛(1 ∗ 𝐴 + 1 ∗ 𝐵 − 0.5)
NAND gate
03 퍼셉트론 선형 분류기
NAND(NOT-AND) gate
A/B C
𝟎/𝟎 1
𝟏/𝟎 1
𝟎/𝟏 1
𝟏/𝟏 0
A
B
C
퍼셉트론과 논리 회로 – NAND gate
03 퍼셉트론 선형 분류기
Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛((−1) ∗ 𝐴 + (−1) ∗ 𝐵 + 1.5)
NOR gate
03 퍼셉트론 선형 분류기
A
B
C
NOR(NOT-OR) gate
A/B C
𝟎/𝟎 1
𝟏/𝟎 0
𝟎/𝟏 0
𝟏/𝟏 0
퍼셉트론과 논리 회로 – NOR gate
03 퍼셉트론 선형 분류기
Ex) 𝐶 = 𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛((−1) ∗ 𝐴 + (−1) ∗ 𝐵 + 0.5)
선형 분류를 위한 퍼셉트론 : 단층 퍼셉트론(Single Layer Perceptron)
03 퍼셉트론 선형 분류기
Input Layer Output Layer
𝑥1
Σ
𝑤1
𝑤2
𝑏𝑖𝑎𝑠
𝑥1
𝑥2 𝑦
𝑤0
03 퍼셉트론 선형 분류기
입력층(Input Layer)
외부로부터 데이터를 입력 받는 신경망 입구의 Layer
Input Layer Output Layer
𝑥1
Σ
𝑤1
𝑤2
𝑏𝑖𝑎𝑠
𝑥1
𝑥2 𝑦
𝑤0
03 퍼셉트론 선형 분류기
출력층(Output Layer)
모델의 최종 연산 결과를 내보내는 신경망 출구의 Layer
Input Layer Output Layer
𝑥1
Σ
𝑤1
𝑤2
𝑏𝑖𝑎𝑠
𝑥1
𝑥2 𝑦
𝑤0
퍼셉트론를 활용한 선형 분류기
03 퍼셉트론 선형 분류기
0, 1 데이터를 계산하던
퍼셉트론 논리 회로에서 확장
선형 분류기로써 데이터 분류 가능
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
