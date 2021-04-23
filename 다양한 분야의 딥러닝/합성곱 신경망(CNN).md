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


