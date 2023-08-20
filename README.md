# Classification of women’s body types

# 프로젝트 개요

- **기간** : 2023.03.10 ~ 2023.03.31
- **인원 구성** : 3명
- **주요업무 및 상세역할**
    - 여성의 일자 체형 이미지 **크롤링**
    - **데이터 레이블링** 및 Resize(224*224), Padding, Crop, Gray-Scaling, Zero-Centering, Augmentation(horizontal flip, rotation range, height shift range)를 적용하여 **데이터 전처리**
    - node num, hidden layer, learning rate, dropout, early stopping, Reduce LR On Platue를 적용해보며 ResNet **모델 핸들링**
- **사용언어 및 개발환경** : Google colab Pro+, Python, Selenium, BeautifulSoup, Numpy, Pandas, Matplotlib, OpenCV

## 문제 정의

<img width="779" alt="스크린샷 2023-07-27 오후 1 02 48" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/b05fb6cc-4ca7-4154-9da9-fb3307ee068c">

- 옷을 구매하려는 패션과 거리가 먼 소비자들은 자신에게 어떤 옷이 어울리는지, 자신의 체형을 보완해 줄 수 있는 옷이 무엇인지에 대해 알고 싶어하는 니즈가 늘어나고 있어 위 사진과 같이 유튜브에서도 체형별 스타일링 추천 영상 등의 업로드가 늘어나는 추세이다.

<img width="761" alt="스크린샷 2023-07-27 오후 1 03 09" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/117523aa-0001-46ad-9640-510711d88261">

- 두 가지의 패션 중에서 더 나은 패션을 판단하고자 한다면, 대다수의 사람들은 오른쪽 사진의 패션이 더 나은 패션이라고 답할 것이다. 즉, 사진을 통해 스타일링을 어떻게 하느냐에 따라 체격이 말라 보이는지, 부해 보이는지 확연한 차이를 느낄 수 있는 것처럼 한 사람의 두 가지 패션을 비교해 보았을 때, 스타일링으로 체형의 단점을 보완할 수 있다는 것을 알 수 있다.

## 해결 방안

### <프로젝트 목적>

- Image Classification을 통해 데이터 전처리, Image Labeling과 Augmentation을 직접 적용해 보며 데이터 핸들링을 경험하고, 체형 사진에 대해 일관성 있는 분류가 가능하도록 모델링 경험

### <프로젝트 내용>

- 팀 프로젝트를 통해 여성 소비자의 체형 사진이 어떤 체형에 해당하는지 분류하여 소비자가 스타일링으로 자신의 체형을 보완할 수 있도록 최적의 스타일링 결정에 도움을 줄 수 있는 체형 정보 제공

## 데이터 설명

- **출처** : 한국인 전신 형상 및 치수 측정 데이터 ( 출처 : AI hub )
- **데이터** : 여자 500명, 남자 500명 (자세, 코디, 방향 별로 인당 1,000장 씩 총 1,000,000장)
    - **6가지 자세** : 차렷, 팔 벌리기, 팔 접기, 앞으로 나란히, 걷기
    - **6가지 코디** : 체형이 잘 드러나는 1가지 코디와 체형이 잘 드러나지 않는 5가지 코디
    - **32가지 촬영 시점** : 8방위로 4가지 촬영 높이의 각도
  
<img width="1062" alt="스크린샷 2023-07-27 오후 1 05 41" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/9fa25bd5-49bb-444d-ad04-49a0c3e1e3a7">


- **치수 데이터** : 키, 목뒤 높이, 엉덩이 높이, 겨드랑이 높이, 허리 높이, 허리 둘레, 골반 둘레, …
  
    <img width="425" alt="스크린샷 2023-07-27 오후 1 06 17" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/2a1296cd-b70c-4969-bdd2-114fbbdd9bea">
    

- **최종 데이터 선정**
    - **성별** : 여자
    - **포즈** : 걷는 포즈를 제외한 5가지 포즈
    - **코디** : 체형이 잘 드러나는 1가지 코디
    - **방위** : 정면과 후면으로 4가지 촬영 높이 각도
    - **AI hub의 최종 Dataset** : 인당 40장으로 총 20,000장

## 데이터 레이블링

- **모래시계형(0)** : 허리가 들어가고, 골반이 넓은 마른 허벅지의 체형
- **일자형(1)** :  허리와 골반의 차이가 크지 않은 체형
- **삼각형(2)** : 허리가 얇지만 상대적으로 복부나 허벅지에 살이 많은 체형
- **원형(3)** : 허리가 얇지 않고 상대적으로 복부에 살이 많은 체형
  
<img width="1297" alt="스크린샷 2023-07-27 오후 1 07 56" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/eaa37b42-2f23-4ec3-891a-8d0ef050865a">

- **Label 분포**

    <img width="270" alt="스크린샷 2023-07-27 오후 1 08 18" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/92b47903-6c29-4c1d-8ad8-f54e0c752d9e">

    - 직접 Labeling한 결과로, 데이터가 불균형적으로 분포되어 있기에 모래시계형과 일자형, 원형의 체형 사진을 크롤링하여 Dataset의 Label 균형을 맞추어 모델 성능에 있어서 과적합 방지

## 데이터 전처리

- **Crop** : Overfit의 경우에 Underfit 방향으로 갈 수 있도록 사진의 일부를 자르는 Overfit 해소 전략이지만, 사진의 특성상 배경의 비중 차지가 커서 Underfit의 여지가 있을 것이라 판단되어 사람의 체형을 잘 학습하도록 Crop
- **Resize & Padding** : 고화질이기 때문에 총 200GB로 용량이 부족하여 이를 해결하기 위해 사진의 크기를 1960x2940x3에서 224x224x3으로 조정하여 최종적 데이터셋 용량을 30MB로 축소하고 학습의 난이도를 낮추기 위해 정사각형에 맞게 Padding 적용
- **Gray-Scaling** : Overfit 상태에서 학습을 어렵게 하는 Overfit 해소 전략
- **Zero-Centering** : 각각의 이미지가 가지고 있는 0에서 255 사이의 pixel 값을 Zero-Centering을 통해 모든 이미지의 평균을 0으로 만들어 데이터의 중심을 0으로 맞춰줌 → Machine이 데이터에 따라 학습을 더 잘 하도록 데이터의 복잡도를 낮춰 Normalization 해주는 Underfit 해소 전략

<img width="1260" alt="스크린샷 2023-07-27 오후 1 09 10" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/6ff9f8cb-2bfe-4d00-8d34-142335fd5618">

- **Augmentation** : Overfit 상태에서 학습을 어렵게 하는 Overfit 해소 전략 [Horizontal Flip, Rotation Range, Height Shift Range]
  
<img width="1280" alt="스크린샷 2023-07-27 오후 1 09 37" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/1168cc5f-4cca-4390-86d4-576aeaff2eaa">

## 모델 성능 개선

### AlexNet

- 모델의 기본 구조
  
    <img width="462" alt="스크린샷 2023-07-27 오후 1 10 14" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/f3561e14-4bb0-4695-aadf-72e2030f54b5">

    - Image의 Input Shape은 227x227x3이며, 5개의 Conv Layer에서 이미지의 특징을 추출하고, 3개의 Fully Connected Layer에서 추출된 특징을 바탕으로 분류하는 구조로, 6,000만 개의 파라미터를 가지고 있다.
- 모델의 성능 지표
    
    
    | Layer1 | Layer2 | Layer3 | Train Accuracy | Validation Accuracy | Test Accuracy | Validation Loss |
    | --- | --- | --- | --- | --- | --- | --- |
    | Dropout / 9216 | Dropout / 4096 | 4096 | 0.25 | 0.25 | 0.25 | 1 |
    | 9216 | 4096 |  | 0.99 | 0.82 | 0.83 | 0.74 |
    | Dropout / 9216 | Dropout / 4096 | 4096 | 0.99 | 0.79 | 0.79 | 0.62 |
    - **Hyper Parameter**
        - 성능 지표의 회색줄은 Adam Optimizer를 적용시킨 결과로, AlexNet 모델에 상대적으로 최적화를 더 잘 시켜주는 SGD Optimizer로 선정하였다.
        - 4개의 클래스로 분류해야 하므로 Sparse_Categorical_CrossEntropy Loss를 채택하며, 64 Batch로 Learning Rate는 0.001, iteration은 30 Epoch 학습하였다.
        - Trainable을 False로 하여 분류기만 학습해 여성 체형 데이터셋에 과적합되는 것을 방지하였다.
    - AlexNet 모델의 성능은 Train과 Validation의 Accuracy를 통해 Overfit 상태로 판단할 수 있으며, Hyper Parameter를 고정시킨 상태에서 Layer를 조정했을 때 가장 높았던 Score는 각각 0.99, 0.82, 0.83이다.

### GoogLeNet

- 모델의 기본 구조

    <img width="480" alt="스크린샷 2023-07-27 오후 1 10 36" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/129beb50-478a-4aa9-96d2-6e0110c7e0ad">
  
    - Image의 Input Shape은 224x224x3이며, 3개의 Conv Layer를 거쳐서 9개의 Inception Module에서 여러 필터의 크기로 병렬 연산을 진행하고, 1x1 합성곱으로 차원을 감소시키고 max pooling을 통해 효과적으로 특징을 추출하고, 1개의 Fully Connected Layer를 통해 추출된 특징을 바탕으로 분류한다. 이에 더하여 2개의 Auxiliary Layer로 Vanishing Gradient 문제를 해결하는 구조로, 800만 개의 파라미터를 가지고 있다.
      
- 모델의 성능 지표
    
    
    | Layer1 | Layer2 | Layer3 | Train Accuracy | Validation Accuracy | Test Accuracy | Validation Loss |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1024 | 512 |  | 1.00 | 0.77 | 0.75 | 0.72 |
    | 1024 | 1024 | 512 | 1.00 | 0.75 | 0.75 | 0.9 |
    | 512 | 128 | 32 | 1.00 | 0.87 | 0.87 | 0.47 |
    - **Hyper Parameter**
        - Adam Optimizer와 AlexNet과 동일하게 Sparse_Categorical_CrossEntropy Loss를 채택하였다.
        - 128 Batch로 Learning Ratesms 0.001, iteration은 30 Epoch, Trainable을 False로 하여 분류기만 학습해 여성 체형 데이터셋에 과적합되는 것을 방지하였다.
    - Hyper Parameter를 고정시킨 상태에서 Layer를 조정했을 때 GoogLeNet 모델의 최종 Score는 각각 1.00, 0.87, 0.87 으로, 심한 Overfit의 양상을 나타내고 있다는 것을 알 수 있다.

### VGG 16

- 모델의 기본 구조
  
    <img width="475" alt="스크린샷 2023-07-27 오후 1 11 01" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/917a4089-6e1a-41f5-95c6-f6047c4dbe7b">

    
    - Image의 Input Shape은 224x224x3이며, 13개의 Conv Layer에서 이미지의 특징을 추출하고, 3개의 Fully Connected Layer에서 추출된 특징을 바탕으로 분류하는 구조로, 약 1억 4000만 개의 파라미터를 가지고 있다.
- 모델의 성능 지표 [모델의 Dense와 Learning Rate 조정]
    
    
    | Learning Rate | Layer1 | Layer2 | Train Accuracy | Validation Accuracy | Test Accuracy | Validation Loss |
    | --- | --- | --- | --- | --- | --- | --- |
    | 0.001 | 4096 | 4096 | 0.384 | 0.376 | 0.375 | 0.3762 |
    | 0.00001 | 4096 | 4096 | 0.643 | 0.511 | 0.521 | 2.8223 |
    | 0.00001 | 4096, Dropout | 4096, Dropout | 0.551 | 0.542 | 0.556 | 2.712 |
    | 0.00001 | 2048, Dropout | 1024, Dropout | 0.701 | 0.632 | 0.659 | 0.9342 |
    | 0.00001 | 1024, Dropout | 512, Dropout | 0.676 | 0.620 | 0.616 | 0.9849 |
    - **Hyper Parameter**
        - Adam Optimizer와 Sparse_Categorical_CrossEntropy Loss를 채택하였으며, 8 Batch로 iteration은 50 Epoch, Trainable은 False로 하여 분류기만 학습해 여성 체형 데이터셋에 과적합되는 것을 방지하였다.
    - Hyper Parameter를 고정시킨 상태에서 Learning Rate와 Layer를 조정하여 분류기만 학습시켜 최적의 Dense 구조를 찾아내 모델의 성능을 개선하고자 하였다.
    - 그 결과 VGG 16 모델의 최종 Score는 Train, Valid, Test Accuaracy 각각 0.701, 0.632, 0.659으로, Learning Ratesms 0.00001, Layer는 2048 Node, 1024 Node에 Dropout을 0.3으로 적용한 구조가 가장 높은 성능을 가지는 구조라고 판단하였다.

- 모델의 성능 지표 [ReduceLROnPlateau & 데이터 전처리 : Gray Scaling, Augmentation 적용]
    
    
    | PreProcessing | Epoch / Early Stopping | Train Accuracy | Validation Accuracy | Test Accuracy | Validation Loss |
    | --- | --- | --- | --- | --- | --- |
    |  | 18 / 200 | 0.8278 | 0.7972 | 0.7865 | 0.5712 |
    | Gray Scaling | 21 / 200 | 0.9114 | 0.8012 | 0.8672 | 0.4278 |
    | Gray Scaling, Augmentation | 29 / 200 | 0.9736 | 0.8712 | 0.8571 | 0.2970 |
    - **Hyper Parameter**
        - Adam Optimizer와 Sparse_Categorical_CrossEntropy Loss를 채택하였으며, 8 Batch로 iteration은 50 Epoch, Trainable은 True로 하여 전체 학습을 통해 과적합시킨 후 validation loss를 모니터링하여 patience 20의 Early Stopping과 factor 0.1에 patience 7의 ReduceLROnPlatue을 적용하였다.
    - Layer는 첫 번째 성능 지표를 통해 최종적으로 판단하였으며, 2048 Layer[ Dropout 0.3 ]에 1024 Layer[ Dropout 0.3 ]로 고정하였다.
    - 분류기만 학습한 경우보다 전체 학습을 시킨 경우에 각각의 Accuracy가 0.12~0.16 상승한 것을 통하여 과적합의 상황에서 loss를 모니터링하여 최적화에 도움을 줄 수 있는 기능인 Early Stopping과 ReduceLROnPlatue를 적용해 모델의 성능이 상승한 것을 알 수 있다.
    - 이후에 Overfit을 해소하기 위해 Gray Scaling을 적용한 결과 0.04~0.12 상승하였지만, Validation Accuracy는 Train Accuracy보다 0.11 낮게 나온 것을 통해 심한 Overfit 상태에 있다고 판단하였다.
    - 이에 따라 Horizontal Flip, Rotation Range, Height Shift Range인 데이터 Augmentation을 통해 Overfit을 해소하고자 하였으며, 그 결과 Train과 Validation Accuracy가 각각 0.06~0.07 상승하고, Test Accuracy는 0.01 정도 하락하였다.
    - Train과 Valid Score에 따르면 Overfit 상태에 있지만, Valid Score가 상승하면서 Test와 Valid 간의 Score 균형이 어느정도 맞춰진 것으로, Augmentation 적용 전의 Score에 비하여 상대적으로 일반화된 모델이라고 할 수 있다.

### ResNet

- 모델의 기본 구조

    <img width="561" alt="스크린샷 2023-07-27 오후 1 11 44" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/a2c09c04-ebcf-4d24-9fc6-ea7b60e0cd62">
    
    - Image의 Input Shape은 224x224x3이며, 49개의 Conv Layer에서 이미지의 특징을 추출하고, 1개의 Fully Connected Layer에서 추출된 특징을 바탕으로 분류하는 구조로, 약 700만 개의 파라미터를 가지고 있다.
- 모델의 성능 지표 [모델의 Dense와 Learning Rate 조정 & * : Gray Scaling 적용]
    
    
    | Learning Rate | Layer1 | Layer2 | Layer3 | Train Accuracy | Validation Accuracy | Test Accuracy | Validation Loss |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Zero-Centering(X) & Resnet 50 & 0.0001 | 512 | 256 | 256 | 0.000e+00 | 0 | 0 | 51.95 |
    | ResNet 50 & 0.0001 | 512, Dropout | 512, Dropout | 256 | 0.60 | 0.68 | 0.73 | 0.65 |
    | 0.00001 | 512 | 512, Dropout | 256, Dropout | 0.96 | 0.76 | 0.76 | 0.70 |
    | 0.00001 | 512, Dropout | 256 | 256, Dropout | 0.98 | 0.78 | 0.82 | 0.66 |
    | 0.00001(*) | 512, Dropout | 256 | 256, Dropout | 0.95 | 0.83 | 0.83 | 0.50 |
    - **Hyper Parameter**
        - Adam Optimizer와 Sparse_Categorical_CrossEntropy Loss를 채택하였으며, 128 Batch로 iteration은 15 Epoch, 0.4 Dropout, Trainable은 False로 하여 분류기만 학습해 여성 체형 데이터셋에 과적합되는 것을 방지하였다.
    - Hyper Parameter를 고정시킨 상태에서 Learning Rate와 Layer를 조정하여 분류기만 학습시켜 최적의 Dense 구조를 찾아내 모델의 성능을 개선하고자 하였다.
    - 그 결과 ResNet 모델의 최종 Score는 Train, Valid, Test Accuaracy 각각 0.95, 0.83, 0.83으로, Learning Ratesms 0.00001, Layer는 512 Node, 256 Node, 256 Node에 Dropout을 0.4로 적용한 구조가 가장 높은 성능을 가지는 구조라고 판단하였다.

- 모델의 성능 지표 [ReduceLROnPlateau & 데이터 전처리 : Gray Scaling, Augmentation 적용]
    
    
    | PreProcessing | Epoch / Early Stopping | Train Accuracy | Validation Accuracy | Test Accuracy | Validation Loss |
    | --- | --- | --- | --- | --- | --- |
    | Gray Scaling(x) | 28 / 200 | 0.99 | 0.94 | 0.94 | 0.25 |
    |  | 49 / 100 | 1.00 | 0.97 | 0.97 | 0.21 |
    | 약한 Augmentation | 50 / 100 | 0.99 | 0.96 | 0.97 | 0.10 |
    | 강한 Augmentation | 90 / 100 | 0.99 | 0.97 | 0.97 | 0.10 |
    - **Hyper Parameter**
        - Adam Optimizer와 Sparse_Categorical_CrossEntropy Loss를 채택하였으며, 64 Batch로 iteration은 50 Epoch, Trainable은 True로 하여 전체 학습을 통해 과적합시킨 후 validation loss를 모니터링하여 patience 10의 Early Stopping과 factor 0.1에 patience 5의 ReduceLROnPlatue을 적용하였다.
    - Layer는 첫 번째 성능 지표를 통해 최종적으로 판단하였으며, 512 Layer[ Dropout 0.4 ]에 256 Layer[ Dropout 0.4 ]로 고정하였다.
    - 분류기만 학습한 경우보다 전체 학습을 시킨 경우에 각각의 Accuracy가 0.04~0.09 상승한 것을 통하여 과적합의 상황에서 loss를 모니터링하여 최적화에 도움을 줄 수 있는 기능인 Early Stopping과 ReduceLROnPlatue를 적용해 모델의 성능이 상승한 것을 알 수 있다.
    - 이후에 Overfit을 해소하기 위해 Gray Scaling을 적용한 결과 0.01~0.03 상승하였지만, Validation Accuracy는 Train Accuracy보다 0.03 낮게 나온 것을 통해 Overfit 상태에 있다고 판단하였다.
    - 전체 학습을 시켜 Gray Scaling을 적용하지 않은 결과와 적용한 결과를 첫 번째 줄의 Score와 두 번째 줄의 Score로 볼 수 있으며, 각각의 경우에 따라 Gray Scaling을 적용시킨 Score가 더 높게 나옴에 따라 데이터셋에 Gray Scaling을 적용해 모델의 성능이 개선된 것을 알 수 있다.
    - Horizontal Flip, Rotation Range, Height Shift Range인 약한 데이터 Augmentation을 통해 Overfit을 해소하고자 하였으며, 그 결과 Overfit 상태에서 Train과 Validation, Test Accuracy가 각각 0.99, 0.96, 0.97으로, Loss가 0.11 하락하였다.
    - 강한 Augmentation을 적용시켰을 때 Validation Accuracy가 0.1 정도 상승하면서 Validation과 Test의 Accuracy가 균형을 이루고 있는 것을 볼 수 있다.
    - Train과 Valid Score에 따르면 Overfit 상태에 있지만, Valid Score가 상승하면서 Test와 Valid 간의 Score 균형이 어느정도 맞춰진 것으로, Augmentation 적용 전의 Score에 비하여 상대적으로 일반화된 모델이라고 할 수 있다.
    - 프로젝트의 최종 선정 모델은 **ResNet 101**으로 선정하여, **전체 개방하여 학습**시켰을 때 **Train Score는 0.99, Validation Score는 0.97, Test Score는 0.97, Loss는 0.1**로 확인할 수 있다.

<img width="1013" alt="스크린샷 2023-07-27 오후 1 12 12" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/44af340e-b0b7-462f-87fd-e09e1a404805">

- Train과 Validation Accuracy 간의 약 0.2 Score의 차이를 통해 Overfit 상태라고 판단하였으며, Overfit을 해소하기 위한 전략으로 Gray Scaling과 Augmentation, ReduceLROnPlatue를 적용한 결과 Train과 Validation Accuracy 그래프와 같이 Overfit을 어느 정도 해결한 것으로 볼 수 있다.
- 이에 더하여, Overfit 해소 전략을 적용하기 이전의 Loss는 하락하는 추세에서 불안정하게 튀는 경향이 나타났다면 적용한 후의 Loss는 상대적으로 안정적이게 하락하는 것을 볼 수 있다.

## 개선사항 & 기대효과

- **개선사항**
    - 주관적으로 Labeling을 하게 되면서 기준이 명확하지 않아 모델이 잘못 예측하는 문제가 발생하였으며, 다양한 체형이 존재하지만 제한된 Label의 개수로, 분류한 Label 외의 체형은 잘못 예측하게 되는 문제가 발생함에 따라 Label의 범위를 넓혀 개선
    - AI Hub 데이터셋은 사람의 체형이 고정된 위치에 있어 동일한 위치의 체형 사진 데이터셋을 학습시킨 결과 해당 체형 위치 사진이 아닌 경우에 모델의 추론 성능이 떨어지는 문제가 발생함에 따라 다양한 위치에 존재하는 체형사진을 크롤링하여 개선
- **기대효과** : Application

<img width="1056" alt="스크린샷 2023-07-27 오후 1 12 34" src="https://github.com/Yu-Miri/Classification_of_Womens_Body_Types/assets/121469490/c64cf040-8199-47c6-8e20-73eaf7b0a9b1">

## 참고 문헌

- He, Kaiming, et al. "Deep residual learning for image recognition." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

  

## Installation

### Requirements
~~~
git clone https://github.com/Yu-Miri/Classification_of_Women's_Body_Types.git
pip install tensorflow
~~~

### Preparing the Dataset
https://drive.google.com/drive/folders/189-Ycj7FJf6aC19Q2Yhr1vMTNNpHNKHQ?usp=sharing


### Training
~~~
from train import training
training('./datasets/bodyshape_underwear/train','./datasets/bodyshape_underwear/val','./datasets/bodyshape_underwear/test')
~~~

### Inference
~~~
from inference import inference
inference('./datasets/infer.png')
~~~
