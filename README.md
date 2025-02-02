# ML-2Team

# 0. Introduction Team (팀 소개)

### 🎥팀명 : 무빙

- **"movie + bing = mobing"**  영화계의 **Bing**(Microsoft Bing Search)
- 프로젝트 기간: 2025.01.22 ~ 02.03
<table align=center>
  <tbody>
    <tr>
    <br>
      <td align=center><b>김영서</b></td>
      <td align=center><b>유지은</b></td>
      <td align=center><b>전성원</b></td>
      <td align=center><b>허정윤</b></td>
    </tr>
    <tr>
      <td align="center">
          <img src="https://github.com/user-attachments/assets/cdc693b5-b048-4f54-a5bd-2b59c13ca869"  width="200px;" alt="김영서"/>
      <td align="center">
          <img src="https://github.com/user-attachments/assets/1f1060b9-993e-4bb4-99c7-bfb28a28dc5f" width="200px;" alt="유지은"/>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/afb0ea10-9ded-4904-872e-e0671aadbeb8" width="200px;" alt="전성원"/>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/1b530781-7ce5-4eeb-b95b-ea973c145739" width="200px;" alt="허정윤"/>
      </td>
    </tr>
    <tr>
      <td><a href="https://github.com/youngseo98"><div align=center>@youngseo98</div></a></td>
      <td><a href="https://github.com/yujitaeng"><div align=center>@yujitaeng</div></a></td>
      <td><a href="https://github.com/Hack012"><div align=center>@Hack012</div></a></td>
      <td><a href="https://github.com/devunis"><div align=center>@devunis</div></a></td>
    </tr>
  </tbody>
</table>

# 1. Introduction Project (프로젝트 개요)

### 🎥프로젝트 명

**무빙**: 데이터 기반 영화 제작 전략 분석 및 흥행 예측

### 🎥프로젝트 소개

최근 20년간의 영화 데이터를 종합적으로 분석하여 영화 시장의 트렌드와 성공 요인을 도출하고, 이를 통해 영화 제작사들이 효과적인 의사결정을 할 수 있도록 지원하는 데이터 분석 및 AI 예측 프로젝트입니다. 마케팅, 데이터 분석, 홍보, 스토리 개발 등 영화 제작의 핵심 분야별 인사이트를 제공하고 흥행 성적을 예측하여 실질적인 비즈니스 가치를 창출하고자 합니다.

### 🎥프로젝트 필요성(배경)

- 배경:
    - 영화 산업의 투자 리스크 증가와 관객 취향의 다변화
    - 데이터 기반 의사결정의 중요성 증대
    - 제작비 상승으로 인한 ROI 관리의 필요성
- 목표:
    - 객관적 데이터 분석을 통한 영화 시장 트렌드 파악
    - 장르별, 예산별 ROI 분석을 통한 최적 투자 전략 도출
    - 관객 선호도 분석을 통한 성공적인 콘텐츠 기획 가이드라인 제시
- 프로젝트에 사용된 데이터 출처:
    - 출처: IMDB Movies Dataset (1960-2023)
    - 링크:  https://www.kaggle.com/datasets/raedaddala/imdb-movies-from-1960-to-2023
    - 분석 범위: 2005-2024년 (최근 20년) 데이터
    - 주요 분석 항목:
        - 장르별 흥행 추이
        - 예산 대비 수익률
        - 관객 평점과 흥행의 상관관계

# 2. Data Pre-Processing (데이터 전처리)

>EDA에서 전처리를 진행한 데이터 사용 ([eda_2team](https://github.com/Encore-SKN-9/eda_2team?tab=readme-ov-file#2-data-pre-processing))
### 삭제하거나 사용하지 않은 Feature List
  - 국가, 촬영지, 언어, 수상 정보, 평점 정보, 북미/주말 수익 Feature 삭제
  - genres Feature의 경우 Category Feature로 대체
  - 영화 등급, 개봉년도, 상영시간 Feature 삭제
### 학습한 Feature List
  - **budget** (예산)
  - **directors** (감독)
  - **writers** (작가)
  - **stars** (배우)
  - **production_companies** (제작사)
  - **Category** (카테고리 (큰 장르))
### Encoding
  - 감독, 작가, 배우, 제작사의 경우 콤마를 구분으로 데이터가 존재함 ex) “봉준호, 하정우”
  - 콤마를 포함하여 학습을 진행할 경우 학습의 정확도(흥행과의 상관관계) 가 떨어질 수 있음
  - 콤마로 구분된 값 중 첫번째 값만 채택하도록 전처리 진행
        
    ```py
    df['directors'] = df['directors'].apply(lambda x: x.split(',')[0])
    ```
        
  - 문자열로 된 컬럼(**감독, 작가, 배우, 제작사, 카테고리**)을 **Label Encoding** 진행
### 예측을 위한 Profit, Hit 컬럼 생성
  - Profit : 수익 / 예산 * 100
  - Hit(흥행여부) : Profit ≥ 100 (True/False Boolean List)
### Target Feature
  - 위에서 생성한 **Hit** 컬럼

# 3. Using Model and Performances
### 모델별 성능
| 모델 | 개선 전 정확도 | 개선 후 정확도 | 흥행(1) F1-score 개선 |
|------|-------------|-------------|-----------|
| **LogisticRegression** | **72.29%** | **54.21%** | **0.00 → 0.42 (개선됨)** |
| **SVM(SVC)** | **73.27%** | - | - |
| **DecisionTreeClassifier** | **75%** | - | - |
| **KNeighborsClassifier** | **77.24%** | **77.94%** | - |
| **RandomForest** | **79%** | - | 0.52 |
| **XGBoost** | **80.37%** | **77.62%** | **0.59 → 0.62 (개선됨)** |
| **LGBMClassifier** | **81.91%** | **82.66%** | **0.54 → 0.62 (개선됨)** |
  - 사용한 분류 모델 중 LGBMClassifier 모델이 가장 높은 성능을 도출
    - 하이퍼 파라메터 튜닝을 통하여 약83%의 정확도 달성 (0.75%의 성능 향상)
  - LGBMClassifier의 성능이 가장 높았던 이유
    - 데이터가 **비선형적 관계를 포함** → 트리 기반 모델이 유리
    - LGBM은 **원-핫 인코딩 없이도 범주형 변수 처리 가능**
- GridSearchCV 방식을 통하여 최적의 하이퍼 파라메터 도출
   ```python
   param_grid = {
    'num_leaves': [7, 15, 31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
  }
   # Best parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'num_leaves': 15}
   ```
## 시각화 부분
  ### Feature 별 중요도 
  ![output](https://github.com/user-attachments/assets/fda3bf24-5f89-4826-9751-6306a30cb6ce)
  ### 혼동행렬과 평가 지표
  ![cm](https://github.com/user-attachments/assets/47a644b0-624a-45ff-a2ec-3822313c35d3)
  ```
                precision    recall  f1-score   support

             0       0.84      0.93      0.89      1568
             1       0.75      0.53      0.62       572

      accuracy                           0.83      2140
     macro avg       0.80      0.73      0.75      2140
  weighted avg       0.82      0.83      0.82      2140
  ```
  ##### 평가지표 분석
  - 1 클래스 (흥행 성공) 부분에서 Recall(재현율)이 낮아 데이터 증강 혹은 가중치 조정을 고려해야함
    - Precision (정밀도): 0.75
      - 모델이 예측한 1 클래스 중에서 75%가 실제로 1 클래스임.
    - Recall (재현율): 0.53
      - 실제 1 클래스 중에서 53%만 모델이 1로 정확히 예측함.
    - F1-Score: 0.62
      - Precision과 Recall의 조화 평균으로, 1 클래스의 예측 성능이 다소 낮음을 보여줌.
  ### ROC 커브 → 이진 분류의 성능을 나타낸 지표 (True Positive와 False Positive의 비율)
  ![roc](https://github.com/user-attachments/assets/2f69ec36-367c-4311-b719-a8f0bdeb3a29)
  - AUC 가 1에 가까울수록 좋은 성능
  - 0.7 이상의 경우 쓸만한 성능 → 0.85

# 4. Predict Results (실제 예측 결과)
### 데이터 셋 중 무작위 데이터를 골라서 예측과 예측 확률을 표시
```
--- Prediction Result ---
--------------------------------------------------
Title:  470. Hit Man
--------------------------------------------------
Data 1: Not Hit (4.72%)
Actual Value: Not Hit
--------------------------------------------------
Title:  183. A Hidden Life
--------------------------------------------------
Data 2: Not Hit (8.80%)
Actual Value: Not Hit
--------------------------------------------------
Title:  1. Avatar
--------------------------------------------------
Data 3: Hit (94.46%)
Actual Value: Hit
--------------------------------------------------
Title:  420. Appropriate Behavior
--------------------------------------------------
Data 4: Not Hit (8.00%)
Actual Value: Not Hit
--------------------------------------------------
Title:  388. Before the Sunset
--------------------------------------------------
Data 5: Not Hit (5.45%)
Actual Value: Not Hit

--- Prediction Result ---
--------------------------------------------------
Title:  56. Jupiter Ascending
--------------------------------------------------
Data 1: Hit (88.06%)
Actual Value: Hit
--------------------------------------------------
Title:  203. Frank and Ava
--------------------------------------------------
Data 2: Not Hit (18.20%)
Actual Value: Not Hit
--------------------------------------------------
Title:  527. Senior Year
--------------------------------------------------
Data 3: Not Hit (6.33%)
Actual Value: Not Hit
--------------------------------------------------
Title:  26. The Mummy: Tomb of the Dragon Emperor
--------------------------------------------------
Data 4: Hit (88.62%)
Actual Value: Hit
--------------------------------------------------
Title:  79. Bonhoeffer: Pastor. Spy. Assassin.
--------------------------------------------------
Data 5: Not Hit (2.97%)
Actual Value: Not Hit
```
# 5. Expectations (기대 효과)
  - 제작사에게 영화 제작 관련 가이드 제공
  - 영화 제작간 영화 흥행 가능성에 대한 근거 자료
  - 투자자들에게 성공률 높은 영화에 투자 전략 제공
  - 리서치 업체에서 새로 개봉한 영화의 흥행 가능성에 대한 지표 제공
