# 프로젝트 내용 정리

### **주제**

- 영화 **장르별 핵심 키워드** 추출
  - 장르별 영화 요소(키워드)의 **중요도 파악** => LDA 스코어로?
  - 한 장르에서 어떤 키워드의 **중요도 파악** => 어떤키워드 스코어 / 모든 키워드스코어 합
  - 장르간 추출된 키워드가 비슷하다면 해당 키워드의 유사의미단어(CTM)들의 차이 제시

- ...



### **코드정리**

1. 데이터 수집.ipynb

2. 데이터 분석 및 전처리.ipynb
3. 모델링.ipynb
   - LSA
   - LDA1
   - LDA2
   - KeyBERT
   - CTM
   - Gensim앙상블
   - Tomotopy앙상블
4. 키워드 결과.ipynb
   - LSA / LDA1 / LDA2 / KeyBERT / CTM / **LSA,LDA,KeyBERT** / Gemsim앙상블 / **Tomotopy앙상블**
5. .ipynb



- 목차 대문자

- 데이터 로드부분 
- Coherence & Perp~ .py로 모듈화, 위치
- 모델들 순서
- 데이터 저장(positive, negative)
- 전체적인 변수네이밍(LDA tomotopy many -> Ensemble LDA (Tomotopy))
- 모델 튜닝



## **0. Contents**

1. 프로젝트 및 수행과정 소개

2. 수행과정
   1. 데이터 소개 및 수집
   2. 데이터 전처리
   3. 토픽 모델링
   4. 결과 분석
3. 셀프 피드백(시행착오들, 개선점들)
4. Reference (1page)
5. QnA



## 1. **프로젝트 및 수행과정 소개**

1. ### **팀원소개 및 역할**

   - 김준호(팀장)

     - 데이터 크롤링 코드 구현, 데이터 수집, (데이터 전처리), (자료조사), (결과분석)

     - 모델 (LDA, LSA, KeyBERT, CTM) 구현

     - 키워드 추출 코드 구현

     - 강석창

       - 데이터 수집
       - 모델 (TextRank)

     - 박재현

       - 데이터 수집, 데이터 전처리 코드 구현
       - 모델 (Ensemble LDA)
       - PPT 발표

     - 김예슬

       - 데이터 수집
       - PPT 제작

       

2. ### **프로젝트 소개**

  - 다양한 분야에서 소비자 리뷰 분석을 통해 마케팅, 제품 개선 등에 이용
    - 관련 연구 사례 캡쳐
  - (소비자 데이터의 중요성 상기 및 영화 분야에 적용)
  - 영화 리뷰데이터로 장르별 핵심 키워드 추출을 통해 B2B 측면에서 ...



3. ### **수행과정**

  - 수행과정 도식화



## 2. 수행과정

### 1) **데이터 수집**

- 네이버 영화 리뷰
  - 캡쳐 (네이버 영화 -> 영화랭킹 -> 평점순 페이지)
  - (다양한 영화 리뷰 사이트 But 리뷰 수와 크롤링 편의성 등 고려해 네이버 영화로 선정)



### 2) 데이터 전처리

1. 데이터 클리닝

   - 결측치 캡쳐

   - 중복리뷰 캡쳐

2. 데이터 전처리

   - Okt, Mecab 사용해 명사만 추출
   - Okt 실행했던 과정
   - Mecab 사용해 일반명사, 고유명사만 추출
   - 명사 추출 후 불용어 제거, 1글자 단어 제거



### 3) 토픽 모델링




#### 1. Topic models

- 문서 집합에서 "토픽"이라는 추상적인 주제를 찾기 위한 통계적 모델 중 하나로, 텍스트 본문의 숨겨진 의미 구조를 발견하기 위해 사용되는 텍스트 마이닝 기법입니다.
- A topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for the discovery of hidden semantic structures in a text body. (paperswithcode)



#### 1) 잠재의미분석 (LSA, Latent Semantic Analysis)

- **DTM(Document Term Matrix)를 만든 후 Truncated SVD를 통해 문서에 숨어있는 의미를 추출**
- 결과 캡쳐
- **쉽고 빠르게 구현 가능한 장점이 있어 대략적인 인사이트를 얻는데 활용**



#### 2) 잠재 디리클레 할당 (LDA, Latent Dirichlet Allocation)

- 주어진 문서들에 대하여 각 문서에 어떤 주제들이 존재하는지에 대한 확률모형
- 문서가 생성되는 과정을 확률적으로 모델링
- (미리 알고 있는 주제별 단어수 분포를 바탕으로, 주어진 문서에서 발견된 단어수 분포를 분석함으로써 해당 문서가 어떤 주제들을 함께 다루고 있을지를 예측)
  특정 토픽에 특정 단어가 나타날 확률
- ![image-20220525104003691](README.assets/image-20220525104003691.png)
- 다시 말해 LDA는 토픽의 단어분포와 문서의 토픽분포의 결합으로 문서 내 단어들이 생성된다고 가정합니다. LDA의 inference는 실제 관찰가능한 문서 내 단어를 가지고 우리가 알고 싶은 토픽의 단어분포, 문서의 토픽분포를 추정하는 과정입니다.

우선 글감 내지 주제를 정해야 합니다. 이후 실제 글을 작성할 때는 어떤 단어를 써야할지 결정합니다. LDA도 마찬가지입니다.
우선 말뭉치로부터 얻은 토픽 분포로부터 토픽을 뽑습니다. 이후 해당 토픽에 해당하는 단어들을 뽑습니다. 이것이 LDA가 가정하는 문서 생성 과정입니다.

이제 반대 방향으로 생각해보겠습니다. 현재 문서에 등장한 단어들은 어떤 토픽에서 뽑힌 단어들일까요? 이건 명시적으로 알기는 어렵습니다. 말뭉치에 등장하는 단어들 각각에 꼬리표가 달려있는 건 아니니까요.

- 토픽의 단어분포와 문서의 토픽분포의 결합확률이 커지도록 해야 한다
- Gemsim과 Sklearn에서 지원
- 문서 생성 가정(model)부분 Inference부분
- 문서들은 여러가지 토픽들로 표현된다
- 각 **토픽**들은 **단어의 분포**이다 (토픽은 beta를 하이퍼파라미터로 갖는 디리클레 분포)
- 각 **문서**는 **corpus-wide topic들의 혼합**이다 => 문서는 토픽의 비중(분포)의 혼합
  => corpus가 동일하다면 즉 문서집합들이 변하지않으면  각 문서의 토픽의 비중은, 토픽을 나타내는 단어들의 분포는 변하지 않는다
  => 전체문서가 (30%, 50%, 20%) 라면
- 각 **단어**들은 **토픽에 의해 생성**된다 -> 어떤 토픽에서 뽑혔는지



#### 3) KeyBERT

- **BERT를 이용해 문서와 단어를 Embedding한 후 유사도를 이용해 키워드 추출하는 방법**
- First, document embeddings are extracted with BERT to get a document-level representation. Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity to find the words/phrases that are the most similar to the document. The most similar words could then be identified as the words that best describe the entire document.
- BERT를 이용해 문서 레벨 (document-level)에서의 주제 (representation)를 파악하도록 하고, N-gram을 위해 단어를 임베딩 합니다. 이후 코사인 유사도를 계산하여 어떤 N-gram 단어 또는 구가 문서와 가장 유사한지 찾아냅니다. 가장 유사한 단어들은 문서를 가장 잘 설명할 수 있는 키워드로 분류됩니다 (



#### 4) 복합토픽모델(CTM, Combined Topic Models)

- 빈도수 기반 문서 벡터화 방식인 Bag of Words와 사전 훈련된 언어 모델의 문서 임베딩 방식인 SBERT를 결합하여 사용하는 복합 토픽 모델



#### 5) Ensemble LDA

- To Extracting Reliable Topics,

- LDA의 중요한 문제는 재현성(reproducibility), 처음에 랜덤으로 초기화 해주기 때문에 다시 실험했을때도 똑같은 결과가 나오
- 그래서 시드를 고정해 놓고 

- Gensim로 구현
  - Gensim 라이브러리에서 지원하는 Ensemble LDA


- Tomotopy로 직접 구현
  - 기존 Gemsim이나 Sklearn같은 경우는 한개의 LDA 모델을 돌리는데 5분정도로 많은 시간소요
  - 




#### 6) Ensemble LDA (with Tomotopy)

- 실행시간에서 Gensim보다 







