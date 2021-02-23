#!/usr/bin/env python
# coding: utf-8

# ## py파일로 변환 해주어야 함

# In[5]:


## mnist.csv 파일을 가지고 mnist_train_folds를 생성한다.
## 각 샘플이 어떤 fold에 속했는지를 알려주는 kfold라는 새로운 행이 있다.

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # 학습 데이터
    df = pd.read_csv('mnist_train.csv')
    
    # kfold라는 새로운 열을 생성하고 -1로 채운다.
    df['kfold'] = -1
    
    # 다음 단계는 데이터의 행을 랜덤하게 섞는 것이다.
    df = df.sample(frac = 1).reset_index(drop = True)
    
    # model selection 모듈의 kfold 클래스를 초기화한다.
    kf = model_selection.KFold(n_splits = 5)
    
    # kfold 열을 폴드 아이디로 설정한다.
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    # 데이터를 kfold 열과 함께 새로운 csv 파일로 저장한다.
    df.to_csv('mnist_train_folds.csv', index = False)

