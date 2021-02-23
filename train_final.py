### src/train.py + config.py를 사용

## 학습에 사용할 코드
import os
import argparse

import model_dispatcher
import config

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree


def run(fold, model):
    # 폴드가 정의되어있는 학습 데이터를 불러온다.
    df = pd.read_csv(config.Training_File)

    # 학습 데이터는 kfold 열의 값이 제공된 fold와 다른 샘플들이다.
    # 인덱스를 리셋하였음을 유의하라
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # 검증 데이터는 kfold 열의 값이 제공된 fold와 같은 샘플들이다.
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 타겟 열을 제거하고 남은 피쳐를 values를 통해 numpy 행렬로 변환한다.
    # 타겟 변수는 데이터 프레임의 label 열이다.
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # 검증 데이터도 동일하게 적용한다.
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values

    # sklearn의 의사결정트리 모델을 초기화한다.
    clf = model_dispatcher.models[model]

    # 모델을 학습 데이터로 학습한다.
    clf.fit(x_train, y_train)

    # 검증 데이터의 예측 값을 생성한다.
    preds = clf.predict(x_valid)

    # 정확도를 계산하여 출력
    acc = metrics.accuracy_score(y_valid, preds)
    print(f'Fold = {fold}, Accuracy={acc}')

    # 모델을 저장한다.
    joblib.dump(clf, os.path.join(config.Model_Output, f'../models/dt_{fold}.bin'))


if __name__ == '__main__':
    # argparse의 ArgumentParser()
    parser = argparse.ArgumentParser()

    # 필요한 입력 변수와 타입을 추가한다. 현재는 폴드 밖에 없다.
    parser.add_argument(
        '--fold',
        type=int
    )
    parser.add_argument(
        '--model',
        type=str
    )

    # 입력 변수를 콘솔로부터 읽어 들인다.
    args = parser.parse_args()

    # 콘솔로 읽어 들인 폴드 값에 대해 코드를 실행한다.
    run(fold=args.fold,
        model=args.model)