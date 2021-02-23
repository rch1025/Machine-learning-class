## model_dispatcher.py

## 다양한 종류의 코드를 학습 코드로 파견
from sklearn import tree
from sklearn import ensemble

models = {
    'decision_tree_gini' : tree.DecisionTreeClassifier(
        criterion = 'gini'
    ),
    'decision_tree_enthropy' : tree.DecisionTreeClassifier(
        criterion = 'entropy'
    ),
    'rf' : ensemble.RandomForestClassifier(),
}
