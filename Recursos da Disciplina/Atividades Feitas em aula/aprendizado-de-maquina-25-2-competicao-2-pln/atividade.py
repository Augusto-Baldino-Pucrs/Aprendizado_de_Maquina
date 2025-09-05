# atividade_knn_nb_dt.py
# KNN, Naive Bayes, Decision Tree — com TF-IDF melhorado e vetorização dentro do pipeline

import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1) Carregar dados
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

# 2) Limpeza de texto
def clean_text(text):
    if pd.isna(text):
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_review'] = train_df['review'].astype(str).apply(clean_text)
test_df['clean_review']  = test_df['review'].astype(str).apply(clean_text)

X_text = train_df['clean_review'].values
y      = train_df['label'].values
X_test_text = test_df['clean_review'].values

# 3) Vetorização combinada (word + char)
word_vect = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    max_features=15000,
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)
char_vect = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    max_features=5000,
    min_df=2,
    sublinear_tf=True
)
union_vect = FeatureUnion([('word', word_vect), ('char', char_vect)])

# 4) Modelos e grids
pipe_nb = Pipeline([
    ('vect', union_vect),
    ('clf',  ComplementNB())
])
grid_nb = {
    'clf__alpha': [0.1, 0.3, 0.5, 1.0]
}

pipe_knn = Pipeline([
    ('vect', union_vect),
    ('clf',  KNeighborsClassifier())
])
grid_knn = {
    'clf__n_neighbors': [3, 5, 7],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan']
}

pipe_dt = Pipeline([
    ('vect', union_vect),
    ('clf',  DecisionTreeClassifier(random_state=42))
])
grid_dt = {
    'clf__max_depth': [5, 10, 15, None],
    'clf__min_samples_split': [2, 5, 10]
}

# 5) Função de treino e avaliação
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tune_and_eval(name, pipe, params):
    print(f"\n=== {name} ===")
    gs = GridSearchCV(
        pipe, params, cv=skf, scoring='accuracy',
        n_jobs=-1, verbose=1, refit=True
    )
    gs.fit(X_text, y)
    print("Best params:", gs.best_params_)
    print("Best CV score:", gs.best_score_)

    # Hold-out
    X_tr, X_val, y_tr, y_val = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
    best_model = gs.best_estimator_
    best_model.fit(X_tr, y_tr)
    preds = best_model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print("Hold-out accuracy:", acc)
    print(classification_report(y_val, preds))
    return gs, acc

gs_nb,  acc_nb  = tune_and_eval("ComplementNB", pipe_nb,  grid_nb)
gs_knn, acc_knn = tune_and_eval("KNN",          pipe_knn, grid_knn)
gs_dt,  acc_dt  = tune_and_eval("DecisionTree", pipe_dt,  grid_dt)

# 6) Escolher melhor e treinar no dataset completo
scores = {'NB': acc_nb, 'KNN': acc_knn, 'DT': acc_dt}
best_key = max(scores, key=scores.get)
best_gs = {'NB': gs_nb, 'KNN': gs_knn, 'DT': gs_dt}[best_key]
best_model = best_gs.best_estimator_

print("\n>>> Melhor modelo:", best_key, "com acc =", scores[best_key])
best_model.fit(X_text, y)

# 7) Predição no teste
test_pred = best_model.predict(X_test_text)
submission = pd.DataFrame({'id': test_df['id'], 'label': test_pred})
submission.to_csv('submission.csv', index=False)
print("\nSubmission salvo em submission.csv")
