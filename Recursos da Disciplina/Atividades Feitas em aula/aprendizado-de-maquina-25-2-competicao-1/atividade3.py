# ======================================
# SOLUÇÃO SIMPLES E EFETIVA - SEM OVERENGINEERING
# ======================================

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuração para reproduzibilidade
np.random.seed(42)

print("=== CARREGANDO DADOS ===")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Remover coluna de índice
for df in [train_df, test_df]:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

# Preparar dados
X = train_df.drop(columns=["class"])
y = train_df["class"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("=== ANÁLISE RÁPIDA ===")
print("Distribuição classes:", pd.Series(y_encoded).value_counts().to_dict())

# 1. USAR APENAS AS 2 FEATURES MAIS IMPORTANTES
print("=== FEATURES SIMPLES ===")
selected_features = ['degree_spondylolisthesis', 'sacral_slope']
X_selected = X[selected_features]
test_selected = test_df[selected_features]

# 2. PRÉ-PROCESSAMENTO MÍNIMO
print("=== PRÉ-PROCESSAMENTO MÍNIMO ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
test_scaled = scaler.transform(test_selected)

# 3. MODELOS SIMPLES
print("=== MODELOS SIMPLES ===")

# KNN - focado em Spondylolisthesis
knn = KNeighborsClassifier(
    n_neighbors=15,
    weights='distance',
    metric='manhattan'
)

# Naive Bayes - bom para distribuição geral
nb = GaussianNB(var_smoothing=1e-6)

# Decision Tree - conservador
dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=12,
    min_samples_leaf=5,
    random_state=42
)

# 4. VALIDAÇÃO CRUZADA
print("=== VALIDAÇÃO ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {'KNN': knn, 'NaiveBayes': nb, 'DecisionTree': dt}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")

# 5. ENSEMBLE SIMPLES
print("=== ENSEMBLE SIMPLES ===")
ensemble = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('nb', nb),
        ('dt', dt)
    ],
    voting='soft',
    weights=[0.5, 0.3, 0.2]  # Foco no KNN que é melhor
)

ensemble_scores = cross_val_score(ensemble, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"Ensemble: {ensemble_scores.mean():.4f}")

# 6. TREINAMENTO E PREVISÃO SIMPLES
print("=== PREVISÃO SIMPLES ===")
ensemble.fit(X_scaled, y_encoded)
test_predictions = ensemble.predict(test_scaled)
test_labels = label_encoder.inverse_transform(test_predictions)

# 7. SUBMISSÃO DIRETA
print("=== SUBMISSÃO DIRETA ===")
submission = sample_submission.copy()
submission['class'] = test_labels

# Verificar distribuição
final_counts = pd.Series(test_predictions).value_counts()
print("Distribuição das previsões:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name}: {final_counts.get(i, 0)}")

submission.to_csv("submission.csv", index=False)
print("✅ Submission criada com sucesso!")

# 8. VALIDAÇÃO FINAL
print("=== VALIDAÇÃO FINAL ===")
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
ensemble.fit(X_train, y_train)
val_pred = ensemble.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)

print(f"Acurácia validação: {val_accuracy:.4f}")
print(f"Acurácia CV: {ensemble_scores.mean():.4f}")

if val_accuracy > 0.80:
    print("🔥 CHANCE DE 81%+ NO KAGGLE!")
else:
    print("⚠️  Execute novamente")