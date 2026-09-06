import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef

FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

df = pd.read_csv("data.csv", header=None, names=FEATURES + ["status"])
df = df.apply(pd.to_numeric, errors="coerce").dropna()

X = df[FEATURES]
y = df["status"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)

pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier())
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

print("========== PARKINSON'S KNN MODEL ==========")
print(f"Dataset rows: {len(df)}")
print(f"10-fold CV accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")
print(f"Holdout accuracy: {accuracy_score(y_test, pred)*100:.2f}%")
print(f"MCC: {matthews_corrcoef(y_test, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

# Final model uses all clean data for the website
final_model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("knn", KNeighborsClassifier())
])
final_model.fit(X, y)

joblib.dump(final_model, "models/parkinsons_knn_pipeline.pkl")

metadata = {
    "features": FEATURES,
    "dataset_rows": int(len(df)),
    "cv_accuracy_mean": float(scores.mean()),
    "cv_accuracy_std": float(scores.std()),
    "holdout_accuracy": float(accuracy_score(y_test, pred)),
    "mcc": float(matthews_corrcoef(y_test, pred)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}

with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved to models/parkinsons_knn_pipeline.pkl")
