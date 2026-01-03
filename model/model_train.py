import joblib
from sklearn.ensemble import GradientBoostingClassifier
from utils.preprocess import preprocess

X, y, le, scaler, n_features = preprocess("data/raw/clinical_data.csv")

model = GradientBoostingClassifier(n_estimators=150)
model.fit(X, y)

joblib.dump(model, "model/model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
joblib.dump(scaler, "model/scaler.pkl")
