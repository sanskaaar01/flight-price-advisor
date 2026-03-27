import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from model.preprocess import preprocess_data

# ---------- LOAD ----------
df = pd.read_csv("data/dataset.csv")

# ---------- PREPROCESS ----------
df = preprocess_data(df)

# ---------- SPLIT ----------
X = df.drop("Price", axis=1)
y = df["Price"]

# 🔥 Save columns (VERY IMPORTANT)
pickle.dump(X.columns, open("model/columns.pkl", "wb"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- MODEL ----------
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# ---------- PREDICT ----------
y_pred = model.predict(X_test)

# ---------- EVALUATION ----------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

# ---------- SAVE MODEL ----------
pickle.dump(model, open("model/model.pkl", "wb"))

# ---------- FEATURE IMPORTANCE ----------
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop Features:")
print(importance.head(10))