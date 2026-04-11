import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import cross_val_score

df = pd.read_csv(rf"E:\Projects\ML\Fake Profile Detection\cleaned_fake_profiles_dataset.csv")

df['profile_pic'] = df['profile_pic'].map(
    {'1': 1, '0': 0, 'True': 1, 'False': 0}
).fillna(0).astype(int)

df["posts_per_follower"]    = df["post_count"] / (df["followers_count"] + 1)
df["posts_per_following"]   = df["post_count"] / (df["following_count"] + 1)
df["engagement_proxy"]      = df["followers_count"] / (df["post_count"] + 1)
df["low_followers_flag"]    = (df["followers_count"] < 50).astype(int)
df["high_following_flag"]   = (df["following_count"] > 1000).astype(int)
df["suspicious_ratio_flag"] = (df["follower_following_ratio"] < 0.1).astype(int)
df["has_bio"]               = (df["bio_length"] > 0).astype(int)
df["no_posts_flag"]         = (df["post_count"] == 0).astype(int)
df["ff_ratio_clipped"]      = df["follower_following_ratio"].clip(upper=100)

df["followers_count"] = np.log1p(df["followers_count"])
df["following_count"] = np.log1p(df["following_count"])
df["post_count"]      = np.log1p(df["post_count"])

y = df["label"]
X = df.drop("label", axis=1).select_dtypes(include=[np.number]).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
]
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500),
    cv=5, n_jobs=-1
)
model.fit(X_train_s, y_train)

# ── Evaluate with 0.70 threshold ─────────────────────────────────────────────
probas = model.predict_proba(X_test_s)
preds  = (probas[:, 1] >= 0.70).astype(int)

print("Stacking")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, target_names=['Real', 'Fake']))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,             r"E:\Projects\ML\Fake Profile Detection\model.pkl")
joblib.dump(scaler,            r"E:\Projects\ML\Fake Profile Detection\scaler.pkl")
joblib.dump(X.columns.tolist(),r"E:\Projects\ML\Fake Profile Detection\features.pkl")

print("Model saved!")

cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
print("CV Scores:  ", cv_scores)
print("CV Mean:    ", cv_scores.mean())
print("CV Std Dev: ", cv_scores.std())
# Accuracy: 0.9698195395146235