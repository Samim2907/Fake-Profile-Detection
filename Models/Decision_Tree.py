import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"E:\Projects\ML\Fake Profile Detection\complete_dataset.csv")

df["posts_per_follower"] = df["post_count"] / (df["followers_count"] + 1)
df["posts_per_following"] = df["post_count"] / (df["following_count"] + 1)
df["engagement_proxy"] = df["followers_count"] / (df["post_count"] + 1)

df["low_followers_flag"] = (df["followers_count"] < 50).astype(int)
df["high_following_flag"] = (df["following_count"] > 1000).astype(int)
df["suspicious_ratio_flag"] = (df["follower_following_ratio"] < 0.1).astype(int)

df["followers_count"] = np.log1p(df["followers_count"])
df["following_count"] = np.log1p(df["following_count"])
df["post_count"] = np.log1p(df["post_count"])

y = df["label"]
X = df.drop("label", axis=1)

X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Accuracy: 0.9002800248911015