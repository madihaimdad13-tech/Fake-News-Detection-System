import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# 2. Labeling
fake["label"] = 0
true["label"] = 1

# 3. Merge data
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

# 4. Combine text
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 8. Prediction
y_pred = model.predict(X_test_vec)

# 9. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# 11. Save model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")