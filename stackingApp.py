import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import streamlit as st

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stacking Classifier Demo", layout="centered")
st.title("📊 Stacking Classifier – Social Network Ads")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("Social_Network_Ads.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Features & Target
# -----------------------------
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# -----------------------------
# Train-test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# -----------------------------
# Models
# -----------------------------
base_models = [
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier(max_depth=3)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

meta_model = LogisticRegression()

classifier = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# -----------------------------
# Train & Predict
# -----------------------------
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# -----------------------------
# Metrics
# -----------------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {acc:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("🧩 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
