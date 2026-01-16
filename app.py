import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Sleep Quality Predictor",
    page_icon="🌙",
    layout="centered"
)

st.title("🌙 Sleep Quality Predictor")
st.write("Enter details from your **last 3 nights of sleep** to predict tonight’s sleep quality.")

# -----------------------------
# Dummy Dataset (replace with yours if needed)
# -----------------------------
np.random.seed(42)
data = pd.DataFrame({
    "sleep_duration": np.random.uniform(4, 9, 300),
    "sleep_latency": np.random.uniform(5, 60, 300),
    "wake_ups": np.random.randint(0, 5, 300),
    "sleep_quality": np.random.choice([0, 1], 300)  # 0 = poor, 1 = good
})

X = data.drop("sleep_quality", axis=1)
y = data["sleep_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Metrics
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🛌 Last 3 Nights Sleep Data")

def sleep_inputs(night):
    st.sidebar.subheader(f"Night {night}")
    duration = st.sidebar.slider(f"Sleep Duration (hrs) – Night {night}", 3.0, 10.0, 7.0)
    latency = st.sidebar.slider(f"Time to Fall Asleep (mins) – Night {night}", 0, 90, 20)
    wakeups = st.sidebar.slider(f"Wake-ups – Night {night}", 0, 10, 1)
    return duration, latency, wakeups

n1 = sleep_inputs(1)
n2 = sleep_inputs(2)
n3 = sleep_inputs(3)

# Average last 3 nights
avg_features = np.array([
    np.mean([n1[0], n2[0], n3[0]]),
    np.mean([n1[1], n2[1], n3[1]]),
    np.mean([n1[2], n2[2], n3[2]])
]).reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("🔮 Tonight’s Prediction")

if st.button("Predict Sleep Quality"):
    prediction = model.predict(avg_features)[0]
    probability = model.predict_proba(avg_features)[0][prediction]

    if prediction == 1:
        st.success(f"🌟 Good sleep predicted (Confidence: {probability:.2f})")
    else:
        st.warning(f"⚠️ Poor sleep predicted (Confidence: {probability:.2f})")

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
st.subheader("🧩 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.caption("Built with Streamlit • For educational purposes")
