import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Streamlit Page Config ---
st.set_page_config(page_title="Iris Linear Regression", page_icon="🌸", layout="centered")

# --- Title and Description ---
st.title("Linear Regression")
st.write("""
This simple app demonstrates how to use **Linear Regression** to predict one flower feature 
based on another using the **Iris dataset**.
""")

# --- Step 1: Load Dataset ---
df = pd.read_csv('Iris.csv')

with st.expander('This Is The Iris Dataset'):
    st.dataframe(df)

# --- Step 2: Choose Feature and Target ---
st.subheader("2️⃣ Select Variables for Regression")

x_feature = st.selectbox("Select the independent variable (X):", df.columns[:-1], index=2)
y_feature = st.selectbox("Select the dependent variable (Y):", df.columns[:-1], index=3)

# x_feature = st.selectbox("🧩 Pick what the computer will use to guess (X):", iris.columns[:-1])

# y_feature = st.selectbox("🎯 Pick what the computer should guess (Y):", iris.columns[:-1])



# --- Step 3: Train the Model ---
X = df[[x_feature]]
y = df[y_feature]

model = LinearRegression()
model.fit(X, y)

# --- Step 4: Display Model Results ---
st.subheader("3️⃣ Model Results")
st.write(f"**Equation:**  {y_feature} = {model.coef_[0]:.2f} × {x_feature} + {model.intercept_:.2f}")

# --- Step 5: Make Prediction ---
st.subheader("4️⃣ Try Your Own Prediction")

x_value = st.slider(f"Select a value for {x_feature}:", float(X.min()), float(X.max()), float(X.mean()))
predicted_y = model.predict([[x_value]])[0]
st.write(f"🔮 Predicted {y_feature}: **{predicted_y:.2f}**")

# --- Step 6: Plot the Regression Line ---
st.subheader("5️⃣ Visualization")

fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X, model.predict(X), color='red', label='Best Fit Line')
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.legend()
st.pyplot(fig)

with st.sidebar:
    st.subheader('Lotus-Gold')

# --- Footer ---
st.markdown("---")
st.caption("Copyright: Lotus-Gold Consulting 2025")



