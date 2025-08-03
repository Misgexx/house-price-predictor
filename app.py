import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train.csv")

# Train model
features = ["BedroomAbvGr", "FullBath", "YearBuilt"]
X = df[features]
y = df["SalePrice"]

model = LinearRegression()
model.fit(X, y)

# App UI
st.title("🏠 House Price Predictor")

beds = st.number_input("Number of bedrooms", min_value=0, step=1)
baths = st.number_input("Number of bathrooms", min_value=0, step=1)
year = st.number_input("Built after year", min_value=1800, max_value=2050, step=1)

if st.button("Predict Price"):
    # Filter dataset
    filtered = df[
        (df["BedroomAbvGr"] == beds) &
        (df["FullBath"] == baths) &
        (df["YearBuilt"] >= year)
    ]

    if len(filtered) > 0:
        avg_price = filtered["SalePrice"].mean()
        min_price = filtered["SalePrice"].min()
        max_price = filtered["SalePrice"].max()

        st.success(f"✅ Found {len(filtered)} similar homes.")
        st.write(f"📉 Minimum Price: ${round(min_price, 2)}")
        st.write(f"📈 Maximum Price: ${round(max_price, 2)}")
        st.write(f"📊 Average Price: ${round(avg_price, 2)}")
    else:
        st.warning("⚠️ No exact matches found.")

    # ML prediction
    user_input = pd.DataFrame([{
        "BedroomAbvGr": beds,
        "FullBath": baths,
        "YearBuilt": year
    }])
    predicted_price = model.predict(user_input)[0]
    st.subheader(f"🤖 ML Predicted Price: ${round(predicted_price, 2)}")
