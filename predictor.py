import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("train.csv")

# Ask for user input
beds = int(input("Enter number of bedrooms: "))
baths = int(input("Enter number of full bathrooms: "))
year = int(input("Enter the earliest year built you're okay with: "))

# Filter dataset
filtered = df[
    (df["BedroomAbvGr"] == beds) &
    (df["FullBath"] == baths) &
    (df["YearBuilt"] >= year)
]

# Show average, min, max if matches exist
if len(filtered) > 0:
    avg_price = filtered["SalePrice"].mean()
    min_price = filtered["SalePrice"].min()
    max_price = filtered["SalePrice"].max()

    print(f"\n✅ Found {len(filtered)} similar homes.")
    print(f"📉 Minimum price: ${round(min_price, 2)}")
    print(f"📈 Maximum price: ${round(max_price, 2)}")
    print(f"📊 Average price: ${round(avg_price, 2)}")
else:
    print("\n⚠️ No exact matches found in the dataset.")

# Train ML model
features = ["BedroomAbvGr", "FullBath", "YearBuilt"]
X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict with user input
user_input = pd.DataFrame([{
    "BedroomAbvGr": beds,
    "FullBath": baths,
    "YearBuilt": year
}])
predicted_price = model.predict(user_input)[0]

print(f"\n🤖 Predicted price using ML model: ${round(predicted_price, 2)}")
