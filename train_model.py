
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# 1. Load the dataset
data = pd.read_csv("students.csv")

# 2. Select Features and Target
X = data[["Study_Hours", "Attendance", "Previous_Score"]]
y = data["Final_Score"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create the model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Completed!")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# 8. Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
