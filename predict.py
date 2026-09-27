import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("data/students.csv")

# Prepare the data
X = data[["study_hours"]]
y = data["marks"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Enter study hours
hours = float(input("Enter study hours: "))

# Predict marks
prediction = model.predict([[hours]])

print("Predicted marks:", prediction[0])
