import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/students.csv")

X = data[["study_hours"]]
y = data["marks"]

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")

hours = [[6.5]]
prediction = model.predict(hours)

print("Predicted marks:", prediction[0])
