
import joblib
import pandas as pd

# Load the saved model
model_data = joblib.load("models/best_model.joblib")

model = model_data["model"]
class_names = model_data["classes"]

# Example flower measurements in centimetres
flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

# Predict the flower species
prediction = model.predict(flower)[0]

print("Predicted species:", class_names[prediction])
