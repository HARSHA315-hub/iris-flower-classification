
import csv
import io
import joblib
import pandas as pd

from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template, make_response
from sqlalchemy import func
from models import db, Prediction

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///iris_predictions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()



# Load the trained model once when the application starts
model_data = joblib.load("models/best_model.joblib")

model = model_data["model"]
class_names = model_data["classes"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Send measurements as JSON."}), 400

        feature_names = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]

        input_keys = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        # Check that all four measurements are provided
        if not all(key in data for key in input_keys):
            return jsonify({
                "error": "Please provide all four flower measurements."
            }), 400

        # Convert measurements to numbers
        try:
            values = [float(data[key]) for key in input_keys]
        except (TypeError, ValueError):
            return jsonify({
                "error": "All measurements must be numbers."
            }), 400

        # Reject invalid values
        if any(value <= 0 or value > 10 for value in values):
            return jsonify({
                "error": "Measurements must be greater than 0 and at most 10 cm."
            }), 400

        # Create input in the same format used during training
        flower = pd.DataFrame([values], columns=feature_names)

        # Predict the species
        prediction = int(model.predict(flower)[0])
        species = class_names[prediction]

        probabilities = model.predict_proba(flower)[0]

        class_probabilities = {
            class_names[int(class_index)]: round(
                float(probability) * 100, 2
            )
            for class_index, probability in zip(
                model.classes_, probabilities
            )
        }
        record = Prediction(
        sepal_length=values[0],
        sepal_width=values[1],
        petal_length=values[2],
        petal_width=values[3],
        predicted_species=species
        )

        db.session.add(record)
        db.session.commit()

        return jsonify({
            "predicted_species": species,
            "probabilities": class_probabilities
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/history", methods=["GET"])
def history():
    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        predictions=predictions
    )



@app.route("/dashboard")
def dashboard():
    total_predictions = Prediction.query.count()

    species_names = ["setosa", "versicolor", "virginica"]

    rows = db.session.query(
        Prediction.predicted_species,
        func.count(Prediction.id)
    ).group_by(
        Prediction.predicted_species
    ).all()

    saved_counts = {
        species: count for species, count in rows
    }

    species_data = {
        species: saved_counts.get(species, 0)
        for species in species_names
    }

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        species_data=species_data
    )


@app.route("/export")
def export_predictions():
    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID",
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width",
        "Predicted Species",
        "Created At"
    ])

    for item in predictions:
        writer.writerow([
            item.id,
            item.sepal_length,
            item.sepal_width,
            item.petal_length,
            item.petal_width,
            item.predicted_species,
            item.created_at.isoformat()
        ])

    response = make_response("\ufeff" + output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = (
        "attachment; filename=iris_prediction_history.csv"
    )

    return response


if __name__ == "__main__":
    app.run(debug=True)

