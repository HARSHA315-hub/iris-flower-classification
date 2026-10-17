
import os
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import load_data, split_data


# 1. Load the dataset
X, y, class_names = load_data()

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = split_data(X, y)


# 3. Define the five machine-learning models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1, probability=True, random_state=42)
    )
}


# 4. Set up five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
best_model_name = None
best_cv_accuracy = -1
best_model = None


# 5. Train and evaluate every model
for name, model in models.items():

    # Train using the training data
    model.fit(X_train, y_train)

    # Predict species for the test data
    predictions = model.predict(X_test)

    # Calculate test-set metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    # Calculate five-fold cross-validation accuracy
    cv_scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy"
    )
    mean_cv_accuracy = cv_scores.mean()

    # Store the results
    results.append({
        "Model": name,
        "Test Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100,
        "CV Accuracy": mean_cv_accuracy * 100
    })

    # Print this model's results
    print("\n" + "=" * 45)
    print("Model:", name)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision * 100:.2f}%")
    print(f"Macro Recall: {recall * 100:.2f}%")
    print(f"Macro F1 Score: {f1 * 100:.2f}%")
    print(f"5-Fold CV Accuracy: {mean_cv_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Select the model with the highest CV accuracy
    if mean_cv_accuracy > best_cv_accuracy:
        best_cv_accuracy = mean_cv_accuracy
        best_model_name = name
        best_model = model


# 6. Display the comparison table
print("\n\n========== MODEL COMPARISON ==========")

for result in results:
    print(
        f"{result['Model']:<22}"
        f"Test: {result['Test Accuracy']:6.2f}%  "
        f"CV: {result['CV Accuracy']:6.2f}%"
    )


# 7. Save the best model
os.makedirs("models", exist_ok=True)

model_data = {
    "model": best_model,
    "name": best_model_name,
    "classes": class_names
}

joblib.dump(model_data, "models/best_model.joblib")

print("\n========== BEST MODEL ==========")
print("Best model:", best_model_name)
print(f"Mean CV accuracy: {best_cv_accuracy * 100:.2f}%")
print("Saved to: models/best_model.joblib")
