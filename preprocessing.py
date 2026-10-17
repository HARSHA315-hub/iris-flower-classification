
# preprocessing.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# The four input features used by our Machine Learning model
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]


def load_data():
    """Load the Iris dataset into features and target labels."""

    data = load_iris(as_frame=True)

    X = data.data
    y = data.target
    class_names = list(data.target_names)

    return X, y, class_names


def check_quality(X, y):
    """Check missing values, duplicate feature rows and class balance."""

    print("\n--- DATASET QUALITY REPORT ---")

    print("Missing values:", int(X.isna().sum().sum()))
    print("Duplicate feature rows:", int(X.duplicated().sum()))

    print("\nClass counts:")
    print(y.value_counts().sort_index())


def split_data(X, y, test_size=0.2, seed=42):
    """Split data while preserving the proportion of each species."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Load the dataset
    X, y, class_names = load_data()

    # Display basic dataset information
    print("Dataset loaded successfully!")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    print("\nFirst five samples:")
    print(X.head())

    print("\nSpecies labels:")
    print(class_names)

    print("\nFeature statistics:")
    print(X.describe())

    # Check the dataset quality
    check_quality(X, y)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n--- TRAIN-TEST SPLIT ---")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class counts:")
    print(y_test.value_counts().sort_index())