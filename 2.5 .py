# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Iris dataset (this is a standard, labeled dataset)
# The URL in the prompt (archive.ics.uci.edu/ml/datasets/iris) points to the same data
iris = load_iris()

# The data is already labeled! This confirms we have labeled data.
# Features: sepal length, sepal width, petal length, petal width
# Labels: Setosa, Versicolour, Virginica (0, 1, 2)
X = iris.data  # Features
y = iris.target  # Labels

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("Microsoft Azure ML - Model Selection")
print("=" * 60)
print("1. Dataset Analysis:")
print(f"   - Features: {list(iris.feature_names)}")
print(f"   - Target: Species ({iris.target_names})")
print(f"   - Total samples: {len(df)}")
print(f"   - Data type: LABELED (supervised learning problem)")

# Since we detected LABELED data, we select a Supervised ML model
print("\n2. Model Selection Logic:")
print("   - Input: Labeled data detected")
print("   - Decision: Select Supervised Machine Learning")
print("   - Output: Classification Model (predicting flower species)")

# Implement the chosen supervised model (Classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train a classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n3. Azure ML Implementation Result:")
print(f"   - Model trained: Random Forest Classifier")
print(f"   - Test accuracy: {accuracy:.2%}")
print(f"   - Industry: Azure ML")
print("=" * 60)

# Show sample predictions
sample_indices = [0, 50, 100]  # One of each species
print("\nSample Predictions:")
for idx in sample_indices:
    true_species = iris.target_names[iris.target[idx]]
    pred_species = iris.target_names[model.predict([iris.data[idx]])[0]]
    print(f"   Sample {idx}: True={true_species:12} → Predicted={pred_species}")
