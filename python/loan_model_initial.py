import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import joblib

# Load the data
file_path = 'D:\GenerativeAI LLMs\Day2\loan_data_nov2023.csv'  # Replace with your file path
loan_data = pd.read_csv(file_path)

# Identify categorical and numerical columns
categorical_cols = ['grade', 'ownership']
numerical_cols = ['years', 'income', 'age', 'amount', 'interest']

# Define the feature columns and target variable
X = loan_data.drop('default', axis=1)
y = loan_data['default']

# Splitting the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Dictionary to store evaluation metrics
model_metrics = {}

# Training, evaluating each model and saving them to the specified path
for name, model in models.items():
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Store the metrics
    model_metrics[name] = {'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

    # Save the trained model to the specified path
    model_filename = 'D:\GenerativeAI LLMs\Day2\Result\\' + name.replace(" ", "_").lower() + '_model.pkl'
    joblib.dump(pipeline, model_filename)
    print(f"{name} model saved as {model_filename}")

# Display the evaluation metrics for each model
for model_name, metrics in model_metrics.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print()

for model_name, metrics in model_metrics.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print()