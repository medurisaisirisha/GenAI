import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the data
data = pd.read_csv('D:/GenerativeAI LLMs/Day2/loan_data_nov2023.csv')

# Split the data into features and target variable
X = data.drop(['default'], axis=1)
y = data['default']

# Split the data into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
class_weights = y_train.value_counts(normalize=True).to_dict()

# Define the preprocessor
numeric_features = ['years', 'income', 'age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['ownership', 'grade']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ... (previous code remains unchanged)

# ... (previous code remains unchanged)

# Define models
models = {
    'Logistic Regression': (LogisticRegression(random_state=42, class_weight=class_weights),
                            {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
    'Decision Tree': (DecisionTreeClassifier(random_state=42),
                      {'classifier__max_depth': [3, 5, 7, 10],
                       'classifier__min_samples_split': [2, 5, 10],
                       'classifier__min_samples_leaf': [1, 2, 4]}),
    'Random Forest': (RandomForestClassifier(random_state=42, class_weight=class_weights),
                      {'classifier__n_estimators': [50, 100],
                       'classifier__max_depth': [3, 5],
                       'classifier__min_samples_split': [2, 5],
                       'classifier__min_samples_leaf': [1, 2]}),
    'XGBoost': (XGBClassifier(random_state=42),
                {'classifier__n_estimators': [50, 100],
                 'classifier__max_depth': [3, 5, 7]}),
    'Naive Bayes': (GaussianNB(), {})
}

# Train and evaluate models
best_models = {}
for name, (model, param_grid) in models.items():
    # Create pipeline with preprocessing and the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])

    # Hyperparameter tuning using grid search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'\nMetrics for {name}:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Save the best model
    model_filename = 'D:\GenerativeAI LLMs\Day2\Result\\' + name.replace(" ", "_").lower() + '_model.pkl'
    with open(model_filename, 'wb') as file:
        joblib.dump(best_model, file)
    print(f'Model "{name}" saved successfully.')