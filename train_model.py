import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.impute import SimpleImputer

file_path = "C:/Users/Shrivastava/Downloads/student_performance_data.csv"
df = pd.read_csv(file_path)

# Columns
selected_features = ['Attendance', 'Marks Obtained', 'Total Marks', 'Percentage', 'Class Rank']
target_column = 'Overall Performance'

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[selected_features] = imputer.fit_transform(df[selected_features])

# Features (X) and target (y)
X = df[selected_features]
y = df[target_column]

# ====> YAHAN SE ADD KARNA ✅ <====
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy check
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and imputer
joblib.dump(model, 'student_performance_model.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("Model training done and saved successfully!")
print("Model classes:", model.classes_)



