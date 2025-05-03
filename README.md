# Student Performance Prediction 🎓

This project predicts student performance based on various factors like gender, lunch type, parental education, etc.

## 🔧 Tech Stack:
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn

## 🎯 Goal:
Predict the average student score using regression models.


Student Performance/
│
├── static/
│   └── style.css              # (optional CSS)
│
├── templates/
│   └── index.html             # Webpage for user input
│
├── model/
│   └── student_model.pkl      # Saved ML model
│
├── app.py                     # Flask backend
├── main.py                    # Your ML training script
├── requirements.txt           # Required packages
└── README.md                  # Project info


.\venv\Scripts\Activate

python -m pip install dash-core-components
python -m pip install dash-html-components

SimpleImputer ka use karte hue hum missing values ko fill kar rahe hain. Yaha strategy mean use ki gayi hai, jo har column ka average value fill kar dega jaha data missing hai.

fit_transform se hum imputer ko data pe fit kar rahe hain aur transform kar rahe hain (matlab missing values ko replace kar rahe hain).

train_test_split ka use karke data ko train aur test sets mein divide kiya gaya hai.

test_size=0.2 ka matlab hai 20% data testing ke liye aur 80% training ke liye use hoga.

random_state=42 se hum ensure kar rahe hain ki data ko har baar same tarike se split kiya jaye.

RandomForestClassifier ek machine learning model hai jo classification problems ke liye use hota hai.

n_estimators=100 ka matlab hai ki model 100 decision trees ka ensemble (group) use karega.

fit(X_train, y_train) se hum model ko training data (X_train aur y_train) pe train kar rahe hain.

model.classes_ se hum model ki internal class mapping ko print kar rahe hain.

Isse hume pata chalega ki model kis class ko kis numeric value se map karta hai.

Agar Overall Performance ke andar values "Excellent", "Good", "Average" hain, to model internal representation mein unhe numeric values se map karega (e.g., 0 = "Average", 1 = "Excellent").

model.predict(X_test) se hum model se predictions karwa rahe hain. Yaha X_test ke liye predicted values mil rahi hain.

accuracy_score ka use karke hum model ki accuracy calculate kar rahe hain.

accuracy * 100 se percentage me result milta hai.

joblib.dump ka use karke hum trained model aur imputer ko files mein save kar rahe hain.

Ye files future mein use ho sakti hain jab hume predictions karni ho ya model ko dubara train nahi karna ho.

venv/
