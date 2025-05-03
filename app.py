from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and imputer
model = joblib.load('student_performance_model.pkl')
imputer = joblib.load('imputer.pkl')

# Load CSV for dashboard and data validation
data = pd.read_csv('C:/Users/Shrivastava/Downloads/student_performance_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form se data lena
        attendance = float(request.form['attendance'])
        marks_obtained = float(request.form['marks_obtained'])
        total_marks = float(request.form['total_marks'])
        percentage = float(request.form['percentage'])
        class_rank = int(request.form['class_rank'])

        # Check student exist or not
        matched = data[
            (data['Attendance'] == attendance) &
            (data['Marks Obtained'] == marks_obtained) &
            (data['Total Marks'] == total_marks) &
            (data['Percentage'] == percentage) &
            (data['Class Rank'] == class_rank)
        ]

        if matched.empty:
            
            message = "Student not found in the database. Please check the details."
            return render_template('index.html', message=message)

        # Prediction ke liye input prepare
        input_data = pd.DataFrame([[attendance, marks_obtained, total_marks, percentage, class_rank]],
                                  columns=['Attendance', 'Marks Obtained', 'Total Marks', 'Percentage', 'Class Rank'])
        input_data = imputer.transform(input_data)
        prediction = model.predict(input_data)

        return render_template('index.html', performance=f"Predicted Performance: {prediction[0]}")

    except Exception as e:
        return render_template('index.html', message=f"Error: {str(e)}")

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard_data')
def dashboard_data():
    try:
        avg_attendance = round(data['Attendance'].mean(), 2)
        avg_marks = round(data['Marks Obtained'].mean(), 2)
        avg_percentage = round(data['Percentage'].mean(), 2)

        top_students = data[['Name', 'Marks Obtained']].sort_values(by='Marks Obtained', ascending=False).head(5).to_dict(orient='records')

        return jsonify({
            'average_attendance': avg_attendance,
            'average_marks': avg_marks,
            'average_percentage': avg_percentage,
            'top_students': top_students
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

