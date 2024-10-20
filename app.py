from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Update the model path to the correct location
model_path = r'C:\Users\prath\Desktop\AgroVisionAI\models\model.pkl'
model = joblib.load(model_path)

# Home route to display the main page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit_form():
    # Get form data
    name = request.form['name']
    mobile = request.form['mobile']
    message = request.form['message']
    
    # Process the form data as needed
    print(f"Name: {name}, Mobile: {mobile}, Message: {message}")
    
    # After processing the form, redirect to the thank-you page
    return redirect(url_for('thank_you'))

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data for prediction
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0]}')

# Thank you route to display the thank-you message
@app.route('/thankyou')
def thank_you():
    return render_template('thankyou.html')

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
