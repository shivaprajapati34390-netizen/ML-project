import joblib
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Load trained model
model = joblib.load("book2/housing_model.pkl")  # <-- make sure the filename is exact

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import pandas as pd
    
    # Extract features in correct order
    features = {
        'longitude': float(request.form['longitude']),
        'latitude': float(request.form['latitude']),
        'housing_median_age': float(request.form['housing_median_age']),
        'total_rooms': float(request.form['total_rooms']),
        'total_bedrooms': float(request.form['total_bedrooms']),
        'population': float(request.form['population']),
        'households': float(request.form['households']),
        'median_income': float(request.form['median_income']),
        'ocean_proximity': request.form['ocean_proximity']
    }
    
    # Create DataFrame
    input_data = pd.DataFrame([features])
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_value = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f"Predicted House Price: ${predicted_value:,}")

if __name__ == "__main__":
    app.run(debug=True)

