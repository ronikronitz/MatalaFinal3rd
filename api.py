import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from def1 import *

app = Flask(__name__)
pipeline = pickle.load(open('trained_model.pkl', 'rb'))

# Define the route for the prediction form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    time= request.form['entrance_date']
    entranceDate=format_enter(time)
    myDict = {
        'City': [request.form['City']],
        'type': [request.form['type']],
        'room_number': [float(request.form['room_number'])],
        'Area': [float(request.form['Area'])],
        'Street': [request.form['Street']],
        'number_in_street': [float(request.form['number_in_street'])],
        'city_area': [request.form['city_area']],
        'num_of_images': [float(request.form['num_of_images'])],
        'floor': [request.form['floor']],
        'total_floors': [request.form['total_floors']],
        'hasElevator': [float(request.form['hasElevator'])],
        'hasParking': [float(request.form['hasParking'])],
        'hasBars': [float(request.form['hasBars'])],
        'hasStorage': [float(request.form['hasStorage'])],
        'condition': [request.form['condition']],
        'hasAirCondition': [float(request.form['hasAirCondition'])],
        'hasBalcony': [float(request.form['hasBalcony'])],
        'hasMamad': [float(request.form['hasMamad'])],
        'handicapFriendly': [float(request.form['handicapFriendly'])],
        'entrance_date': [entranceDate],
        'furniture': [request.form['furniture']],
        'publishedDays': [float(request.form['publishedDays'])],
        'description': [request.form['description']]
    }

    prediction = pd.DataFrame(myDict)

    # Perform the prediction
    predicted_price = pipeline.predict(prediction)

    # Return the predicted price to the user
    return render_template('index.html', prediction_text='The home should be shekels {}'.format(predicted_price))

  
if __name__ == "__main__": #when im running this code from somewhere else it won't run this
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
