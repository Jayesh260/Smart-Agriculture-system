from flask import Flask, render_template, Blueprint,request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model_path = r"D:\NewCode\CropYield\crop_yield_xgb.pkl"
model = joblib.load(model_path)

# Define categorical mappings
crops = ['Arecanut' 'Arhar/Tur' 'Castor seed' 'Coconut ' 'Cotton(lint)'
 'Dry chillies' 'Gram' 'Jute' 'Linseed' 'Maize' 'Mesta' 'Niger seed'
 'Onion' 'Other  Rabi pulses' 'Potato' 'Rapeseed &Mustard' 'Rice'
 'Sesamum' 'Small millets' 'Sugarcane' 'Sweet potato' 'Tapioca' 'Tobacco'
 'Turmeric' 'Wheat' 'Bajra' 'Black pepper' 'Cardamom' 'Coriander' 'Garlic'
 'Ginger' 'Groundnut' 'Horse-gram' 'Jowar' 'Ragi' 'Cashewnut' 'Banana'
 'Soyabean' 'Barley' 'Khesari' 'Masoor' 'Moong(Green Gram)'
 'Other Kharif pulses' 'Safflower' 'Sannhamp' 'Sunflower' 'Urad'
 'Peas & beans (Pulses)' 'other oilseeds' 'Other Cereals' 'Cowpea(Lobia)'
 'Oilseeds total' 'Guar seed' 'Other Summer Pulses' 'Moth'] 
seasons = ['Whole Year', 'Kharif', 'Rabi', 'Autumn', 'Summer', 'Winter']
states = ['Assam' 'Karnataka' 'Kerala' 'Meghalaya' 'West Bengal' 'Puducherry' 'Goa'
 'Andhra Pradesh' 'Tamil Nadu' 'Odisha' 'Bihar' 'Gujarat' 'Madhya Pradesh'
 'Maharashtra' 'Mizoram' 'Punjab' 'Uttar Pradesh' 'Haryana'
 'Himachal Pradesh' 'Tripura' 'Nagaland' 'Chhattisgarh' 'Uttarakhand'
 'Jharkhand' 'Delhi' 'Manipur' 'Jammu and Kashmir' 'Telangana'
 'Arunachal Pradesh' 'Sikkim']  # Add more states

cropyield_bp = Blueprint('cropyield', __name__, template_folder='templates')

@cropyield_bp.route('/')
def index():
    return render_template('cy.html')
@app.route('/')
def home():
    return render_template('cy.html')

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        # Extract input values
        crop_name = request.form.get('crop_name')
        crop_year = int(request.form.get('crop_year'))
        season = request.form.get('season')
        state = request.form.get('state')
        area = float(request.form.get('area'))
        production = float(request.form.get('production'))
        annual_rainfall = float(request.form.get('annual_rainfall'))
        fertilizer = float(request.form.get('fertilizer'))

        # Validate input values
        if crop_name not in crops or season not in seasons or state not in states:
            return jsonify({'error': 'Invalid crop, season, or state selection.'})

        # Convert categorical values to numerical representations
        crop_index = crops.index(crop_name)
        season_index = seasons.index(season)
        state_index = states.index(state)

        # Prepare input data
        input_data = np.array([[crop_index, crop_year, season_index, state_index, area, production, annual_rainfall, fertilizer]])

        # Make prediction
        predicted_yield = model.predict(input_data)[0]

        return jsonify({'predicted_yield': float(predicted_yield)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)





