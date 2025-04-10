from flask import Blueprint, render_template, request, jsonify
import logging

# Define the blueprint
fertilizer_bp = Blueprint('fertilizer', __name__, template_folder='templates')

# Configure logging
logging.basicConfig(filename='api.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@fertilizer_bp.route('/')
def home():
    return render_template('fs.html')  # Ensure fs.html exists in templates/fertilizer/

@fertilizer_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received request: {data}")

        # Validate input fields
        required_fields = ["temperature", "humidity", "nitrogen", "phosphorous", "potassium", "soil_type", "crop_name"]
        for field in required_fields:
            if field not in data or data[field] == "":
                logging.error(f"Missing parameter: {field}")
                return jsonify({"error": f"Missing parameter: '{field}'"}), 400

        # Dummy ML Model Response (Replace with actual ML model logic)
        response = {
            "fertilizer": "Urea",
            "dosage": "50 kg/ha"
        }

        logging.info(f"Response: {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500
