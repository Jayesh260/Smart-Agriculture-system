from flask import Blueprint, render_template, request
import requests
import pickle
import numpy as np

croprecommend_bp = Blueprint('croprecommend', __name__, template_folder='templates')

# Load the trained Random Forest model
with open(r"D:\NewCode\Allcombined\Croprecommend\RandomForest.pkl", "rb") as f:
    rf_model = pickle.load(f)

# OpenWeather API Key
API_KEY = "e457e64bcfa973c741603ce17bfa3975"

def get_weather(city):
    """Fetch weather data from OpenWeather API."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": API_KEY, "units": "metric"}

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        rainfall = weather_data.get("rain", {}).get("1h", 0)  # Default to 0 if no rain data
        return temperature, humidity, rainfall
    else:
        return None, None, None  # Handle errors

@croprecommend_bp.route("/", methods=["GET", "POST"])
@croprecommend_bp.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        use_api = request.form.get("use_api")
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            ph = float(request.form["ph"])
        except ValueError:
            return render_template("cr.html", prediction="Invalid input. Please enter valid numbers.")

        # Use Weather API if selected
        if use_api == "yes":
            city = request.form.get("city")
            temperature, humidity, rainfall = get_weather(city)

            if temperature is None:
                return render_template("cr.html", prediction="Weather data not available. Please enter a valid city.")
        else:
            try:
                temperature = float(request.form["temperature"])
                humidity = float(request.form["humidity"])
                rainfall = float(request.form["rainfall"])
            except ValueError:
                return render_template("cr.html", prediction="Invalid weather input. Please enter valid numbers.")

        # Prepare input and predict
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = rf_model.predict(input_features)[0]

    return render_template("cr.html", prediction=prediction)
