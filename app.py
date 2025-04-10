from flask import Flask, render_template
from Chatbot.routes import chatbot_bp
from Croprecommend.routes import croprecommend_bp
from CropYield.routes import cropyield_bp
from Fertilizer.routes import fertilizer_bp  # ✅ Add this line
from DiseaseNew.routes import plantdisease_bp
app = Flask(__name__)
# Register Blueprints
app.register_blueprint(chatbot_bp, url_prefix="/chatbot")
app.register_blueprint(croprecommend_bp, url_prefix="/croprecommend")
# app.register_blueprint(cropyield_bp, url_prefix="/cropyield")
app.register_blueprint(plantdisease_bp, url_prefix="/plantdisease")
# app.register_blueprint(fertilizer_bp, url_prefix='/fertilizer')  # ✅ Register fertilizer

# Render simple HTML home with buttons
@app.route("/")
def home():
    return render_template("home.html")
if __name__ == "__main__":
    app.run(debug=True)
