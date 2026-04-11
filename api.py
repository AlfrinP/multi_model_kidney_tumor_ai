from flask import Flask
from app.api.routes import api_bp
from app.config import Config

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(host=Config.API_HOST, port=Config.API_PORT, debug=False)
