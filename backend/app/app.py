from flask import Flask,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from celery.schedules import crontab,schedule
from flask_jwt_extended import JWTManager

from routes.chat_route import chat_bp, init_chat

app = Flask(__name__)
CORS(app)  
jwt = JWTManager(app)


init_chat()

app.register_blueprint(chat_bp)

if __name__ == '__main__':
    #with app.app_context():
        #db.create_all()  
    app.run(debug=True, host='0.0.0.0', port=5000)