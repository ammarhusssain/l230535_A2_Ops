import os
class Config:
    INFERENCE_MODE = os.getenv('INFERENCE_MODE', 'local')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/champion.keras')
    MODEL_NAME = os.getenv('MODEL_NAME', 'visionguard')
    MODEL_VERSION = os.getenv('MODEL_VERSION', '2')
    THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///visionguard.db')
    API_URL = os.getenv('API_URL', '')
    API_KEY = os.getenv('API_KEY', '')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_MB', '5')) * 1024 * 1024
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-me')