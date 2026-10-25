import io
from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from PIL import Image, UnidentifiedImageError
from app.config import Config
from app.inference import build_predictor, ModelServiceError
from app.db import init_db, save_prediction, recent_predictions, db_ok

ALLOWED = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'}


def validate(file):
    if not file or file.filename == '':
        abort(400, 'No file selected.')
    name = secure_filename(file.filename)
    if '.' not in name or name.rsplit('.', 1)[1].lower() not in ALLOWED:
        abort(415, 'Only JPG, PNG, WEBP, BMP or GIF images are accepted.')
    data = file.read()
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        abort(422, 'This file has an image extension but cannot be decoded as an image.')
    return name, data


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    init_db(Config.DATABASE_URL)
    predictor = build_predictor(Config)

    def run_model(data, name):
        try:
            return predictor.predict(data, name)
        except ModelServiceError as e:      # remote mode: cloud API down or rejected the request
            abort(e.code, e.message)

    @app.get('/')
    def index():
        return render_template('index.html', max_mb=Config.MAX_CONTENT_LENGTH // 1048576)

    @app.post('/predict')
    def predict():
        name, data = validate(request.files.get('image'))
        res = run_model(data, name)
        save_prediction(name, res, source='web', client_ip=request.remote_addr)
        return render_template('result.html', r=res, filename=name)

    @app.get('/history')
    def history():
        return render_template('history.html', rows=recent_predictions(100))

    @app.get('/health')
    def health():
        return jsonify(status='ok', mode=Config.INFERENCE_MODE, model=Config.MODEL_NAME,
                       version=Config.MODEL_VERSION, database='ok' if db_ok() else 'down')

    @app.post('/api/predict')
    def api_predict():
        if not Config.API_KEY or request.headers.get('X-API-Key') != Config.API_KEY:
            return jsonify(error='invalid or missing API key'), 401
        name, data = validate(request.files.get('image'))
        res = run_model(data, name)
        save_prediction(name, res, source='api', client_ip=request.remote_addr)
        return jsonify(res)

    @app.errorhandler(HTTPException)
    def handle(e):
        if request.path.startswith('/api/'):
            return jsonify(error=e.description), e.code
        return render_template('error.html', code=e.code, message=e.description), e.code

    return app


app = create_app()
