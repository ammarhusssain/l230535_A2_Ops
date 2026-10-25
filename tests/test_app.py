from app.app import create_app
import io
import os
os.environ.update(INFERENCE_MODE='remote', API_URL='http://example.invalid', API_KEY='k', DATABASE_URL='sqlite://')

client = create_app().test_client()


def test_health():
    assert client.get('/health').status_code == 200


def test_wrong_type():
    r = client.post('/predict', data={'image': (io.BytesIO(b'hello'), 'notes.txt')},
                    content_type='multipart/form-data')
    assert r.status_code == 415


def test_fake_image():
    r = client.post('/predict', data={'image': (io.BytesIO(b'not an image'), 'fake.jpg')},
                    content_type='multipart/form-data')
    assert r.status_code == 422


def test_api_needs_key():
    assert client.post('/api/predict').status_code == 401
