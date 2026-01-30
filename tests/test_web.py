from io import BytesIO
from web.app import app
from PIL import Image, ImageDraw


def create_synthetic_png_bytes():
    img = Image.new('RGB', (800, 300), 'white')
    draw = ImageDraw.Draw(img)
    for x in range(0, 800, 25):
        draw.line([(x, 0), (x, 300)], fill=(230, 230, 230))
    # simple sine as trace
    import numpy as np
    t = np.linspace(0, 10, 800)
    sig = 140 + 40 * np.sin(2 * np.pi * 1.0 * t)
    for i, y in enumerate(sig.astype(int)):
        draw.point((i, y), fill='black')
    bio = BytesIO()
    img.save(bio, format='PNG')
    bio.seek(0)
    return bio


def test_web_process_endpoint():
    client = app.test_client()
    bio = create_synthetic_png_bytes()
    data = {'file': (bio, 'synthetic.png')}
    resp = client.post('/process', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    j = resp.get_json()
    assert 'features' in j
