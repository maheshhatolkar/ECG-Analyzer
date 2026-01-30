from flask import Flask, request, jsonify, render_template
import os
import tempfile

from ecg_analyzer.extractor import process_file

"""Flask web application exposing a single-file upload API.

Routes:
- GET / -> serves a simple HTML page with an upload form
- POST /process -> accepts a single file upload, runs the pipeline, and
  returns JSON results. The endpoint uses a temporary file for compatibility
  with the existing disk-based `process_file` API.

The server is intentionally minimal - it uses the same processing pipeline
as the CLI/GUI so behavior is consistent across frontends.
"""

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))


@app.route('/')
def index():
    # Serve the static HTML UI
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    # Accept a multipart upload with form field 'file'
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    # Write upload to a temporary file and pass the path to the existing pipeline
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1]) as tmp:
        tmp.write(f.read())
        tmp.flush()
        tmp_path = tmp.name
    try:
        res = process_file(tmp_path, output=None, out_format='json', show_plot=False)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == '__main__':
    # Development server (not for production use)
    app.run(port=5000, debug=True)
