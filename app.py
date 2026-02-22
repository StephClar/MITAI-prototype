from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import analyze_sequence, compare_sequences, load_trained_model

app = Flask(__name__, static_folder='.')
CORS(app)

# Pre-load model on startup
print("Loading model...")
try:
    load_trained_model()
    print("Model ready!")
except Exception as e:
    print(f"Model will train on first request: {e}")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    sequence = data.get('sequence', '').strip()
    if not sequence:
        return jsonify({"error": "No sequence provided"}), 400
    result = analyze_sequence(sequence)
    return jsonify(result)

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    seq1 = data.get('sequence1', '').strip()
    seq2 = data.get('sequence2', '').strip()
    if not seq1 or not seq2:
        return jsonify({"error": "Both sequences are required"}), 400
    result = compare_sequences(seq1, seq2)
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model": "loaded"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)