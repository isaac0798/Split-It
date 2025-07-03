from flask import Flask, request, jsonify
import cv2
import numpy as np
# ... your other imports

app = Flask(__name__)
  
@app.route('/')
def home():
    return jsonify({'message': 'API is running'})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        return jsonify({'result': 'test', 'status': 'success'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)