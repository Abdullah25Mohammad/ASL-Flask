from flask import Flask, Response, render_template, request
from flask_cors import CORS
import cv2
import numpy as np
from tf_keras.models import load_model
from ASLTranslator import translator

app = Flask(__name__)
CORS(app)

# Load the model once at the start
smnist_model = load_model('smnist.h5')
letter = "None"
conf = 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global letter, conf  # Use global variables here if necessary
    
    # Read the incoming frame
    file = request.files['frame'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process the frame for analysis
    detected, a_frame = translator.analysis_frame(frame)
    if detected:
        letter, conf = translator.analyze(a_frame, model=smnist_model)

    # Optionally, draw the detected letter on the frame for visualization
    cv2.putText(frame, f'Letter: {letter}, Conf: {conf:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Encode the processed frame back to JPEG format
    ret, buffer = cv2.imencode('.jpg', frame)
    processed_frame = buffer.tobytes()

    return Response(processed_frame, mimetype='image/jpeg')

@app.route('/get_letter', methods=['GET'])
def get_letter():
    global letter, conf
    print(f'Letter: {letter}, Conf: {conf:.2f}')
    return {'letter': letter, 'conf': conf}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
