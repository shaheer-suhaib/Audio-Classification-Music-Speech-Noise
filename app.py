from flask import Flask, request, render_template
import os
import librosa
import numpy as np
import pickle
from werkzeug.utils import secure_filename


label_mapping = {0: "Noise", 1: "Speech", 2: "Music"}

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Ensure it's inside the static folder
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def feature_extraction(audio, sample_rate):
    """Extract features from the audio signal."""
    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=64)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Extract Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_scaled_features = np.mean(zcr.T, axis=0)

    # Combine features
    combined_features = np.concatenate((mfccs_scaled_features, zcr_scaled_features))
    return combined_features

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('index.html', prediction="No file uploaded. Please upload a WAV file.")

        file = request.files['audio']

        if file.filename == '':
            return render_template('index.html', prediction="No file selected. Please upload a WAV file.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the uploaded file

            # Load the audio file
            audio, sample_rate = librosa.load(file_path)

            # Extract features
            features = feature_extraction(audio, sample_rate)
            features = features.reshape(1, -1)  # Reshape for the model

            # Make a prediction
            prediction = model.predict(features)
            predicted_label = np.argmax(prediction, axis=1)[0]  # Get the scalar value of the predicted class index

            # Map the predicted label index to a human-readable label
            predicted_class = label_mapping.get(predicted_label, "Unknown")

            prediction_text = f"Classified as: {predicted_class}"

            # Pass the filename to render the audio player
            return render_template('index.html', prediction=prediction_text, audio_file=filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
