#!/usr/bin/env python3
import os
import sqlite3
import logging
import base64
import pickle
from datetime import datetime
from pathlib import Path
from functools import wraps
from flask import (Flask, flash, redirect, render_template_string, request,
                   session, url_for, jsonify)
import cv2
import mediapipe as mp
import numpy as np

# App and DB config
APP = Flask(__name__)
APP.secret_key = os.environ.get("SVM_SECRET_KEY", "svm_demo_secret_key_change_me")
BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "svm_admin.db"

# Admin credentials
ADMIN_USER = "poomalai005"
ADMIN_PASS = "Poomalai2005@"

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Ensure DB
def get_conn():
    """Returns a new database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database tables and migrations if they don't exist."""
    created = not DB_PATH.exists()
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS voters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_id TEXT UNIQUE,
            name TEXT,
            dob TEXT,
            phone TEXT,
            fingerprint TEXT,
            has_voted INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter_id TEXT,
            candidate TEXT,
            timestamp TEXT
        )
    """)
    
    # Check for the new 'face_data' column and add it if it's missing.
    c.execute("PRAGMA table_info(voters)")
    columns = [row['name'] for row in c.fetchall()]
    if 'face_data' not in columns:
        c.execute("ALTER TABLE voters ADD COLUMN face_data TEXT")
    
    conn.commit()
    conn.close()
    return created
init_db()

# Utility functions
def calculate_age(dob_str):
    """Calculates age in years from a YYYY-MM-DD date string."""
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
    except Exception:
        return -1
    today = datetime.utcnow()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, today.day))
    return age

def send_sms(to_number, message):
    """
    Simulates sending an SMS notification.
    In a real-world scenario, this would use a service like Twilio or Vonage.
    For this demo, we'll log the message to the console.
    """
    logging.info(f"Simulating SMS to {to_number}: {message}")

# --- MediaPipe/Face Recognition Functions ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_face_data(image_data_b64):
    """Detects a face in a base64 image and returns the serialized landmark data."""
    try:
        img_bytes = base64.b64decode(image_data_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        if results.detections:
            landmarks = results.detections[0].location_data.relative_keypoints
            landmark_array = np.array([(p.x, p.y) for p in landmarks])
            return pickle.dumps(landmark_array) # Serialize for storage
    except Exception as e:
        logging.error(f"Error processing image with MediaPipe: {e}")
    return None

def compare_faces(known_face_data_b64, live_face_data_b64):
    """Compares two pickled face data arrays and returns true if they are a match."""
    try:
        known_landmarks = pickle.loads(known_face_data_b64)
        live_landmarks = pickle.loads(live_face_data_b64)
        distance = np.linalg.norm(live_landmarks - known_landmarks)
        logging.info(f"Face comparison distance: {distance}")
        return distance < 0.4  # Increased tolerance for a match
    except Exception as e:
        logging.error(f"Error comparing faces: {e}")
    return False

# --------------------
# Admin pages (using render_template_string)
# --------------------
ADMIN_LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<title>Admin Login</title>
<style>
body{font-family: Arial; max-width: 500px; margin: 20px auto; padding: 15px; background-color: #000; color: #fff; border-radius: 8px; box-shadow: 0 4px 8px rgba(255,255,255,0.1);}
h2{color: #fff; text-align: center;}
form{display: flex; flex-direction: column; gap: 10px;}
input{padding: 10px; border-radius: 5px; border: 1px solid #555; background-color: #333; color: #fff;}
button{padding: 10px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;}
a{color: #3498db;}
</style>
</head>
<body>
<h2>Admin Login</h2>
{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        <ul class="flashes">
        {% for cat, msg in messages %}
            <li class="{{ cat }}">{{ msg }}</li>
        {% endfor %}
        </ul>
    {% endif %}
{% endwith %}
<form method="post">
    <label for="username">Username:</label>
    <input type="text" id="username" name="username" required>
    <label for="password">Password:</label>
    <input type="password" id="password" name="password" required>
    <button type="submit">Login</button>
</form>
<p><a href="/">Back to Voting</a></p>
</body>
</html>
"""

ADMIN_ADD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<title>Add Voter</title>
<style>
body{font-family: Arial; max-width: 700px; margin: 20px auto; padding: 15px; background-color: #000; color: #fff; border-radius: 8px; box-shadow: 0 4px 8px rgba(255,255,255,0.1);}
h2{color: #fff; text-align: center;}
p a{color: #3498db;}
form{display: flex; flex-direction: column; gap: 10px;}
label{font-weight: bold;}
input{padding: 10px; border-radius: 5px; border: 1px solid #555; background-color: #333; color: #fff;}
button{padding: 10px; background-color: #28a745; color: white; border: none; cursor: pointer;}
video, canvas, #face-preview { border: 1px solid white; display: block; margin-top: 10px; }
.face-section { margin-top: 20px; padding: 15px; border: 1px dashed #555; border-radius: 8px; }
</style>
</head>
<body>
<h2>Admin — Add Voter</h2>
<p><a href="/admin/list">List voters</a> | <a href="/admin/logout">Logout</a></p>
{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        <ul class="flashes">
        {% for cat, msg in messages %}
            <li class="{{ cat }}">{{ msg }}</li>
        {% endfor %}
        </ul>
    {% endif %}
{% endwith %}
<form method="post" id="add-voter-form">
    <label for="voter_id">Voter QR ID (unique):</label>
    <input type="text" id="voter_id" name="voter_id" required>
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" required>
    <label for="dob">DOB (YYYY-MM-DD):</label>
    <input type="date" id="dob" name="dob" required>
    <label for="phone">Phone:</label>
    <input type="text" id="phone" name="phone">
    <label for="fingerprint">Fingerprint template (text placeholder):</label>
    <input type="text" id="fingerprint" name="fingerprint">
    
    <div class="face-section">
        <h3>Enroll Face</h3>
        <p>Position the voter's face in the camera frame.</p>
        <video id="webcam" width="320" height="240" autoplay></video>
        <button type="button" id="capture-face-btn">Capture Face</button>
        <canvas id="face-preview" width="320" height="240" style="display: none;"></canvas>
        <input type="hidden" id="face-data" name="face_data">
    </div>

    <button type="submit">Add Voter</button>
</form>

<script>
let video = document.getElementById('webcam');
let faceDataInput = document.getElementById('face-data');
let captureBtn = document.getElementById('capture-face-btn');
let facePreviewCanvas = document.getElementById('face-preview');

// Start the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing the webcam: ", err);
    });

// Capture and encode the face on button click
captureBtn.addEventListener('click', () => {
    let canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageDataURL = canvas.toDataURL('image/jpeg');
    const base64Data = imageDataURL.replace(/^data:image\/(png|jpeg);base64,/, '');
    
    // Simulate sending to the backend for processing
    fetch('/api/enroll_face', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image_data: base64Data
        })
    }).then(response => response.json())
      .then(data => {
          if (data.ok) {
              faceDataInput.value = data.face_data;
              alert('Face captured and processed successfully!');
              
              // Draw the captured image to the preview canvas
              let ctx = facePreviewCanvas.getContext('2d');
              let img = new Image();
              img.onload = function() {
                ctx.drawImage(img, 0, 0, facePreviewCanvas.width, facePreviewCanvas.height);
                facePreviewCanvas.style.display = 'block';
              };
              img.src = imageDataURL;
              
          } else {
              alert('Error capturing face: ' + data.detail);
          }
      })
      .catch(err => {
          console.error('API Error:', err);
          alert('Error: Could not connect to face recognition server.');
      });
});
</script>
</body>
</html>
"""

ADMIN_EDIT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<title>Edit Voter</title>
<style>
body{font-family: Arial; max-width: 700px; margin: 20px auto; padding: 15px; background-color: #000; color: #fff; border-radius: 8px; box-shadow: 0 4px 8px rgba(255,255,255,0.1);}
h2{color: #fff; text-align: center;}
p a{color: #3498db;}
form{display: flex; flex-direction: column; gap: 10px;}
label{font-weight: bold;}
input{padding: 10px; border-radius: 5px; border: 1px solid #555; background-color: #333; color: #fff;}
button{padding: 10px; background-color: #28a745; color: white; border: none; cursor: pointer;}
</style>
</head>
<body>
<h2>Admin — Edit Voter</h2>
<p><a href="/admin/list">List voters</a> | <a href="/admin/add">Add voter</a> | <a href="/admin/logout">Logout</a></p>
{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        <ul class="flashes">
        {% for cat, msg in messages %}
            <li class="{{ cat }}">{{ msg }}</li>
        {% endfor %}
        </ul>
    {% endif %}
{% endwith %}
<form method="post">
    <label for="voter_id">Voter QR ID:</label>
    <input type="text" id="voter_id" name="voter_id" value="{{ voter.voter_id }}" readonly>
    <label for="name">Name:</label>
    <input type="text" id="name" name="name" value="{{ voter.name }}" required>
    <label for="dob">DOB (YYYY-MM-DD):</label>
    <input type="date" id="dob" name="dob" value="{{ voter.dob }}" required>
    <label for="phone">Phone:</label>
    <input type="text" id="phone" name="phone" value="{{ voter.phone }}">
    <label for="fingerprint">Fingerprint template (text placeholder):</label>
    <input type="text" id="fingerprint" name="fingerprint" value="{{ voter.fingerprint }}">
    <input type="hidden" name="face_data" value="{{ voter.face_data }}">
    <button type="submit">Update Voter</button>
</form>
</body>
</html>
"""

ADMIN_LIST_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<title>Voters</title>
<style>
body{font-family: Arial; max-width: 900px; margin: 20px auto; padding: 15px; background-color: #000; color: #fff; border-radius: 8px; box-shadow: 0 4px 8px rgba(255,255,255,0.1);}
h2{color: #fff; text-align: center;}
p a{color: #3498db;}
table{width: 100%; border-collapse: collapse; margin-top: 15px;}
th, td{border: 1px solid #555; padding: 8px; text-align: left;}
th{background-color: #333;}
button{padding: 5px 10px; color: white; border: none; border-radius: 4px; cursor: pointer;}
button.edit { background-color: #3498db; }
button.delete { background-color: #dc3545; margin-left: 5px; }
button.reset { background-color: #f39c12; margin-left: 5px; }
.toggle-form { display: inline-flex; align-items: center; gap: 5px; }
</style>
</head>
<body>
<h2>Registered Voters</h2>
<p><a href="/admin/add">Add voter</a> | <a href="/admin/logout">Logout</a></p>
{% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
        <ul class="flashes">
        {% for cat, msg in messages %}
            <li class="{{ cat }}">{{ msg }}</li>
        {% endfor %}
        </ul>
    {% endif %}
{% endwith %}
<table>
<thead>
    <tr>
        <th>ID</th>
        <th>Voter QR ID</th>
        <th>Name</th>
        <th>DOB</th>
        <th>Phone</th>
        <th>Has Voted</th>
        <th>Actions</th>
    </tr>
</thead>
<tbody>
    {% for v in voters %}
    <tr>
        <td>{{ v.id }}</td>
        <td>{{ v.voter_id }}</td>
        <td>{{ v.name }}</td>
        <td>{{ v.dob }}</td>
        <td>{{ v.phone }}</td>
        <td>{{ 'Yes' if v.has_voted else 'No' }}</td>
        <td>
            <button class="edit" onclick="window.location.href='/admin/edit/{{ v.voter_id }}'">Edit</button>
            <form action="/admin/delete/{{ v.voter_id }}" method="post" style="display:inline-block;">
                <button class="delete" type="submit" onclick="return confirm('Are you sure?')">Delete</button>
            </form>
            <form action="/admin/reset_voter/{{ v.voter_id }}" method="post" style="display:inline-block;">
                <button class="reset" type="submit" onclick="return confirm('Are you sure you want to reset the vote status for this voter?')">Reset Vote</button>
            </form>
        </td>
    </tr>
    {% endfor %}
</tbody>
</table>
</body>
</html>
"""

# --------------------
# Voting UI & API
# --------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Voting</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        body {
            font-family: 'Roboto', sans-serif;
            max-width: 900px;
            margin: 20px auto;
            padding: 15px;
            color: #fff;
            line-height: 1.6;
            background-color: #000;
            border-radius: 12px;
        }
        .container {
            background-color: #1a1a1a;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        .container:hover {
            transform: translateY(-5px);
        }
        h1, h2, h3 {
            color: #fff;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 20px;
        }
        h1 {
            font-size: 3em;
            color: #e6e9f0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        p {
            color: #ccc;
        }
        .status-message {
            font-weight: 700;
            margin-top: 10px;
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid transparent;
            transition: all 0.3s ease;
            animation: fade-in 0.5s ease;
        }
        .status-message.success {
            color: #2ecc71;
            background-color: rgba(46, 204, 113, 0.2);
            border-color: #2ecc71;
        }
        .status-message.error {
            color: #e74c3c;
            background-color: rgba(231, 76, 60, 0.2);
            border-color: #e74c3c;
        }
        input, button {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #555;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        input {
            flex-grow: 1;
            margin-right: 10px;
            background-color: #333;
            color: #fff;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            cursor: pointer;
            padding: 12px 25px;
            box-shadow: 0 4px 6px rgba(52, 152, 219, 0.3);
        }
        button:hover {
            background-color: #2980b9;
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 15px rgba(52, 152, 219, 0.4);
        }
        button.candidate {
            background-color: #e74c3c; /* Red for candidates */
        }
        button.candidate:hover {
            background-color: #c0392b;
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 15px rgba(52, 152, 219, 0.4);
        }
        .step-section {
            margin-bottom: 25px;
            padding: 20px;
            border: 1px dashed #555;
            border-radius: 10px;
            transition: opacity 0.5s ease;
        }
        .flex-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #reader {
            width: 100%;
            max-width: 500px;
            margin: 0 auto;
        }
        video, canvas { border: 1px solid white; display: block; margin-top: 10px; width: 100%; max-width: 500px; }
        
        /* Custom Modal CSS */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        .modal-overlay.visible {
            opacity: 1;
            visibility: visible;
        }
        .modal-content {
            background: #1a1a1a;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            max-width: 400px;
            width: 90%;
            transform: translateY(-20px);
            transition: transform 0.3s ease;
        }
        .modal-overlay.visible .modal-content {
            transform: translateY(0);
        }
        .modal-content h3 {
            margin-top: 0;
            border-bottom: none;
        }
        .modal-buttons {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            gap: 15px;
        }
        .modal-buttons button {
            padding: 10px 20px;
            font-weight: bold;
        }
        .modal-buttons button.confirm {
            background-color: #3498db;
        }
        .modal-buttons button.cancel {
            background-color: #e74c3c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Smart Voting Machine</h1>
        <p>Admin: /admin</p>
        <hr>
        <p>Admin should add voters first (with QR ID). You may also type a QR ID directly below for testing.</p>

        <div id="steps">
            <!-- STEP 1: QR ID and Voter Details -->
            <div id="step1" class="step-section">
                <h2>1) Verify QR ID</h2>
                <div id="reader"></div>
                <div class="flex-row mt-4">
                    <input type="text" id="qr-input" placeholder="Enter QR ID manually">
                    <button id="verify-qr-btn">Verify QR</button>
                </div>
                <p id="qr-status" class="status-message" style="display: none;"></p>
            </div>

            <div id="details" class="step-section" style="display: none;">
                <h2>Voter Details</h2>
                <div id="voter-info"></div>
                <button id="start-fp-btn">Proceed to Fingerprint</button>
            </div>

            <!-- STEP 2: Fingerprint (Simulated) -->
            <div id="step2-fp" class="step-section" style="display: none;">
                <h2>2) Fingerprint (Simulated)</h2>
                <p>Please place your finger on the scanner.</p>
                <div class="flex-row">
                    <input type="text" id="fp-input" placeholder="Enter fingerprint payload">
                    <button id="verify-fp-btn">Verify Fingerprint</button>
                </div>
                <p id="fp-status" class="status-message" style="display: none;"></p>
            </div>
            
            <!-- STEP 3: Facial Recognition (CNN) -->
            <div id="step3-face" class="step-section" style="display: none;">
                <h2>3) Facial Verification</h2>
                <p>Look into the camera to verify your identity.</p>
                <video id="face-webcam" width="320" height="240" autoplay></video>
                <button id="verify-face-btn">Verify Face</button>
                <p id="face-status" class="status-message" style="display: none;"></p>
            </div>

            <!-- STEP 4: Select Candidate -->
            <div id="voting" class="step-section" style="display: none;">
                <h2>4) Select Candidate</h2>
                <p>Please select your preferred candidate.</p>
                <div class="space-y-4">
                    <button class="w-full candidate" data-name="Candidate A">Candidate A</button>
                    <button class="w-full candidate" data-name="Candidate B">Candidate B</button>
                    <button class="w-full candidate" data-name="Candidate C">Candidate C</button>
                </div>
                <p id="vote-status" class="status-message" style="display: none;"></p>
            </div>
        </div>
    </div>
    
    <!-- Custom Modal for Confirmations -->
    <div id="custom-modal-overlay" class="modal-overlay">
        <div class="modal-content">
            <h3 id="modal-message"></h3>
            <div class="modal-buttons">
                <button id="modal-confirm-btn" class="confirm">Confirm</button>
                <button id="modal-cancel-btn" class="cancel">Cancel</button>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/html5-qrcode"></script>
    <script>
    // Basic flow variables
    let currentVoter = null;
    let html5QrcodeScanner = null;
    let faceVideo = document.getElementById('face-webcam');
    
    // Helper to set status message with style
    function setStatus(elementId, message, type = 'info') {
        const el = document.getElementById(elementId);
        el.textContent = message;
        el.className = 'status-message ' + type;
        el.style.display = 'block';
    }

    // Custom modal functions
    const modalOverlay = document.getElementById('custom-modal-overlay');
    const modalMessageEl = document.getElementById('modal-message');
    const modalConfirmBtn = document.getElementById('modal-confirm-btn');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');

    function showConfirmModal(message, onConfirmCallback) {
        modalMessageEl.textContent = message;
        modalOverlay.classList.add('visible');
        
        // Clear previous event listeners
        modalConfirmBtn.onclick = null;
        modalCancelBtn.onclick = null;

        modalConfirmBtn.onclick = () => {
            onConfirmCallback();
            modalOverlay.classList.remove('visible');
        };
        modalCancelBtn.onclick = () => {
            modalOverlay.classList.remove('visible');
        };
    }

    // Function to handle QR scan success
    function onScanSuccess(decodedText, decodedResult) {
        if (html5QrcodeScanner) {
            html5QrcodeScanner.clear();
        }
        document.getElementById('qr-input').value = decodedText;
        verifyQrCode(decodedText);
    }

    // Function to handle QR scan failure
    function onScanFailure(error) {
        console.warn(`Code scan error = ${error}`);
    }

    // Initialize QR code scanner
    function startQrScanner() {
        html5QrcodeScanner = new Html5QrcodeScanner("reader", {
            fps: 10,
            qrbox: { width: 250, height: 250 }
        }, false);
        html5QrcodeScanner.render(onScanSuccess, onScanFailure);
    }

    window.addEventListener('load', startQrScanner);

    async function verifyQrCode(qr) {
        if (!qr) { setStatus('qr-status', 'Error: Enter a QR ID.', 'error'); return; }
        const res = await fetch('/api/verify_qr', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({voter_id: qr})});
        const j = await res.json();
        if (j.ok) {
            currentVoter = j.voter;
            setStatus('qr-status', `QR OK. Welcome, ${currentVoter.name}.`, 'success');
            document.getElementById('voter-info').innerHTML = `Name: ${currentVoter.name}<br>DOB: ${currentVoter.dob}<br>Phone: ${currentVoter.phone}`;
            document.getElementById('step1').style.display = 'none';
            document.getElementById('details').style.display = 'block';
        } else {
            setStatus('qr-status', 'Error: ' + (j.detail || 'unknown error'), 'error');
            setTimeout(() => { window.location.reload(); }, 3000);
        }
    }

    // Verify QR ID by server for manual input
    document.getElementById('verify-qr-btn').onclick = () => {
        const qr = document.getElementById('qr-input').value.trim();
        if (html5QrcodeScanner) {
            html5QrcodeScanner.clear();
        }
        verifyQrCode(qr);
    }

    // Fingerprint stage (simulated string match)
    document.getElementById('start-fp-btn').onclick = () => {
        document.getElementById('details').style.display = 'none';
        document.getElementById('step2-fp').style.display = 'block';
    }

    document.getElementById('verify-fp-btn').onclick = async () => {
        const payload = document.getElementById('fp-input').value.trim();
        if (!payload) { setStatus('fp-status', 'Error: Enter fingerprint payload.', 'error'); return; }
        const res = await fetch('/api/verify_fingerprint', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({voter_id: currentVoter.voter_id, fp_payload: payload})});
        const j = await res.json();
        if (j.ok) {
            setStatus('fp-status', 'Fingerprint OK!', 'success');
            document.getElementById('step2-fp').style.display = 'none';
            document.getElementById('step3-face').style.display = 'block';
            
            // Start the face webcam
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    faceVideo.srcObject = stream;
                })
                .catch(err => {
                    console.error("Error accessing the webcam: ", err);
                    setStatus('face-status', 'Error accessing webcam for facial recognition.', 'error');
                });

        } else {
            setStatus('fp-status', 'Fingerprint verification failed.', 'error');
            setTimeout(() => { window.location.reload(); }, 3000);
        }
    }

    // Facial verification stage
    document.getElementById('verify-face-btn').onclick = async () => {
        const canvas = document.createElement('canvas');
        canvas.width = faceVideo.videoWidth;
        canvas.height = faceVideo.videoHeight;
        canvas.getContext('2d').drawImage(faceVideo, 0, 0, canvas.width, canvas.height);
        
        const imageDataURL = canvas.toDataURL('image/jpeg');
        const base64Data = imageDataURL.replace(/^data:image\/(png|jpeg);base64,/, '');

        const res = await fetch('/api/verify_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voter_id: currentVoter.voter_id, image_data: base64Data })
        });
        const j = await res.json();
        
        if (j.ok) {
            setStatus('face-status', 'Face OK! You are verified.', 'success');
            document.getElementById('step3-face').style.display = 'none';
            document.getElementById('voting').style.display = 'block';
            faceVideo.srcObject.getTracks().forEach(track => track.stop()); // Stop the camera
        } else {
            setStatus('face-status', 'Face verification failed.', 'error');
            setTimeout(() => { window.location.reload(); }, 3000);
        }
    };

    // Voting
    Array.from(document.getElementsByClassName('candidate')).forEach(btn => {
        btn.onclick = () => {
            showConfirmModal(`Confirm vote for ${btn.dataset.name}?`, async () => {
                const res = await fetch('/api/cast_vote', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({voter_id: currentVoter.voter_id, candidate: btn.dataset.name})});
                const j = await res.json();
                if (j.ok) {
                    setStatus('vote-status', `Vote for ${btn.dataset.name} has been cast!`, 'success');
                    setTimeout(() => { window.location.reload(); }, 3000); // Reload after 3s
                } else {
                    setStatus('vote-status', 'Error: ' + (j.error || j.detail || 'unknown'), 'error');
                    // Auto-refresh on error
                    setTimeout(() => { window.location.reload(); }, 3000);
                }
            });
        }
    });
    </script>
</body>
</html>
"""

@APP.route("/")
def index():
    """Renders the main voting machine UI."""
    return render_template_string(INDEX_HTML)

# --- API endpoints used by frontend ---
@APP.route("/api/verify_qr", methods=["POST"])
def api_verify_qr():
    """API endpoint for QR code verification."""
    data = request.get_json(force=True)
    voter_id = data.get("voter_id", "").strip()
    if not voter_id:
        return jsonify(ok=False, error="missing_voter_id"), 400
    conn = get_conn()
    r = conn.execute("SELECT * FROM voters WHERE voter_id=?", (voter_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify(ok=False, error="not_registered", detail="Contact Admin or NOT Registered"), 404
    age = calculate_age(r["dob"])
    if age < 18:
        return jsonify(ok=False, error="underage", detail="BABY Your Not Eligible For Vote"), 403
    voter = {"voter_id": r["voter_id"], "name": r["name"], "dob": r["dob"], "phone": r["phone"], "has_voted": bool(r["has_voted"])}
    return jsonify(ok=True, voter=voter)

@APP.route("/api/verify_fingerprint", methods=["POST"])
def api_verify_fingerprint():
    """API endpoint for fingerprint verification."""
    data = request.get_json(force=True)
    voter_id = data.get("voter_id", "").strip()
    fp_payload = data.get("fp_payload")
    if not voter_id or fp_payload is None:
        return jsonify(ok=False, error="missing_data"), 400
    conn = get_conn()
    r = conn.execute("SELECT fingerprint FROM voters WHERE voter_id=?", (voter_id,)).fetchone()
    conn.close()
    if not r:
        return jsonify(ok=False, error="voter_not_found"), 404
    stored = r["fingerprint"] or ""
    ok = (fp_payload == stored)
    return jsonify(ok=bool(ok))

@APP.route("/api/enroll_face", methods=["POST"])
def api_enroll_face():
    """API endpoint to process and enroll a face image from the admin UI."""
    data = request.get_json(force=True)
    image_data = data.get("image_data")
    if not image_data:
        return jsonify(ok=False, detail="No image data provided."), 400
    
    face_data = get_face_data(image_data)
    if face_data:
        # Return the serialized data to the frontend to be saved with the form
        return jsonify(ok=True, face_data=base64.b64encode(face_data).decode('utf-8'))
    else:
        return jsonify(ok=False, detail="No face detected in the image."), 400

@APP.route("/api/verify_face", methods=["POST"])
def api_verify_face():
    """API endpoint to verify a voter's face against the stored data."""
    data = request.get_json(force=True)
    voter_id = data.get("voter_id")
    image_data = data.get("image_data")
    
    if not voter_id or not image_data:
        return jsonify(ok=False, detail="Missing voter ID or image data."), 400
    
    conn = get_conn()
    r = conn.execute("SELECT face_data, has_voted FROM voters WHERE voter_id=?", (voter_id,)).fetchone()
    conn.close()
    
    if not r:
        return jsonify(ok=False, detail="Voter not found."), 404
    
    if r["has_voted"]:
        return jsonify(ok=False, detail="This voter has already voted."), 403
    
    stored_face_data = r["face_data"]
    if not stored_face_data:
        return jsonify(ok=False, detail="No face data enrolled for this voter."), 404
    
    try:
        live_face_data = get_face_data(image_data)
        if not live_face_data:
            return jsonify(ok=False, detail="No face detected in the live image."), 400
        
        # Need to decode the stored base64 data first before passing to compare_faces
        stored_face_bytes = base64.b64decode(stored_face_data)
        
        if compare_faces(stored_face_bytes, live_face_data):
            logging.info(f"Voter {voter_id} facial verification successful.")
            return jsonify(ok=True)
        else:
            logging.warning(f"Voter {voter_id} facial verification failed.")
            return jsonify(ok=False, detail="Facial recognition failed. Please try again or contact an administrator.")
    except Exception as e:
        logging.error(f"Error during face verification for voter {voter_id}: {e}")
        return jsonify(ok=False, detail="An error occurred during facial verification."), 500

@APP.route("/api/cast_vote", methods=["POST"])
def api_cast_vote():
    """API endpoint to cast a vote."""
    data = request.get_json(force=True)
    voter_id = data.get("voter_id")
    candidate = data.get("candidate")
    
    if not voter_id or not candidate:
        return jsonify(ok=False, detail="Missing voter ID or candidate."), 400
        
    conn = get_conn()
    try:
        # Check if voter has already voted
        voter_row = conn.execute("SELECT has_voted FROM voters WHERE voter_id=?", (voter_id,)).fetchone()
        if not voter_row:
            return jsonify(ok=False, detail="Voter not found."), 404
        if voter_row["has_voted"]:
            return jsonify(ok=False, detail="Voter has already cast their vote."), 403

        # Record the vote
        conn.execute("INSERT INTO votes (voter_id, candidate, timestamp) VALUES (?, ?, ?)",
                    (voter_id, candidate, datetime.utcnow().isoformat()))
        
        # Update voter status
        conn.execute("UPDATE voters SET has_voted=1 WHERE voter_id=?", (voter_id,))
        
        conn.commit()
        logging.info(f"Voter {voter_id} cast a vote for {candidate}.")
        return jsonify(ok=True)
    except sqlite3.Error as e:
        conn.rollback()
        logging.error(f"Database error casting vote for {voter_id}: {e}")
        return jsonify(ok=False, detail="Database error."), 500
    finally:
        conn.close()

# --- Admin Routes ---

def require_admin(f):
    """Decorator to protect admin routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("logged_in"):
            return f(*args, **kwargs)
        flash("Please log in to access this page.", "error")
        return redirect(url_for("admin_login"))
    return decorated_function

@APP.route("/admin", methods=["GET", "POST"])
@APP.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["logged_in"] = True
            flash("Logged in successfully.", "success")
            return redirect(url_for("admin_list_voters"))
        else:
            flash("Invalid credentials.", "error")
    return render_template_string(ADMIN_LOGIN_HTML)

@APP.route("/admin/logout")
def admin_logout():
    session.pop("logged_in", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("admin_login"))

@APP.route("/admin/list")
@require_admin
def admin_list_voters():
    conn = get_conn()
    voters = conn.execute("SELECT * FROM voters ORDER BY created_at DESC").fetchall()
    conn.close()
    return render_template_string(ADMIN_LIST_HTML, voters=voters)

@APP.route("/admin/add", methods=["GET", "POST"])
@require_admin
def admin_add_voter():
    if request.method == "POST":
        voter_id = request.form.get("voter_id")
        name = request.form.get("name")
        dob = request.form.get("dob")
        phone = request.form.get("phone")
        fingerprint = request.form.get("fingerprint")
        face_data = request.form.get("face_data")
        created_at = datetime.utcnow().isoformat()
        
        if not voter_id or not name or not dob:
            flash("Voter ID, Name, and DOB are required.", "error")
            return redirect(url_for("admin_add_voter"))
        
        conn = get_conn()
        try:
            conn.execute("INSERT INTO voters (voter_id, name, dob, phone, fingerprint, face_data, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (voter_id, name, dob, phone, fingerprint, face_data, created_at))
            conn.commit()
            flash(f"Voter {voter_id} added successfully.", "success")
        except sqlite3.IntegrityError:
            flash(f"Voter ID '{voter_id}' already exists.", "error")
            conn.rollback()
        finally:
            conn.close()
            
        return redirect(url_for("admin_add_voter"))
    
    return render_template_string(ADMIN_ADD_HTML)

@APP.route("/admin/edit/<voter_id>", methods=["GET", "POST"])
@require_admin
def admin_edit_voter(voter_id):
    conn = get_conn()
    voter = conn.execute("SELECT * FROM voters WHERE voter_id=?", (voter_id,)).fetchone()
    
    if not voter:
        flash(f"Voter '{voter_id}' not found.", "error")
        conn.close()
        return redirect(url_for("admin_list_voters"))
        
    if request.method == "POST":
        name = request.form.get("name")
        dob = request.form.get("dob")
        phone = request.form.get("phone")
        fingerprint = request.form.get("fingerprint")
        face_data = request.form.get("face_data") # Retain existing face_data
        
        if not name or not dob:
            flash("Name and DOB are required.", "error")
        else:
            conn.execute("UPDATE voters SET name=?, dob=?, phone=?, fingerprint=?, face_data=? WHERE voter_id=?",
                         (name, dob, phone, fingerprint, face_data, voter_id))
            conn.commit()
            flash(f"Voter '{voter_id}' updated successfully.", "success")
            return redirect(url_for("admin_list_voters"))
    
    conn.close()
    return render_template_string(ADMIN_EDIT_HTML, voter=voter)

@APP.route("/admin/delete/<voter_id>", methods=["POST"])
@require_admin
def admin_delete_voter(voter_id):
    conn = get_conn()
    conn.execute("DELETE FROM voters WHERE voter_id=?", (voter_id,))
    conn.commit()
    conn.close()
    flash(f"Voter '{voter_id}' deleted successfully.", "success")
    return redirect(url_for("admin_list_voters"))

@APP.route("/admin/reset_voter/<voter_id>", methods=["POST"])
@require_admin
def admin_reset_voter(voter_id):
    """Admin route to reset a voter's has_voted status."""
    conn = get_conn()
    try:
        conn.execute("UPDATE voters SET has_voted = 0 WHERE voter_id = ?", (voter_id,))
        conn.commit()
        flash(f"Vote status for '{voter_id}' has been reset.", "success")
    except sqlite3.Error as e:
        conn.rollback()
        flash(f"Error resetting vote status: {e}", "error")
    finally:
        conn.close()
    return redirect(url_for("admin_list_voters"))


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
