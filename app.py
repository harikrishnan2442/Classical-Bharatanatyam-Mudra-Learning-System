from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for webcam access

# ============================================
# LOAD MODELS (ONCE AT STARTUP)
# ============================================
print("Loading models...")
try:
    model = joblib.load('models/mudra_classifier.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    reference_mudras = joblib.load('models/reference_mudras.pkl')
    print(f"✓ Models loaded successfully")
    print(f"✓ Available mudras: {len(label_encoder.classes_)}")
except Exception as e:
    print(f"ERROR: Could not load models - {e}")
    print("Make sure .pkl files are in the 'models/' folder")
    exit(1)

# ============================================
# INITIALIZE MEDIAPIPE
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ============================================
# NORMALIZATION FUNCTION (SAME AS PHASE 1)
# ============================================
def normalize_hand_landmarks(landmarks):
    """Normalize hand landmarks to be translation/scale invariant."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Center on wrist
    wrist = coords[0]
    centered = coords - wrist
    
    # Normalize scale
    middle_finger_base = centered[9]
    scale = np.linalg.norm(middle_finger_base)
    
    if scale < 1e-6:
        return None
    
    normalized = centered / scale
    return normalized.flatten()

# ============================================
# FEEDBACK CALCULATION
# ============================================
def calculate_joint_errors(user_vector, reference_vector):
    """Calculate error for each of the 21 joints."""
    user_coords = user_vector.reshape(21, 3)
    ref_coords = reference_vector.reshape(21, 3)
    
    errors = np.linalg.norm(user_coords - ref_coords, axis=1)
    return errors.tolist()

def get_joint_name(idx):
    """Get human-readable name for joint index."""
    joint_names = [
        "wrist", 
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    return joint_names[idx] if idx < len(joint_names) else f"joint_{idx}"

def generate_feedback(joint_idx, error):
    """Generate actionable feedback message."""
    joint_name = get_joint_name(joint_idx)
    
    # Friendly names
    friendly_names = {
        'thumb_tip': 'thumb tip',
        'index_tip': 'index finger tip',
        'middle_tip': 'middle finger tip',
        'ring_tip': 'ring finger tip',
        'pinky_tip': 'little finger tip',
        'thumb_mcp': 'thumb base',
        'index_mcp': 'index finger base',
        'middle_mcp': 'middle finger base',
        'ring_mcp': 'ring finger base',
        'pinky_mcp': 'little finger base'
    }
    
    display_name = friendly_names.get(joint_name, joint_name.replace('_', ' '))
    
    if error > 0.15:
        return f"Your {display_name} is significantly off position"
    elif error > 0.08:
        return f"Your {display_name} needs slight adjustment"
    else:
        return f"Your {display_name} is correct"

# ============================================
# API ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html', 
                         mudras=label_encoder.classes_.tolist())

@app.route('/api/mudras', methods=['GET'])
def get_mudras():
    """Return list of all available mudras."""
    return jsonify({
        'mudras': label_encoder.classes_.tolist(),
        'total': len(label_encoder.classes_)
    })

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Process a single frame from the webcam.
    Expects: base64-encoded image
    Returns: prediction, confidence, feedback
    """
    try:
        data = request.json
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return jsonify({
                'success': False,
                'message': 'No hand detected'
            })
        
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Normalize
        normalized = normalize_hand_landmarks(hand_landmarks.landmark)
        
        if normalized is None:
            return jsonify({
                'success': False,
                'message': 'Could not normalize hand'
            })
        
        # Predict mudra
        prediction = model.predict([normalized])[0]
        predicted_mudra = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        try:
            proba = model.predict_proba([normalized])[0]
            confidence = float(proba[prediction])
        except:
            confidence = 1.0
        
        # Extract landmarks for frontend visualization
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })
        
        # Prepare response
        response = {
            'success': True,
            'predicted_mudra': predicted_mudra,
            'confidence': confidence,
            'landmarks': landmarks
        }
        
        # If target mudra is specified, calculate feedback
        target_mudra = data.get('target_mudra')
        if target_mudra and target_mudra in reference_mudras:
            reference = reference_mudras[target_mudra]
            errors = calculate_joint_errors(normalized, reference)
            
            # Find worst joint
            worst_joint_idx = int(np.argmax(errors))
            worst_error = errors[worst_joint_idx]
            
            # Calculate overall accuracy
            avg_error = np.mean(errors)
            accuracy = max(0, min(100, 100 * (1 - avg_error / 0.2)))
            
            # Generate feedback
            feedback_message = generate_feedback(worst_joint_idx, worst_error)
            
            response.update({
                'target_mudra': target_mudra,
                'accuracy': accuracy,
                'worst_joint_idx': worst_joint_idx,
                'worst_joint_error': worst_error,
                'all_errors': errors,
                'feedback': feedback_message,
                'is_correct': accuracy > 90
            })
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/get_reference/<mudra_name>', methods=['GET'])
def get_reference(mudra_name):
    """Get the reference (perfect) landmarks for a specific mudra."""
    if mudra_name not in reference_mudras:
        return jsonify({
            'success': False,
            'message': 'Mudra not found'
        }), 404
    
    reference = reference_mudras[mudra_name]
    
    # Reshape to 21x3 for frontend
    coords = reference.reshape(21, 3)
    landmarks = []
    for coord in coords:
        landmarks.append({
            'x': float(coord[0]),
            'y': float(coord[1]),
            'z': float(coord[2])
        })
    
    return jsonify({
        'success': True,
        'mudra': mudra_name,
        'landmarks': landmarks
    })

@app.route('/api/mudra_info/<mudra_name>', methods=['GET'])
def get_mudra_info(mudra_name):
    """Get detailed information about a specific mudra."""
    # Comprehensive mudra database
    mudra_database = {
        'Alapadmam': {
            'category': 'Samyuta Hasta (Double Hand)',
            'meaning': 'Blooming Lotus',
            'description': 'Alapadma represents a fully bloomed lotus flower, symbolizing purity, divine beauty, and spiritual awakening.',
            'significance': 'The lotus represents beauty and spiritual awakening in Indian culture.',
            'usage': 'Used to show offering flowers, describing gardens, representing goddesses.',
            'history': 'The lotus has been sacred in Indian tradition for over 3000 years.'
        },
        'Anjali': {
            'category': 'Samyuta Hasta (Double Hand)',
            'meaning': 'Salutation',
            'description': 'Formed by joining both palms together in prayer position.',
            'significance': 'Universal gesture of respect, representing namaste and divine recognition.',
            'usage': 'Used for greeting deities, paying respects, expressing devotion.',
            'history': 'Dating back to Vedic period, appears in temple sculptures from 5th century CE.'
        },
        # Add more as needed
    }
    
    # Clean mudra name
    clean_name = mudra_name.replace('(1)', '').strip()
    
    if clean_name in mudra_database:
        return jsonify({
            'success': True,
            'mudra': clean_name,
            'info': mudra_database[clean_name]
        })
    else:
        return jsonify({
            'success': True,
            'mudra': mudra_name,
            'info': {
                'category': 'Traditional Bharatanatyam Mudra',
                'meaning': 'Classical hand gesture',
                'description': f'{mudra_name} is a traditional mudra used in Bharatanatyam dance.',
                'significance': 'This mudra holds deep cultural significance in Indian classical dance.',
                'usage': 'Used in various dance sequences to convey specific meanings and emotions.',
                'history': 'Part of the ancient tradition preserved in Natyashastra texts.'
            }
        })

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("MUDRA GURU - WEB APPLICATION")
    print("="*60)
    print("✓ Models loaded successfully")
    print(f"✓ Available mudras: {len(label_encoder.classes_)}")
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)