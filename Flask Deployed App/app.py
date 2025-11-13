import os
import json
from flask import Flask, redirect, render_template, request, session, url_for, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import CNN
import numpy as np
import torch
import pandas as pd
from chatbot import get_chatbot

# Try to import dynamic class mapping loader
try:
    from load_class_mappings import get_class_mappings
    USE_DYNAMIC_CLASSES = True
except ImportError:
    USE_DYNAMIC_CLASSES = False
    print("Warning: load_class_mappings not available, using hardcoded 39 classes")

# Get the base directory of the Flask app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from Flask Deployed App)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

disease_info = pd.read_csv(os.path.join(BASE_DIR, 'disease_info.csv'), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(BASE_DIR, 'supplement_info.csv'), encoding='cp1252')
crop_info = pd.read_csv(os.path.join(BASE_DIR, 'crop_info.csv'), encoding='utf-8')

# Load translations
translations_path = os.path.join(BASE_DIR, 'translations.json')
with open(translations_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    'te': 'తెలుగు (Telugu)',
    'ta': 'தமிழ் (Tamil)',
    'mr': 'मराठी (Marathi)',
    'gu': 'ગુજરાતી (Gujarati)',
    'kn': 'ಕನ್ನಡ (Kannada)',
    'ml': 'മലയാളം (Malayalam)',
    'pa': 'ਪੰਜਾਬੀ (Punjabi)',
    'or': 'ଓଡ଼ିଆ (Odia)',
    'ur': 'اردو (Urdu)'
}
DEFAULT_LANGUAGE = 'en'

# Load class mappings dynamically or use default
if USE_DYNAMIC_CLASSES:
    try:
        class_mappings = get_class_mappings(prefer_dataset=False)  # Prefer JSON over dataset for app
        num_classes = class_mappings['num_classes']
        print(f"Loaded class mappings dynamically: {num_classes} classes")
    except Exception as e:
        print(f"Warning: Could not load dynamic class mappings: {e}")
        print("Falling back to default 39 classes")
        num_classes = 39
else:
    num_classes = 39
    print(f"Using hardcoded number of classes: {num_classes}")

# Load model and set to CPU mode - use path from Model folder
model_path = os.path.join(PROJECT_ROOT, 'Model', 'plant_disease_model_1_latest.pt')
model_path = os.path.normpath(model_path)  # Normalize the path
print(f"Loading model from: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    print(f"Model file size: {file_size:.2f} MB")
else:
    print(f"ERROR: Model file not found at: {model_path}")

# Create model with dynamically detected number of classes
model = CNN.CNN(num_classes)
print(f"Created model with {num_classes} output classes")
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    
    # Verify model is in eval mode and check a sample weight
    print(f"Model in eval mode: {not model.training}")
    # Check if model weights are non-zero (basic sanity check)
    first_weight = next(model.parameters())
    print(f"First layer weight stats - Mean: {first_weight.mean().item():.6f}, Std: {first_weight.std().item():.6f}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\n⚠️ WARNING: Model file may be missing or corrupted!")
    print("Please download the trained model from:")
    print("https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link")
    print("And place it in the 'Flask Deployed App' folder as 'plant_disease_model_1_latest.pt'")
    raise

# Match the training preprocessing: Resize(255) -> CenterCrop(224) -> ToTensor()
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def prediction(image_path):
    try:
        # Open and convert to RGB if needed
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply the same preprocessing as training
        input_data = transform(image)
        input_data = input_data.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(input_data)
            # Apply softmax to get probabilities
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.detach().numpy()
        
        # Get prediction index and confidence
        index = np.argmax(output)
        confidence = float(np.max(output))
        
        # Debug output
        print(f"Image: {image_path}")
        print(f"Predicted class index: {index}")
        print(f"Confidence: {confidence:.4f}")
        top3_indices = np.argsort(output[0])[-3:][::-1]
        top3_probs = np.sort(output[0])[-3:][::-1]
        print(f"Top 3 predictions: {top3_indices}")
        print(f"Top 3 probabilities: {top3_probs}")
        
        return index
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Initialize chatbot
chatbot = get_chatbot()

def get_language():
    """Get current language from session or default to English."""
    return session.get('language', DEFAULT_LANGUAGE)

def get_translations():
    """Get translations for current language."""
    lang = get_language()
    return translations.get(lang, translations[DEFAULT_LANGUAGE])

@app.route('/set_language/<lang>')
def set_language(lang):
    """Set language and redirect back to referring page."""
    if lang in SUPPORTED_LANGUAGES:
        session['language'] = lang
    referer = request.referrer or url_for('home_page')
    return redirect(referer)

@app.route('/')
def home_page():
    lang = get_language()
    t = get_translations()
    return render_template('home.html', t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/contact')
def contact():
    t = get_translations()
    lang = get_language()
    return render_template('contact-us.html', t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/index')
def ai_engine_page():
    t = get_translations()
    lang = get_language()
    return render_template('index.html', t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/mobile-device')
def mobile_device_detected_page():
    t = get_translations()
    lang = get_language()
    return render_template('mobile-device.html', t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        upload_dir = os.path.join(BASE_DIR, 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)  # Ensure upload directory exists
        file_path = os.path.join(upload_dir, filename)
        image.save(file_path)
        print(f"Image saved to: {file_path}")
        pred = prediction(file_path)
        print(f"Final prediction index: {pred}")
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        t = get_translations()
        lang = get_language()
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link,
                               t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/market', methods=['GET', 'POST'])
def market():
    t = get_translations()
    lang = get_language()
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']),
                           t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/crop/<crop_name>')
def crop_details(crop_name):
    """Display detailed information about a specific crop."""
    try:
        t = get_translations()
        lang = get_language()
        # Decode URL-encoded crop name (spaces, special characters)
        from urllib.parse import unquote
        crop_name_decoded = unquote(crop_name).replace('+', ' ').replace('-', ' ')
        
        # Find crop information (try exact match first, then partial match)
        crop_data = crop_info[crop_info['crop_name'].str.lower() == crop_name_decoded.lower()]
        
        # If not found, try partial match
        if crop_data.empty:
            crop_data = crop_info[crop_info['crop_name'].str.lower().str.contains(crop_name_decoded.lower(), na=False)]
        
        if crop_data.empty:
            # If crop not found, return error
            return render_template('crop_details.html', crop=None, error=f"Crop '{crop_name_decoded}' not found",
                                 t=t, lang=lang, languages=SUPPORTED_LANGUAGES)
        
        # Get the first matching crop
        crop = crop_data.iloc[0].to_dict()
        
        # Get diseases for this crop from disease_info
        # Extract crop name from disease names (e.g., "Apple : Scab" -> "Apple")
        crop_name_for_disease = crop['crop_name']
        crop_diseases = disease_info[disease_info['disease_name'].str.contains(crop_name_for_disease, case=False, na=False)]
        
        return render_template('crop_details.html', crop=crop, crop_diseases=crop_diseases.to_dict('records'),
                             t=t, lang=lang, languages=SUPPORTED_LANGUAGES)
    except Exception as e:
        print(f"Error loading crop details: {e}")
        import traceback
        traceback.print_exc()
        t = get_translations()
        lang = get_language()
        return render_template('crop_details.html', crop=None, error=str(e),
                             t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    """Chatbot API endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        lang = get_language()
        
        if not query:
            return jsonify({
                'success': False,
                'response': 'Please ask a question.'
            }), 400
        
        # Get response from chatbot
        result = chatbot.generate_response(query, lang)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'type': result.get('type', 'default'),
            'data': result.get('data', {})
        })
    except Exception as e:
        print(f"Chatbot error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'response': 'Sorry, I encountered an error. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
