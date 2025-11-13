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
    # Prepare comprehensive test images mapping
    test_images_mapping = {
        # Apple
        'Apple_scab.JPG': {'name': 'Apple : Scab', 'crop': 'Apple'},
        'apple_black_rot.JPG': {'name': 'Apple : Black Rot', 'crop': 'Apple'},
        'Apple_ceder_apple_rust.JPG': {'name': 'Apple : Cedar rust', 'crop': 'Apple'},
        'apple_cedar_rust.JPG': {'name': 'Apple : Cedar rust', 'crop': 'Apple'},  # Alternative filename
        'apple_healthy.JPG': {'name': 'Apple : Healthy', 'crop': 'Apple'},
        # Background
        'background_without_leaves.jpg': {'name': 'Background Without Leaves', 'crop': 'Background'},
        # Blueberry
        'blueberry_healthy.JPG': {'name': 'Blueberry : Healthy', 'crop': 'Blueberry'},
        # Cherry
        'cherry_healthy.JPG': {'name': 'Cherry : Healthy', 'crop': 'Cherry'},
        'cherry_powdery_mildew.JPG': {'name': 'Cherry : Powdery Mildew', 'crop': 'Cherry'},
        # Corn
        'corn_cercospora_leaf.JPG': {'name': 'Corn : Cercospora Leaf Spot | Gray Leaf Spot', 'crop': 'Corn'},
        'corn_common_rust.JPG': {'name': 'Corn : Common Rust', 'crop': 'Corn'},
        'corn_healthy.jpg': {'name': 'Corn : Healthy', 'crop': 'Corn'},
        'corn_northen_leaf_blight.JPG': {'name': 'Corn : Northern Leaf Blight', 'crop': 'Corn'},
        # Squash variations
        'squash_powdery_mildew_1.JPG': {'name': 'Squash : Powdery Mildew', 'crop': 'Squash'},
        # Grape
        'grape_black_rot.JPG': {'name': 'Grape : Black Rot', 'crop': 'Grape'},
        'Grape_esca.JPG': {'name': 'Grape : Esca | Black Measles', 'crop': 'Grape'},
        'grape_healthy.JPG': {'name': 'Grape : Healthy', 'crop': 'Grape'},
        'grape_leaf_blight.JPG': {'name': 'Grape : Leaf Blight | Isariopsis Leaf Spot', 'crop': 'Grape'},
        # Orange
        'orange_haunglongbing.JPG': {'name': 'Orange : Haunglongbing | Citrus Greening', 'crop': 'Orange'},
        # Peach
        'peach_bacterial_spot.JPG': {'name': 'Peach : Bacterial spot', 'crop': 'Peach'},
        'peach_healthy.JPG': {'name': 'Peach : Healthy', 'crop': 'Peach'},
        # Pepper
        'pepper_bacterial_spot.JPG': {'name': 'Pepper, bell : Bacterial spot', 'crop': 'Pepper Bell'},
        'pepper_bell_healthy.JPG': {'name': 'Pepper, bell : Healthy', 'crop': 'Pepper Bell'},
        # Potato
        'potato_early_blight.JPG': {'name': 'Potato : Early Blight', 'crop': 'Potato'},
        'potato_healthy.JPG': {'name': 'Potato : Healthy', 'crop': 'Potato'},
        'potato_late_blight.JPG': {'name': 'Potato : Late Blight', 'crop': 'Potato'},
        # Raspberry
        'raspberry_healthy.JPG': {'name': 'Raspberry : Healthy', 'crop': 'Raspberry'},
        # Soybean
        'soyaben healthy.JPG': {'name': 'Soybean : Healthy', 'crop': 'Soybean'},
        # Squash
        'squash_powdery_mildew.JPG': {'name': 'Squash : Powdery Mildew', 'crop': 'Squash'},
        # Strawberry
        'starwberry_healthy.JPG': {'name': 'Strawberry : Healthy', 'crop': 'Strawberry'},
        'starwberry_leaf_scorch.JPG': {'name': 'Strawberry : Leaf Scorch', 'crop': 'Strawberry'},
        # Tomato
        'tomato_bacterial_spot.JPG': {'name': 'Tomato : Bacterial spot', 'crop': 'Tomato'},
        'tomato-bacterial-spot2.jpg': {'name': 'Tomato : Bacterial spot', 'crop': 'Tomato'},
        'tomato_early_blight.JPG': {'name': 'Tomato : Early Blight', 'crop': 'Tomato'},
        'tomato_healthy.JPG': {'name': 'Tomato : Healthy', 'crop': 'Tomato'},
        'tomato_late_blight.JPG': {'name': 'Tomato : Late Blight', 'crop': 'Tomato'},
        'tomato_leaf_mold.JPG': {'name': 'Tomato : Leaf Mold', 'crop': 'Tomato'},
        'tomato_mosaic_virus.JPG': {'name': 'Tomato : Tomato mosaic virus', 'crop': 'Tomato'},
        'tomato_septoria_leaf_spot.JPG': {'name': 'Tomato : Septoria leaf spot', 'crop': 'Tomato'},
        'tomato_spider_mites_two_spotted_spider_mites.JPG': {'name': 'Tomato : Spider mites Two-spotted spider mite', 'crop': 'Tomato'},
        'tomato_target_spot.JPG': {'name': 'Tomato : Target Spot', 'crop': 'Tomato'},
        'tomato_yellow_leaf_curl_virus.JPG': {'name': 'Tomato : Tomato Yellow Leaf Curl Virus', 'crop': 'Tomato'},
        'tomato_yellow_leaf_curl_virus2.jpg': {'name': 'Tomato : Tomato Yellow Leaf Curl Virus', 'crop': 'Tomato'},
        'tomato-leaf-curl-virus3.jpg': {'name': 'Tomato : Tomato Yellow Leaf Curl Virus', 'crop': 'Tomato'},
        'tomato-mold.jpg': {'name': 'Tomato : Leaf Mold', 'crop': 'Tomato'},
    }
    
    # Get test images directory
    test_images_dir = os.path.join(BASE_DIR, 'static', 'test_images')
    test_images = []
    
    # Check if test_images directory exists and scan for images
    if os.path.exists(test_images_dir):
        for filename in os.listdir(test_images_dir):
            # Skip non-image files
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG')):
                continue
            
            img_path = os.path.join(test_images_dir, filename)
            if os.path.isfile(img_path):
                # Check if we have mapping for this file
                if filename in test_images_mapping:
                    img_info = test_images_mapping[filename].copy()
                    img_info['filename'] = filename
                    img_info['path'] = img_path
                    test_images.append(img_info)
                else:
                    # Try to infer from filename if not in mapping
                    filename_lower = filename.lower()
                    if 'healthy' in filename_lower:
                        crop = filename.split('_')[0].capitalize() if '_' in filename else 'Unknown'
                        test_images.append({
                            'name': f'{crop} : Healthy',
                            'filename': filename,
                            'path': img_path,
                            'crop': crop
                        })
    
    # Sort by crop name for better organization
    test_images.sort(key=lambda x: (x['crop'], x['name']))
    
    print(f"Loaded {len(test_images)} test images from test_images folder")
    
    # Log warning if no test images found
    if not test_images:
        print("Warning: No test images found in test_images folder")
    
    return render_template('index.html', t=t, lang=lang, languages=SUPPORTED_LANGUAGES, test_images=test_images)

@app.route('/mobile-device')
def mobile_device_detected_page():
    t = get_translations()
    lang = get_language()
    return render_template('mobile-device.html', t=t, lang=lang, languages=SUPPORTED_LANGUAGES)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Check if test image path is provided
        test_image_path = request.form.get('test_image_path')
        if test_image_path:
            # Use test image directly - handle both absolute and relative paths
            if os.path.isabs(test_image_path):
                file_path = test_image_path
            else:
                # Try test_images folder first
                file_path = os.path.join(BASE_DIR, 'static', 'test_images', os.path.basename(test_image_path))
                if not os.path.exists(file_path):
                    # Fallback to uploads folder
                    file_path = os.path.join(BASE_DIR, 'static', 'uploads', os.path.basename(test_image_path))
            
            if not os.path.exists(file_path):
                return "Test image not found", 404
        else:
            # Handle uploaded file
            image = request.files.get('image')
            if not image:
                return "No image provided", 400
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
