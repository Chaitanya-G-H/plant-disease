# 🌱 Plant Disease Detection System

An AI-powered web application for detecting plant diseases in fruits and vegetables using Convolutional Neural Networks (CNN) and PyTorch. This system helps farmers and gardeners identify diseases early, get treatment recommendations, and access comprehensive crop information in multiple languages.

## ✨ Features

### 🔬 Disease Detection
- **AI-Powered Detection**: Upload leaf images and get instant disease identification
- **Multiple Disease Classes**: Detects 39+ different plant diseases
- **High Accuracy**: Trained on Plant Village dataset using CNN architecture
- **Detailed Results**: Get disease name, description, prevention steps, and treatment recommendations

### 🌍 Multilingual Support
- **12 Indian Languages + English**: Full website translation support
- **Supported Languages**: English, Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu
- **Language Switcher**: Easy language selection in navigation bar
- **Dynamic Translation**: All UI elements translated in real-time

### 🗣️ Voice-Enabled Chatbot
- **Multilingual Chatbot**: Interactive agricultural assistant
- **Speech-to-Text**: Voice input for farmers who prefer speaking
- **Text-to-Speech**: Audio responses in selected language
- **Smart Responses**: Answers questions about:
  - Plant diseases and symptoms
  - Crop growing techniques
  - Fertilizer recommendations
  - Soil and watering requirements
  - Disease prevention tips

### 🌾 Crop Information Pages
- **Detailed Crop Guides**: Comprehensive information for 14+ crops
- **Growing Instructions**: Season, growth period, soil requirements
- **Care Guidelines**: Watering, sunlight, temperature requirements
- **Disease Prevention**: Tips to prevent common diseases
- **Harvest Information**: When and how to harvest

### 🛒 Supplement & Fertilizer Store
- **Treatment Recommendations**: Suggested supplements for each disease
- **Purchase Links**: Direct links to buy recommended products
- **Fertilizer Guide**: Information about fertilizer requirements

### 📱 Responsive Design
- **Mobile-Friendly**: Works on all devices (desktop, tablet, mobile)
- **Zoom Support**: Flexible zoom in/out for better accessibility
- **Modern UI**: Beautiful, intuitive user interface with Bootstrap 5
- **Fast Loading**: Optimized for quick response times

## 🚀 Technology Stack

### Backend
- **Flask**: Python web framework
- **PyTorch**: Deep learning framework
- **CNN**: Convolutional Neural Network architecture
- **Pandas**: Data manipulation and CSV handling
- **Pillow**: Image processing

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript**: Interactive features and API calls
- **Bootstrap 5**: Responsive UI framework
- **Web Speech API**: Speech-to-text and text-to-speech
- **Jinja2**: Template engine

### Data
- **CSV Files**: Disease info, crop info, supplement info
- **JSON**: Translation files for multilingual support
- **Pre-trained Model**: CNN model for disease classification

## 📋 Supported Crops

The system supports disease detection for the following crops:

- 🍎 Apple
- 🔵 Blueberry
- 🍒 Cherry
- 🌽 Corn
- 🍇 Grape
- 🍊 Orange
- 🍑 Peach
- 🌶️ Pepper Bell
- 🥔 Potato
- 🍓 Raspberry
- 🌾 Soybean
- 🎃 Squash
- 🍓 Strawberry
- 🍅 Tomato

## 🛠️ Installation

### Prerequisites
- **Python 3.8+** (Python 3.9 recommended)
- **pip** (Python package manager)
- **Virtual Environment** (recommended)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Plant-Disease-Detection-main
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
cd "Flask Deployed App"
pip install -r requirements.txt
```

### Step 4: Download Model File
1. Download the pre-trained model file `plant_disease_model_1_latest.pt` from [Google Drive](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link)
2. Place the model file in the `Model` folder:
   ```
   Model/plant_disease_model_1_latest.pt
   ```

### Step 5: Run the Application
```bash
# From Flask Deployed App directory
python app.py
```

The application will be available at: `http://localhost:5000`

## 📁 Project Structure

```
Plant-Disease-Detection-main/
│
├── Flask Deployed App/          # Main application directory
│   ├── app.py                   # Flask application main file
│   ├── CNN.py                   # CNN model architecture
│   ├── chatbot.py               # Chatbot logic and responses
│   ├── load_class_mappings.py   # Dynamic class mapping loader
│   ├── requirements.txt         # Python dependencies
│   ├── translations.json        # Multilingual translations
│   ├── disease_info.csv         # Disease information database
│   ├── crop_info.csv            # Crop information database
│   ├── supplement_info.csv      # Supplement/fertilizer database
│   │
│   ├── templates/               # HTML templates
│   │   ├── base.html           # Base template
│   │   ├── home.html           # Home page
│   │   ├── index.html          # Disease detection page
│   │   ├── submit.html         # Results page
│   │   ├── crop_details.html   # Crop information page
│   │   ├── market.html         # Supplement store page
│   │   ├── contact-us.html     # Contact page
│   │   └── chatbot.html        # Chatbot widget
│   │
│   └── static/                  # Static files
│       └── uploads/             # Uploaded images
│
├── Model/                       # Model directory
│   └── plant_disease_model_1_latest.pt  # Pre-trained model file
│
├── venv/                        # Virtual environment (create this)
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

## 🎯 Usage

### Disease Detection
1. Navigate to the **AI Engine** page
2. Click **Upload Image** or **Choose File**
3. Select a leaf image from your device
4. Click **Submit** to analyze
5. View the detection results with:
   - Disease name and description
   - Prevention steps
   - Recommended supplements
   - Treatment suggestions

### Crop Information
1. Go to the **Home** page
2. Click on any crop card (Apple, Tomato, etc.)
3. View detailed information about:
   - Growing season and growth period
   - Soil and watering requirements
   - Sunlight and temperature needs
   - Planting methods
   - Disease prevention tips
   - Harvest time

### Multilingual Chatbot
1. Click the **chatbot icon** in the bottom-right corner
2. Select your preferred language from the language menu
3. Ask questions about:
   - Plant diseases
   - Crop growing techniques
   - Fertilizers and supplements
   - Soil and watering requirements
4. Use the **microphone button** for voice input
5. Use the **speaker button** to hear responses

### Language Selection
1. Click the **language dropdown** in the navigation bar
2. Select your preferred language
3. The entire website will be translated instantly

## 🔧 Configuration

### Model Path
The model file path is configured in `app.py`:
```python
model_path = os.path.join(PROJECT_ROOT, 'Model', 'plant_disease_model_1_latest.pt')
```

### Supported Languages
Languages are configured in `app.py`:
```python
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    # ... more languages
}
```

### Adding New Translations
1. Open `Flask Deployed App/translations.json`
2. Add translation keys for new languages
3. Update `SUPPORTED_LANGUAGES` in `app.py`

## 🌐 API Endpoints

### Chatbot API
- **Endpoint**: `/api/chatbot`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "query": "What is tomato blight?"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "response": "Tomato blight is...",
    "type": "disease",
    "data": {...}
  }
  ```

### Crop Details
- **Endpoint**: `/crop/<crop_name>`
- **Method**: GET
- **Example**: `/crop/Apple`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Commit your changes**: `git commit -m 'Add some feature'`
4. **Push to the branch**: `git push origin feature/your-feature-name`
5. **Submit a pull request**

### Areas for Contribution
- 🎨 UI/UX improvements
- 🔬 Model accuracy enhancements
- 🌍 Additional language translations
- 📊 New crop and disease data
- 🐛 Bug fixes
- 📝 Documentation improvements

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Chaitanya G H**
- Project: Plant Disease Detection System
- Capstone Project - 7th Semester

## 🙏 Acknowledgments

- **Plant Village Dataset**: For providing the training dataset
- **PyTorch Community**: For the excellent deep learning framework
- **Flask Community**: For the web framework
- **Bootstrap**: For the UI framework
- **All Contributors**: For their valuable contributions

## 📞 Contact

For any queries or suggestions, please contact us through the **Contact Us** page on the website.

## 🔗 Useful Links

- **Blog**: [Plant Disease Detection Using CNN with PyTorch](https://medium.com/analytics-vidhya/plant-disease-detection-using-convolutional-neural-networks-and-pytorch-87c00c54c88f)
- **Model Download**: [Google Drive](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link)

## 📊 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: PyTorch
- **Dataset**: Plant Village Dataset
- **Classes**: 39+ disease categories
- **Input Size**: 224x224 pixels
- **Preprocessing**: Resize(255) → CenterCrop(224) → ToTensor()

## 🚨 Troubleshooting

### Model Not Found Error
- Ensure the model file is in the `Model` folder
- Check the file name: `plant_disease_model_1_latest.pt`
- Verify the file path in `app.py`

### Import Errors
- Activate the virtual environment
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Language Not Working
- Clear browser cache
- Check `translations.json` file
- Verify language code in `SUPPORTED_LANGUAGES`

### Chatbot Not Responding
- Check browser console for errors
- Verify microphone permissions (for voice input)
- Ensure internet connection (for Web Speech API)

---

**⭐ Star this repository if you find it helpful!**

**🌱 Happy Farming!**
