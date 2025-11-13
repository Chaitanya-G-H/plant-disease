import re
import pandas as pd
import os

class PlantDiseaseChatbot:
    def __init__(self, disease_info, crop_info, supplement_info):
        self.disease_info = disease_info
        self.crop_info = crop_info
        self.supplement_info = supplement_info
        
        # Common keywords for different types of questions
        self.disease_keywords = ['disease', 'sick', 'problem', 'infection', 'fungus', 'bacteria', 
                                'virus', 'pest', 'rot', 'spot', 'blight', 'wilt', 'mold', 'scab',
                                'रोग', 'बीमारी', 'समस्या', 'संक्रमण', 'फंगस', 'बैक्टीरिया',
                                'বিকার', 'রোগ', 'সমস্যা', 'সংক্রামণ', 'রোগ', 'সমস্যা']
        
        self.crop_keywords = ['crop', 'plant', 'grow', 'cultivation', 'harvest', 'planting',
                             'फसल', 'पौधा', 'उगाना', 'खेती', 'फसल काटना',
                             'ফসল', 'গাছ', 'বাড়ানো', 'চাষ', 'ফসল কাটা']
        
        self.general_keywords = ['how', 'what', 'when', 'where', 'why', 'help', 'advice', 'suggestion',
                                'कैसे', 'क्या', 'कब', 'कहाँ', 'क्यों', 'मदद', 'सलाह',
                                'কিভাবে', 'কী', 'কখন', 'কোথায়', 'কেন', 'সাহায্য', 'পরামর্শ']
    
    def find_disease_info(self, query, lang='en'):
        """Find disease information based on query"""
        query_lower = query.lower()
        
        # Search in disease names
        for idx, row in self.disease_info.iterrows():
            disease_name = str(row['disease_name']).lower()
            if any(keyword in disease_name for keyword in query_lower.split()):
                return {
                    'type': 'disease',
                    'name': row['disease_name'],
                    'description': row['description'],
                    'prevention': row['Possible Steps'],
                    'image_url': row.get('image_url', '')
                }
        
        # Search in descriptions
        for idx, row in self.disease_info.iterrows():
            description = str(row['description']).lower()
            if any(word in description for word in query_lower.split() if len(word) > 3):
                return {
                    'type': 'disease',
                    'name': row['disease_name'],
                    'description': row['description'],
                    'prevention': row['Possible Steps'],
                    'image_url': row.get('image_url', '')
                }
        
        return None
    
    def find_crop_info(self, query, lang='en'):
        """Find crop information based on query"""
        query_lower = query.lower()
        
        # Search in crop names
        for idx, row in self.crop_info.iterrows():
            crop_name = str(row['crop_name']).lower()
            if crop_name in query_lower or any(word in query_lower for word in crop_name.split()):
                return {
                    'type': 'crop',
                    'name': row['crop_name'],
                    'description': row.get('description', ''),
                    'growing_season': row.get('growing_season', ''),
                    'growth_period': row.get('growth_period', ''),
                    'harvest_time': row.get('harvest_time', ''),
                    'soil_requirements': row.get('soil_requirements', ''),
                    'watering_requirements': row.get('watering_requirements', ''),
                    'sunlight_requirements': row.get('sunlight_requirements', ''),
                    'disease_prevention': row.get('disease_prevention', '')
                }
        
        return None
    
    def generate_response(self, query, lang='en'):
        """Generate response based on query"""
        query_lower = query.lower().strip()
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'namaste', 'namaskar', 'नमस्ते', 'হ্যালো', 'வணக்கம்']
        if any(greeting in query_lower for greeting in greetings):
            return {
                'response': self.get_greeting(lang),
                'type': 'greeting'
            }
        
        # Check for help requests
        if any(word in query_lower for word in ['help', 'सहायता', 'সাহায্য', 'மொத்தம்']):
            return {
                'response': self.get_help_message(lang),
                'type': 'help'
            }
        
        # Try to find disease information
        disease_info = self.find_disease_info(query, lang)
        if disease_info:
            return {
                'response': self.format_disease_response(disease_info, lang),
                'type': 'disease',
                'data': disease_info
            }
        
        # Try to find crop information
        crop_info = self.find_crop_info(query, lang)
        if crop_info:
            return {
                'response': self.format_crop_response(crop_info, lang),
                'type': 'crop',
                'data': crop_info
            }
        
        # General agricultural advice
        if any(word in query_lower for word in ['fertilizer', 'fertiliser', 'खाद', 'উর্বরতা', 'உரம்']):
            return {
                'response': self.get_fertilizer_advice(lang),
                'type': 'general'
            }
        
        if any(word in query_lower for word in ['water', 'watering', 'पानी', 'জল', 'நீர்']):
            return {
                'response': self.get_watering_advice(lang),
                'type': 'general'
            }
        
        if any(word in query_lower for word in ['soil', 'मिट्टी', 'মাটি', 'மண்']):
            return {
                'response': self.get_soil_advice(lang),
                'type': 'general'
            }
        
        # Default response
        return {
            'response': self.get_default_response(lang),
            'type': 'default'
        }
    
    def get_greeting(self, lang='en'):
        greetings = {
            'en': "Hello! I'm your agricultural assistant. How can I help you today? You can ask me about plant diseases, crops, farming techniques, or any agricultural questions.",
            'hi': "नमस्ते! मैं आपकी कृषि सहायक हूं। आज मैं आपकी कैसे मदद कर सकती हूं? आप मुझसे पौधों की बीमारियों, फसलों, खेती की तकनीकों, या किसी भी कृषि प्रश्नों के बारे में पूछ सकते हैं।",
            'bn': "হ্যালো! আমি আপনার কৃষি সহায়ক। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি? আপনি আমাকে উদ্ভিদ রোগ, ফসল, চাষের কৌশল, বা কোনও কৃষি প্রশ্ন সম্পর্কে জিজ্ঞাসা করতে পারেন।",
            'ta': "வணக்கம்! நான் உங்கள் வேளாண்மை உதவியாளர். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்? நீங்கள் என்னிடம் தாவர நோய்கள், பயிர்கள், விவசாய நுட்பங்கள் அல்லது எந்த வேளாண்மை கேள்விகளையும் கேட்கலாம்।",
            'te': "నమస్కారం! నేను మీ వ్యవసాయ సహాయకుడిని. ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను? మీరు నన్ను మొక్కల వ్యాధులు, పంటలు, వ్యవసాయ పద్ధతులు లేదా ఏదైనా వ్యవసాయ ప్రశ్నల గురించి అడగవచ్చు।",
            'mr': "नमस्कार! मी तुमचा कृषी सहाय्यक आहे. आज मी तुम्हाला कशी मदत करू शकतो? तुम्ही मला वनस्पती रोग, पिके, शेती तंत्रज्ञान किंवा कोणत्याही कृषी प्रश्नांबद्दल विचारू शकता।",
            'gu': "નમસ્તે! હું તમારો કૃષિ સહાયક છું. આજે હું તમને કેવી રીતે મદદ કરી શકું? તમે મને છોડ રોગ, પાક, ખેતી તકનીકો અથવા કોઈપણ કૃષિ પ્રશ્નો વિશે પૂછી શકો છો।",
            'kn': "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಸಹಾಯಕ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು? ನೀವು ನನ್ನನ್ನು ಸಸ್ಯ ರೋಗಗಳು, ಬೆಳೆಗಳು, ಕೃಷಿ ತಂತ್ರಗಳು ಅಥವಾ ಯಾವುದೇ ಕೃಷಿ ಪ್ರಶ್ನೆಗಳ ಬಗ್ಗೆ ಕೇಳಬಹುದು।",
            'ml': "ഹലോ! ഞാൻ നിങ്ങളുടെ കൃഷി സഹായിയാണ്. ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം? നിങ്ങൾക്ക് എന്നോട് സസ്യ രോഗങ്ങൾ, വിളകൾ, കൃഷി സാങ്കേതികവിദ്യകൾ അല്ലെങ്കിൽ ഏതെങ്കിലും കൃഷി ചോദ്യങ്ങൾ ചോദിക്കാം।",
            'pa': "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ ਹਾਂ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ? ਤੁਸੀਂ ਮੈਨੂੰ ਪੌਦਿਆਂ ਦੀਆਂ ਬਿਮਾਰੀਆਂ, ਫਸਲਾਂ, ਖੇਤੀਬਾੜੀ ਤਕਨੀਕਾਂ ਜਾਂ ਕੋਈ ਵੀ ਖੇਤੀਬਾੜੀ ਸਵਾਲਾਂ ਬਾਰੇ ਪੁੱਛ ਸਕਦੇ ਹੋ।",
            'or': "ନମସ୍କାର! ମୁଁ ତୁମର କୃଷି ସହାୟକ। ଆଜି ମୁଁ ତୁମକୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି? ଆପଣ ମୋତେ ଉଦ୍ଭିଦ ରୋଗ, ଫସଲ, କୃଷି କୌଶଳ କିମ୍ବା ଯେକୌଣସି କୃଷି ପ୍ରଶ୍ନ ବିଷୟରେ ପଚାରିପାରନ୍ତି।",
            'ur': "السلام علیکم! میں آپ کا زرعی معاون ہوں۔ آج میں آپ کی کس طرح مدد کر سکتا ہوں؟ آپ مجھ سے پودوں کی بیماریاں، فصلیں، کھیتی باڑی کی تکنیک یا کوئی بھی زرعی سوالات کے بارے میں پوچھ سکتے ہیں۔"
        }
        return greetings.get(lang, greetings['en'])
    
    def get_help_message(self, lang='en'):
        help_messages = {
            'en': "I can help you with:\n1. Plant disease identification and treatment\n2. Crop information and cultivation tips\n3. Farming techniques and best practices\n4. Fertilizer and soil advice\n5. Watering and care instructions\n\nJust ask me any question about agriculture!",
            'hi': "मैं आपकी मदद कर सकती हूं:\n1. पौधों की बीमारी की पहचान और उपचार\n2. फसल की जानकारी और खेती के टिप्स\n3. खेती की तकनीक और सर्वोत्तम प्रथाएं\n4. उर्वरक और मिट्टी की सलाह\n5. पानी देने और देखभाल के निर्देश\n\nबस मुझे कृषि के बारे में कोई भी प्रश्न पूछें!",
            'bn': "আমি আপনাকে সাহায্য করতে পারি:\n1. উদ্ভিদ রোগ সনাক্তকরণ এবং চিকিত্সা\n2. ফসলের তথ্য এবং চাষের টিপস\n3. চাষের কৌশল এবং সেরা অনুশীলন\n4. সার এবং মাটির পরামর্শ\n5. জল দেওয়া এবং যত্নের নির্দেশাবলী\n\nশুধু আমাকে কৃষি সম্পর্কে কোন প্রশ্ন জিজ্ঞাসা করুন!",
            'ta': "நான் உங்களுக்கு உதவ முடியும்:\n1. தாவர நோய் அடையாளம் மற்றும் சிகிச்சை\n2. பயிர் தகவல் மற்றும் சாகுபடி உதவிக்குறிப்புகள்\n3. விவசாய நுட்பங்கள் மற்றும் சிறந்த நடைமுறைகள்\n4. உரம் மற்றும் மண் ஆலோசனை\n5. நீர்ப்பாசனம் மற்றும் பராமரிப்பு வழிமுறைகள்\n\nவேளாண்மை பற்றி ஏதேனும் கேள்வியைக் கேளுங்கள்!",
        }
        return help_messages.get(lang, help_messages['en'])
    
    def format_disease_response(self, disease_info, lang='en'):
        if lang == 'en':
            return f"Disease: {disease_info['name']}\n\nDescription: {disease_info['description'][:200]}...\n\nPrevention Steps: {disease_info['prevention'][:200]}...\n\nFor more details, visit our disease detection page!"
        elif lang == 'hi':
            return f"रोग: {disease_info['name']}\n\nविवरण: {disease_info['description'][:200]}...\n\nनिवारण के उपाय: {disease_info['prevention'][:200]}...\n\nअधिक विवरण के लिए, हमारे रोग पहचान पृष्ठ पर जाएं!"
        else:
            return f"Disease: {disease_info['name']}\n\nDescription: {disease_info['description'][:200]}...\n\nPrevention Steps: {disease_info['prevention'][:200]}..."
    
    def format_crop_response(self, crop_info, lang='en'):
        if lang == 'en':
            return f"Crop: {crop_info['name']}\n\nDescription: {crop_info.get('description', '')[:200]}...\n\nGrowing Season: {crop_info.get('growing_season', 'N/A')}\nGrowth Period: {crop_info.get('growth_period', 'N/A')}\nHarvest Time: {crop_info.get('harvest_time', 'N/A')}\n\nFor complete guide, visit our crop details page!"
        elif lang == 'hi':
            return f"फसल: {crop_info['name']}\n\nविवरण: {crop_info.get('description', '')[:200]}...\n\nउगाने का मौसम: {crop_info.get('growing_season', 'N/A')}\nवृद्धि अवधि: {crop_info.get('growth_period', 'N/A')}\nफसल काटने का समय: {crop_info.get('harvest_time', 'N/A')}\n\nपूर्ण गाइड के लिए, हमारे फसल विवरण पृष्ठ पर जाएं!"
        else:
            return f"Crop: {crop_info['name']}\n\n{crop_info.get('description', '')[:200]}..."
    
    def get_fertilizer_advice(self, lang='en'):
        advice = {
            'en': "For healthy plants, use balanced fertilizers (NPK 10-10-10) in early spring. Avoid over-fertilization as it can harm plants. Organic fertilizers like compost and manure are excellent choices. Apply fertilizer based on soil test results for best results.",
            'hi': "स्वस्थ पौधों के लिए, शुरुआती वसंत में संतुलित उर्वरक (NPK 10-10-10) का उपयोग करें। अत्यधिक उर्वरक से बचें क्योंकि यह पौधों को नुकसान पहुंचा सकता है। खाद और खाद जैसे जैविक उर्वरक उत्कृष्ट विकल्प हैं। सर्वोत्तम परिणामों के लिए मिट्टी के परीक्षण परिणामों के आधार पर उर्वरक लगाएं।",
        }
        return advice.get(lang, advice['en'])
    
    def get_watering_advice(self, lang='en'):
        advice = {
            'en': "Water plants deeply but less frequently. Most plants need 1-2 inches of water per week. Water in the morning to allow leaves to dry. Avoid overhead watering for disease-prone plants. Check soil moisture before watering.",
            'hi': "पौधों को गहराई से लेकिन कम बार पानी दें। अधिकांश पौधों को प्रति सप्ताह 1-2 इंच पानी की आवश्यकता होती है। पत्तियों को सूखने देने के लिए सुबह पानी दें। रोग-प्रवण पौधों के लिए ऊपर से पानी देने से बचें। पानी देने से पहले मिट्टी की नमी की जांच करें।",
        }
        return advice.get(lang, advice['en'])
    
    def get_soil_advice(self, lang='en'):
        advice = {
            'en': "Good soil is essential for healthy plants. Most crops prefer well-drained, loamy soil with pH between 6.0-7.0. Test your soil regularly and amend it with organic matter like compost. Ensure proper drainage to prevent waterlogging.",
            'hi': "स्वस्थ पौधों के लिए अच्छी मिट्टी आवश्यक है। अधिकांश फसलें 6.0-7.0 के pH के साथ अच्छी तरह से सूखा, दोमट मिट्टी पसंद करती हैं। अपनी मिट्टी का नियमित रूप से परीक्षण करें और इसे खाद जैसे कार्बनिक पदार्थों के साथ संशोधित करें। जलभराव को रोकने के लिए उचित जल निकासी सुनिश्चित करें।",
        }
        return advice.get(lang, advice['en'])
    
    def get_default_response(self, lang='en'):
        responses = {
            'en': "I'm here to help with agricultural questions! You can ask me about:\n- Plant diseases and their treatment\n- Crop cultivation and care\n- Farming techniques\n- Soil and fertilizer advice\n\nPlease try rephrasing your question or ask about a specific crop or disease.",
            'hi': "मैं कृषि प्रश्नों में मदद के लिए यहां हूं! आप मुझसे पूछ सकते हैं:\n- पौधों की बीमारियां और उनका उपचार\n- फसल की खेती और देखभाल\n- खेती की तकनीक\n- मिट्टी और उर्वरक सलाह\n\nकृपया अपने प्रश्न को पुनः व्यक्त करने का प्रयास करें या किसी विशिष्ट फसल या बीमारी के बारे में पूछें।",
            'bn': "আমি কৃষি প্রশ্নে সাহায্যের জন্য এখানে আছি! আপনি আমাকে জিজ্ঞাসা করতে পারেন:\n- উদ্ভিদ রোগ এবং তাদের চিকিত্সা\n- ফসল চাষ এবং যত্ন\n- চাষের কৌশল\n- মাটি এবং সার পরামর্শ\n\nঅনুগ্রহ করে আপনার প্রশ্নটি পুনরায় লেখার চেষ্টা করুন বা একটি নির্দিষ্ট ফসল বা রোগ সম্পর্কে জিজ্ঞাসা করুন।",
        }
        return responses.get(lang, responses['en'])

# Initialize chatbot
def get_chatbot():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    disease_info = pd.read_csv(os.path.join(BASE_DIR, 'disease_info.csv'), encoding='cp1252')
    crop_info = pd.read_csv(os.path.join(BASE_DIR, 'crop_info.csv'), encoding='utf-8')
    supplement_info = pd.read_csv(os.path.join(BASE_DIR, 'supplement_info.csv'), encoding='cp1252')
    return PlantDiseaseChatbot(disease_info, crop_info, supplement_info)

