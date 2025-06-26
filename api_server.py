from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
from io import BytesIO
import uuid
import random
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

class TongueAnalyzer:
    def __init__(self, model_path="best_tongue_classification_model.pth"):
        """
        Initialize the tongue analyzer for API use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the classification model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.categories = checkpoint.get('categories', [])
            self.category_to_idx = checkpoint.get('category_to_idx', {})
            
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.categories))
            
            # Fix the state dict keys by removing "model." prefix if present
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove "model." prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict)
            print(f"Loaded model with {len(self.categories)} classes")
        else:
            print(f"Model not found at {model_path}")
            self.categories = [
                '淡白舌白苔', '红舌黄苔', '淡白舌黄苔', '绛舌灰黑苔', '绛舌黄苔',
                '绛舌白苔', '红舌灰黑苔', '红舌白苔', '淡红舌灰黑苔', '淡红舌黄苔',
                '淡红舌白苔', '青紫舌白苔', '青紫舌黄苔', '青紫舌灰黑苔', '淡白舌灰黑苔'
            ]
            self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_tongue_features(self, image_path):
        """
        Detect specific tongue features for visualization
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect tongue region using color-based segmentation
        features = self._detect_tongue_region_and_features(image_rgb)
        
        return features

    def _detect_tongue_region_and_features(self, image):
        """
        Detect tongue region and specific features
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for skin/tongue color (adjust these values based on your images)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        
        # Create mask for tongue region
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = {
            'tongue_region': None,
            'cracks': [],
            'coating': [],
            'color_variations': [],
            'teeth_marks': []
        }
        
        if contours:
            # Find the largest contour (likely the tongue)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['tongue_region'] = (x, y, w, h)
            
            # Detect cracks (fissures) using edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines that could be cracks
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is within tongue region
                    if (x <= x1 <= x+w and x <= x2 <= x+w and 
                        y <= y1 <= y+h and y <= y2 <= y+h):
                        features['cracks'].append(((x1, y1), (x2, y2)))
            
            # Detect coating (areas with different texture/color)
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                # Detect areas with different brightness (potential coating)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                coating_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in coating_contours:
                    if cv2.contourArea(contour) > 100:  # Filter small areas
                        cx, cy, cw, ch = cv2.boundingRect(contour)
                        features['coating'].append((x + cx, y + cy, cw, ch))
        
        return features

    def create_annotated_image(self, image_path, features):
        """
        Create an annotated image with detected features
        """
        try:
            # Open original image
            original_image = Image.open(image_path)
            draw = ImageDraw.Draw(original_image)
            
            # Try to load a font that supports Chinese characters
            try:
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",  # macOS
                    "/System/Library/Fonts/STHeiti Light.ttc",  # macOS
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                    "arial.ttf",  # Windows
                    "simhei.ttf",  # Windows Chinese
                    "msyh.ttc",  # Windows Chinese
                ]
                
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, 24)
                        # Test if it can render Chinese
                        test_text = "测试"
                        font.getbbox(test_text)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except:
                font = ImageFont.load_default()
            
            # Draw tongue region
            if features and features['tongue_region']:
                x, y, w, h = features['tongue_region']
                # Draw rectangle around tongue
                draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
                draw.text((x, y-30), "Tongue Region", fill='green', font=font)
            
            # Draw cracks/fissures (label only, no count)
            if features and features['cracks']:
                for (x1, y1), (x2, y2) in features['cracks'][:30]:  # Limit to first 30 for visibility
                    draw.line([x1, y1, x2, y2], fill='red', width=2)
                # Add label for cracks (no count)
                draw.text((10, 10), "Cracks/Fissures Detected", fill='red', font=font)
            
            # Draw coating areas (label only, no count)
            if features and features['coating']:
                for cx, cy, cw, ch in features['coating']:
                    draw.rectangle([cx, cy, cx+cw, cy+ch], outline='blue', width=2)
                # Add label for coating (no count)
                draw.text((10, 40), "Coating Areas Detected", fill='blue', font=font)
            
            return original_image.convert('RGB')
            
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            # Return original image if there's an error
            return Image.open(image_path)

    def classify(self, image_path):
        """
        Classify the tongue image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                primary_classification = self.categories[predicted_idx.item()]
                confidence_value = confidence.item()
                
                return {
                    'primary_classification': primary_classification,
                    'confidence': confidence_value,
                    'all_probabilities': probabilities.cpu().numpy().tolist()[0]
                }
        except Exception as e:
            print(f"Error during classification: {e}")
            return {
                'primary_classification': 'error',
                'confidence': 0.0,
                'all_probabilities': []
            }

class QuestionnaireAnalyzer:
    def analyze(self, gender: str, water_cups: int, symptoms: list):
        """
        Analyzes the questionnaire data and returns structured suggestions.
        """
        recommendations = []
        suggestions = []

        # Process gender
        if gender.lower() == "female":
            recommendations.append("Consider tracking your menstrual cycle as it may affect your health patterns.")

        # Process water intake
        try:
            if 0 <= int(water_cups) <= 6:
                recommendations.append("Recommendation: Drink more water. Aim for 8-10 cups daily for better hydration.")
        except (ValueError, TypeError):
            recommendations.append("Please enter a valid number for water intake.")

        # Process symptoms
        symptom_map = {
            "疲劳": "Eat warming foods such as ginger or cinnamon to boost energy.",
            "头晕": "Eat regular balanced meals to maintain stable blood sugar levels.",
            "多汗": "Eat cooling foods such as watermelon or cucumber to help regulate body temperature.",
            "失眠": "Limit caffeine intake, especially in the afternoon and evening.",
            "消化不良": "Increase dietary fiber intake through fruits, vegetables, and whole grains.",
            "口干": "Drink more water throughout the day to maintain hydration."
        }
        for symptom in symptoms:
            if symptom in symptom_map:
                suggestions.append(symptom_map[symptom])
        
        return {
            "recommendations": recommendations,
            "suggestions": suggestions
        }

class ResultIntegrator:
    def integrate(self, questionnaire_results: dict, classification_result: str):
        """
        Combines analysis from the questionnaire and AI model into a final formatted string.
        """
        symptoms_from_ai = []
        suggestions = questionnaire_results.get("suggestions", [])
        recommendations = questionnaire_results.get("recommendations", [])

        # Process AI classification result to generate symptoms and suggestions
        if classification_result != "error":
            if "crenated" in classification_result.lower():
                symptoms_from_ai.append(f"crenated {random.randint(1, 3)}")
                suggestions.append("Eat a variety of fruits and vegetables.")
            elif "fissured" in classification_result.lower():
                symptoms_from_ai.append(f"fissured {random.randint(1, 3)}")
                suggestions.append("Drink more water to maintain hydration.")
                suggestions.append("Eat foods rich in iron and vitamin B for better health.")

        # --- Build the final output string ---
        output = ""
        if classification_result != "error":
             output += f"AI Classification: {classification_result}\n\n"

        if symptoms_from_ai:
            output += "Symptoms:\n"
            # Use a set to prevent duplicate suggestions
            for i, symptom in enumerate(symptoms_from_ai, 1):
                output += f"{i}. {symptom}\n"
            output += "\n"

        if suggestions:
            output += "Suggestions:\n"
            # Use a set to prevent duplicate suggestions
            for i, suggestion in enumerate(sorted(list(set(suggestions))), 1):
                output += f"{i}. {suggestion}\n"
            output += "\n"

        if recommendations:
            output += "General Recommendations:\n"
            for i, recommendation in enumerate(recommendations, 1):
                output += f"{i}. {recommendation}\n"
        
        return output.strip() if output.strip() else "No specific recommendations at this time."

# Initialize global analyzer instances
tongue_analyzer = TongueAnalyzer()
questionnaire_analyzer = QuestionnaireAnalyzer()
result_integrator = ResultIntegrator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Flutter app"""
    return jsonify({
        'status': 'healthy',
        'message': 'Tongue Analysis API is running',
        'model_loaded': len(tongue_analyzer.categories) > 0
    })

@app.route('/analyze', methods=['POST'])
def analyze_tongue():
    """
    Main analysis endpoint for Flutter app
    Expects: multipart form data with image, gender, water_cups, symptoms
    Returns: JSON with analysis results and base64 encoded annotated image
    """
    try:
        # Get form data
        image_file = request.files.get('image')
        gender = request.form.get('gender', '')
        water_cups = int(request.form.get('water_cups', 0))
        symptoms = request.form.getlist('symptoms') if 'symptoms' in request.form else []
        
        # Validate image file
        if not image_file or image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file uploaded'
            })
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        file_extension = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            })
        
        # Create unique filename to avoid conflicts
        unique_filename = f"uploaded_image_{uuid.uuid4().hex[:8]}.jpg"
        temp_image_path = unique_filename
        
        # Save uploaded image temporarily
        image_file.save(temp_image_path)
        
        # Verify the image can be opened
        try:
            with Image.open(temp_image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(temp_image_path, 'JPEG')
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            })
        
        # Perform classification
        classification_result = tongue_analyzer.classify(temp_image_path)
        
        # Analyze questionnaire
        questionnaire_results = questionnaire_analyzer.analyze(gender, water_cups, symptoms)
        
        # Integrate results
        final_results = result_integrator.integrate(questionnaire_results, classification_result['primary_classification'])
        
        # Detect features for visualization
        features = tongue_analyzer.detect_tongue_features(temp_image_path)
        
        # Create annotated image
        annotated_image = tongue_analyzer.create_annotated_image(temp_image_path, features)
        
        # Convert annotated image to base64 for Flutter
        buffer = BytesIO()
        annotated_image.save(buffer, format='PNG', quality=95)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Clean up temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # Add random number to classification
        random_num = random.randint(1, 3)
        classification_with_number = f"{classification_result['primary_classification']} {random_num}"
        
        # Return comprehensive results for Flutter
        return jsonify({
            'success': True,
            'annotated_image': img_data,
            'classification': classification_with_number,
            'final_results': final_results,
            'features_detected': {
                'tongue_region': features['tongue_region'] is not None if features else False,
                'cracks_detected': len(features['cracks']) > 0 if features else False,
                'coating_detected': len(features['coating']) > 0 if features else False,
                'crack_count': len(features['cracks']) if features else 0,
                'coating_count': len(features['coating']) if features else 0
            },
            'questionnaire_results': questionnaire_results,
            'filename': image_file.filename
        })
        
    except Exception as e:
        # Clean up any temporary files
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/classify-only', methods=['POST'])
def classify_only():
    """
    Endpoint for classification only (without questionnaire analysis)
    """
    try:
        image_file = request.files.get('image')
        
        if not image_file or image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file uploaded'
            })
        
        # Create unique filename
        unique_filename = f"classify_{uuid.uuid4().hex[:8]}.jpg"
        temp_image_path = unique_filename
        
        # Save and process image
        image_file.save(temp_image_path)
        
        try:
            with Image.open(temp_image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(temp_image_path, 'JPEG')
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return jsonify({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            })
        
        # Perform classification
        classification_result = tongue_analyzer.classify(temp_image_path)
        
        # Clean up
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # Add random number
        random_num = random.randint(1, 3)
        classification_with_number = f"{classification_result['primary_classification']} {random_num}"
        
        return jsonify({
            'success': True,
            'classification': classification_with_number,
            'all_probabilities': classification_result.get('all_probabilities', [])
        })
        
    except Exception as e:
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 