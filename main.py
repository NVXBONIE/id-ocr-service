import easyocr
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import re
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Initialize EasyOCR reader for Romanian and English
reader = easyocr.Reader(['ro', 'en'])

class RomanianIDParser:
    def __init__(self):
        # Romanian ID card field patterns
        self.patterns = {
            'cnp': r'\b[1-8]\d{12}\b',  # CNP pattern: starts with 1-8, followed by 12 digits
            'serie': r'\b[A-Z]{2}\s*\d{6}\b',  # Serie pattern: 2 letters + 6 digits
            'valabilitate': r'\b\d{2}[./]\d{2}[./]\d{4}\b',  # Date pattern
        }
        
        # Field keywords to help identify sections
        self.field_keywords = {
            'nume': ['nume', 'family name', 'nom'],
            'prenume': ['prenume', 'given name', 'prénom'],
            'cnp': ['cnp', 'personal numerical code'],
            'cetatenie': ['cetățenie', 'nationality', 'nationalité'],
            'loc_nastere': ['loc naștere', 'place of birth', 'lieu de naissance'],
            'domiciliu': ['domiciliu', 'address', 'adresse'],
            'emisa_de': ['emisă de', 'issued by', 'émis par'],
            'valabilitate': ['valabilitate', 'valid until', 'valable jusqu\'au']
        }

    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised

    def extract_text_with_positions(self, image):
        """Extract text using EasyOCR with position information"""
        results = reader.readtext(image)
        
        # Sort results by vertical position (top to bottom)
        results.sort(key=lambda x: x[0][0][1])
        
        return results

    def find_cnp(self, text_list):
        """Extract CNP using pattern matching"""
        full_text = ' '.join(text_list)
        cnp_match = re.search(self.patterns['cnp'], full_text.replace(' ', ''))
        return cnp_match.group() if cnp_match else None

    def find_serie(self, text_list):
        """Extract serie using pattern matching"""
        full_text = ' '.join(text_list)
        serie_match = re.search(self.patterns['serie'], full_text.replace(' ', ''))
        return serie_match.group().replace(' ', '') if serie_match else None

    def find_field_by_keyword(self, results, field_keywords, next_lines=2):
        """Find field value by looking for keywords and extracting nearby text"""
        text_list = [result[1].lower() for result in results]
        
        for i, text in enumerate(text_list):
            for keyword in field_keywords:
                if keyword.lower() in text:
                    # Look for the value in the next few lines or same line
                    values = []
                    
                    # Check same line (after the keyword)
                    current_text = results[i][1]
                    keyword_pos = current_text.lower().find(keyword.lower())
                    if keyword_pos != -1:
                        after_keyword = current_text[keyword_pos + len(keyword):].strip()
                        if after_keyword and not any(kw in after_keyword.lower() for kw_list in self.field_keywords.values() for kw in kw_list):
                            values.append(after_keyword)
                    
                    # Check next lines
                    for j in range(1, next_lines + 1):
                        if i + j < len(results):
                            next_text = results[i + j][1].strip()
                            # Skip if it's another field keyword
                            if not any(kw in next_text.lower() for kw_list in self.field_keywords.values() for kw in kw_list):
                                if next_text:
                                    values.append(next_text)
                    
                    return ' '.join(values) if values else None
        
        return None

    def find_valabilitate(self, text_list):
        """Extract valabilitate (validity date)"""
        full_text = ' '.join(text_list)
        date_match = re.search(self.patterns['valabilitate'], full_text)
        return date_match.group() if date_match else None

    def clean_extracted_text(self, text):
        """Clean and format extracted text"""
        if not text:
            return None
        
        # Remove extra whitespace and clean up
        cleaned = ' '.join(text.split())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-./]', '', cleaned)
        
        return cleaned.strip() if cleaned.strip() else None

    def parse_romanian_id(self, image):
        """Main parsing function for Romanian ID card"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Extract text with positions
        results = self.extract_text_with_positions(processed_image)
        
        # Extract all text for pattern matching
        all_text = [result[1] for result in results]
        
        # Extract fields
        extracted_data = {
            'cnp': self.find_cnp(all_text),
            'nume': self.find_field_by_keyword(results, self.field_keywords['nume']),
            'prenume': self.find_field_by_keyword(results, self.field_keywords['prenume']),
            'serie': self.find_serie(all_text),
            'cetatenie': self.find_field_by_keyword(results, self.field_keywords['cetatenie']),
            'loc_nastere': self.find_field_by_keyword(results, self.field_keywords['loc_nastere']),
            'domiciliu': self.find_field_by_keyword(results, self.field_keywords['domiciliu']),
            'emisa_de': self.find_field_by_keyword(results, self.field_keywords['emisa_de']),
            'valabilitate': self.find_valabilitate(all_text)
        }
        
        # Clean extracted data
        for key, value in extracted_data.items():
            extracted_data[key] = self.clean_extracted_text(value)
        
        return extracted_data

# Initialize parser
id_parser = RomanianIDParser()

@app.route('/extract-romanian-id', methods=['POST'])
def extract_romanian_id():
    """
    Endpoint to extract data from Romanian ID card image
    
    Expected input:
    - JSON with base64 encoded image: {"image": "base64_string"}
    - OR multipart/form-data with image file
    
    Returns:
    - JSON with extracted fields
    """
    try:
        image = None
        
        # Handle different input formats
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'No image provided in JSON'}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            image = np.array(Image.open(BytesIO(image_data)))
            
        elif 'image' in request.files:
            # Handle multipart form data
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            # Convert uploaded file to numpy array
            image = np.array(Image.open(file.stream))
        
        else:
            return jsonify({'error': 'No image provided. Send as JSON with base64 or as multipart form data'}), 400
        
        # Extract data from Romanian ID
        extracted_data = id_parser.parse_romanian_id(image)
        
        # Add metadata
        response_data = {
            'success': True,
            'extracted_data': extracted_data,
            'timestamp': datetime.now().isoformat(),
            'fields_found': sum(1 for value in extracted_data.values() if value is not None)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Romanian ID OCR Service',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Romanian ID OCR Service',
        'version': '1.0.0',
        'endpoints': {
            '/extract-romanian-id': 'POST - Extract data from Romanian ID card',
            '/health': 'GET - Health check'
        },
        'usage': {
            'method': 'POST',
            'content_type': 'application/json or multipart/form-data',
            'input_json': {'image': 'base64_encoded_image_string'},
            'input_form': 'image file in multipart form data'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8002))
    app.run(debug=False, host='0.0.0.0', port=port)