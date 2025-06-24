import gradio as gr
import easyocr
import cv2
import numpy as np
import re
from datetime import datetime
from PIL import Image

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

def process_romanian_id(image):
    """
    Main function for Gradio interface
    Takes an image and returns extracted Romanian ID data
    """
    try:
        if image is None:
            return {
                "error": "No image provided",
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
        
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Extract data from Romanian ID
        extracted_data = id_parser.parse_romanian_id(image)
        
        # Format response for better display
        response = {
            "success": True,
            "extracted_data": extracted_data,
            "timestamp": datetime.now().isoformat(),
            "fields_found": sum(1 for value in extracted_data.values() if value is not None)
        }
        
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def format_output_for_display(result):
    """Format the result for better display in Gradio"""
    if not result.get("success", False):
        return f"❌ Error: {result.get('error', 'Unknown error')}"
    
    extracted = result.get("extracted_data", {})
    fields_found = result.get("fields_found", 0)
    
    output = f"✅ Successfully extracted {fields_found} fields:\n\n"
    
    field_labels = {
        'nume': '👤 Nume (Last Name)',
        'prenume': '👤 Prenume (First Name)', 
        'cnp': '🆔 CNP',
        'serie': '📄 Serie',
        'cetatenie': '🌍 Cetățenie (Nationality)',
        'loc_nastere': '📍 Loc Naștere (Place of Birth)',
        'domiciliu': '🏠 Domiciliu (Address)',
        'emisa_de': '🏛️ Emisă de (Issued by)',
        'valabilitate': '📅 Valabilitate (Valid until)'
    }
    
    for field, value in extracted.items():
        label = field_labels.get(field, field.title())
        if value:
            output += f"{label}: {value}\n"
        else:
            output += f"{label}: ❌ Not found\n"
    
    return output

# Create Gradio interface
with gr.Blocks(title="Romanian ID OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🇷🇴 Romanian ID Card OCR
    
    Upload an image of a Romanian ID card to extract text data automatically.
    This service can extract: Name, CNP, Serie, Nationality, Address, and more.
    
    **Supported formats**: JPG, PNG, WEBP
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Romanian ID Card Image",
                type="pil",
                sources=["upload", "clipboard"]
            )
            
            extract_btn = gr.Button("🔍 Extract Data", variant="primary", size="lg")
            
        with gr.Column():
            # JSON output for API usage
            json_output = gr.JSON(
                label="📋 Extracted Data (JSON)",
                visible=True
            )
            
            # Formatted text output for human reading
            text_output = gr.Textbox(
                label="📝 Formatted Results",
                lines=15,
                max_lines=20,
                visible=True
            )
    
    # Example images section
    gr.Markdown("### 📝 Usage Notes:")
    gr.Markdown("""
    - For best results, ensure the ID card is well-lit and clearly visible
    - The image should be as straight as possible (not tilted)
    - Higher resolution images generally produce better results
    - This app supports both Romanian and English text recognition
    """)
    
    # Event handlers
    def process_and_format(image):
        result = process_romanian_id(image)
        formatted = format_output_for_display(result)
        return result, formatted
    
    extract_btn.click(
        fn=process_and_format,
        inputs=[image_input],
        outputs=[json_output, text_output]
    )
    
    # Auto-process when image is uploaded
    image_input.change(
        fn=process_and_format,
        inputs=[image_input],
        outputs=[json_output, text_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )