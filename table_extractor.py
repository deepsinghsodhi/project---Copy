# table_extractor.py
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

class TableExtractor:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def preprocess_image(self, image_path):
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Increase image size for better OCR
            image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply binary thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise
            kernel = np.ones((1,1), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return thresh

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise

    def clean_value(self, value):
        """Clean up extracted values"""
        if not value:
            return None
            
        # Remove common OCR artifacts
        value = value.replace('°', '')
        value = value.replace('·', '.')
        value = value.replace(',', '.')
        value = value.replace(';', '')
        value = value.replace(':', '')
        value = value.replace('|', '')
        value = value.replace('I', '1')
        value = value.replace('O', '0')
        value = value.replace('`', '')
        value = value.replace("'", '')
        
        # Remove any remaining special characters
        value = ''.join(c for c in value if c.isalnum() or c in '.-')
        
        try:
            # Try to convert to float if it's a number
            return float(value)
        except:
            return value.strip()

    def extract_table(self, image_path):
        try:
            # Preprocess image
            preprocessed = self.preprocess_image(image_path)
            
            # Configure tesseract for table data
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            # Extract text with additional parameters
            data = pytesseract.image_to_data(
                preprocessed, 
                config=custom_config, 
                output_type=pytesseract.Output.DATAFRAME
            )
            
            # Filter confident text only
            data = data[data['conf'] > 30]
            
            # Group by lines
            lines = []
            current_line = []
            current_line_num = -1

            for _, row in data.iterrows():
                if row['text'].strip():
                    if current_line_num != row['line_num']:
                        if current_line:
                            lines.append(current_line)
                        current_line = []
                        current_line_num = row['line_num']
                    current_line.append(row['text'].strip())
            
            if current_line:
                lines.append(current_line)

            # Process headers (first line)
            headers = self.process_headers(lines[0] if lines else [])
            
            # Process data rows
            data_rows = []
            for line in lines[1:]:
                row = [self.clean_value(val) for val in line]
                if any(row):  # Only add non-empty rows
                    data_rows.append(row)

            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Assign headers
            if not df.empty:
                # Ensure we have enough headers
                while len(headers) < len(df.columns):
                    headers.append(f'Column_{len(headers)+1}')
                df.columns = headers[:len(df.columns)]

            return df

        except Exception as e:
            print(f"Error extracting table: {e}")
            raise

    def process_headers(self, header_row):
        """Process and clean header row"""
        headers = []
        for header in header_row:
            # Clean header text
            clean_header = header.strip()
            clean_header = clean_header.replace('?', '')
            clean_header = clean_header.replace('·', '')
            clean_header = clean_header.replace(':', '')
            
            # Map common OCR mistakes in headers
            header_map = {
                'SDMP': '%DM',
                'SCFD': '%CFD',
                'SCP': '%CP',
                'ASP': 'ASP',
                'THR': 'THR',
                'SER': 'SER',
                'GLU': 'GLU',
                'PRO': 'PRO',
                'GLY': 'GLY',
                'ALA': 'ALA',
                'CYS': 'CYS',
                'VAL': 'VAL',
                'MET': 'MET',
                'ILE': 'ILE',
                'LEU': 'LEU',
                'TYR': 'TYR',
                'PHE': 'PHE',
                'HIS': 'HIS',
                'LYS': 'LYS',
                'ARG': 'ARG',
                'TRP': 'TRP'
            }
            
            clean_header = header_map.get(clean_header.upper(), clean_header)
            headers.append(clean_header)
            
        return headers

    def save_to_excel(self, df, output_path):
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Extracted Table', index=False)
                
                workbook = writer.book
                worksheet = writer.sheets['Extracted Table']
                
                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'bg_color': '#D9D9D9',
                    'border': 1
                })
                
                # Format headers
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Adjust column widths
                for idx, col in enumerate(df.columns):
                    max_len = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(idx, idx, max_len)
                    
            print(f"Table saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            raise