from flask import Flask, render_template, request, send_file, jsonify
from io import BytesIO
import os
import base64
import google.generativeai as genai
from werkzeug.utils import secure_filename
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
import shutil
from PIL import Image
from bs4 import BeautifulSoup
from flask import send_file
import zipfile
import io
import uuid
from flask_cors import CORS
import torch
import requests






# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)
UPLOAD_FOLDER = 'static/uploads'
EXTRACTED_FOLDER = 'static/extracted_tables'

UPLOAD_FOLDER = os.path.join('static', 'images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

STATIC_FOLDER = os.path.join('static', 'images')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/save_to_static', methods=['POST'])
def save_to_static():
    try:
        data = request.json
        filenames = data.get('filenames', [])
        
        if not filenames:
            return jsonify({'success': False, 'error': 'No files selected'})

        saved_files = []
        for filename in filenames:
            source_path = os.path.join('extracted_tables', filename)
            dest_path = os.path.join(STATIC_FOLDER, filename)
            
            if os.path.exists(source_path):
                # Copy the file to static folder
                shutil.copy2(source_path, dest_path)
                saved_files.append(filename)
            else:
                print(f"Source file not found: {source_path}")

        if saved_files:
            return jsonify({
                'success': True,
                'message': f'Saved {len(saved_files)} images to static folder',
                'saved_files': saved_files
            })
        else:
            return jsonify({'success': False, 'error': 'No files were saved'})

    except Exception as e:
        print(f"Error saving files: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, EXTRACTED_FOLDER]:
    os.makedirs(folder, exist_ok=True)
# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_FOLDER'] = EXTRACTED_FOLDER



# Replace this with your valid API key
GOOGLE_API_KEY = 'AIzaSyDjvrPdJcj3rrMHpNrjAQAy24NjV7XHN40' 
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model with Gemini 1.5 Flash
model = genai.GenerativeModel('gemini-1.5-flash')

def format_table_html(data):
    """Convert the extracted data into HTML table format with combined sample names"""
    lines = data.strip().split('\n')
    
    # Start HTML table
    html = '<table class="table table-bordered table-striped">\n'
    
    # Add headers
    html += '<thead>\n<tr>\n'
    # Add all headers from the original table
    headers = lines[0].split('|')
    headers = [h.strip() for h in headers if h.strip()]  # Remove empty elements and strip whitespace
    
    # Replace "Sample Name" with "Sample" in headers
    headers[0] = "Sample"
    
    for header in headers:
        html += f'<th>{header}</th>\n'
    html += '</tr>\n</thead>\n'
    
    # Add body
    html += '<tbody>\n'
    
    for line in lines[2:]:  # Skip header and separator lines
        if '|' in line:
            cells = line.split('|')
            cells = [cell.strip() for cell in cells if cell.strip()]  # Remove empty elements and strip whitespace
            
            if cells:  # Make sure we have cells to process
                html += '<tr>\n'
                for cell in cells:
                    html += f'<td>{cell}</td>\n'
                html += '</tr>\n'
            
    html += '</tbody>\n'
    html += '</table>'
    
    return html

@app.route('/')
def index():
    # Get list of images from extracted_tables directory
    if not os.path.exists(app.config['EXTRACTED_FOLDER']):
        os.makedirs(app.config['EXTRACTED_FOLDER'])
    
    images = [f for f in os.listdir(app.config['EXTRACTED_FOLDER']) 
             if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort images by page number and index
    images.sort(key=lambda x: tuple(map(int, ''.join(c if c.isdigit() else ' ' for c in x)
                                      .split())))
    
    return render_template('index.html', images=images)

# Make sure your static folder is properly configured
app.static_folder = 'static'

@app.route('/debug/images')
def debug_images():
    image_dir = os.path.join('static', 'images')
    try:
        files = os.listdir(image_dir)
        return jsonify({
            'image_dir': image_dir,
            'exists': os.path.exists(image_dir),
            'files': files,
            'readable_files': [f for f in files if os.access(os.path.join(image_dir, f), os.R_OK)]
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'image_dir': image_dir,
            'exists': os.path.exists(image_dir)
        })
    
@app.route('/extract_data', methods=['POST'])
def extract_data():
    try:
        selected_images = request.json.get('selected_images', [])
        results = []

        for image_name in selected_images:
            # Use the correct path with app.config['EXTRACTED_FOLDER']
            image_path = os.path.join(app.config['EXTRACTED_FOLDER'], image_name)
            print(f"Looking for image at: {image_path}")  # Debug print
            
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            try:
                # Open and process image
                with Image.open(image_path) as img:
                    # Ensure image is in RGB mode
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Resize image if needed
                    max_size = (1024, 1024)
                    img.thumbnail(max_size)
                    
                    # Convert image to bytes
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue()

                    # Generate prompt for Gemini
                    prompt = """
Analyze this table image and extract ALL data following these strict rules:

1. TABLE FORMAT CHECK:
   CRITICAL: Output format must be:
   | Sample | Column1 | Column2 | ... |
   | FirstRowFromImage | Value1 | Value2 | ... |    (Include Casein if present)
   | SecondRowFromImage | Value1 | Value2 | ... |
   NO OTHER ROWS ALLOWED

   IF ANY VALUES APPEAR IN FIRST COLUMN (amino acids, digestibility, AAS, PDCAAS):
   - Transform table to include ONLY the values present
   - Keep EXACT format: "Met + Cys" and "Phe + Tyr" (with spaces)
   - Format: | Sample | His | Lys | Met + Cys | Thr | Iso | Leu | Val | Phe + Tyr | Trp |
   - Move column headers (raw, broiled, smoked) to Sample column
   - Start data immediately after header row
   - Transform ALL values into rows
   - Include ALL values from original columns

   IF AMINO ACIDS ARE IN HEADER ROW:
   - Keep original format including italics for "Red Lentil" and "Green Lentil"
   - Include: %DM, %CF, %CP, ASP, THR, SER, GLU, PRO, GLY, ALA, CYS, VAL, MET, ILE, LEU, TYR, PHE, HIS, LYS, ARG, TRP
   - Start data immediately after header row
   - MUST include Casein row if present in image
   - Extract ALL rows exactly as they appear in image

   IF TABLE SHOWS PROTEIN SOURCES:
   - First column must be "Sample"
   - Second row onwards must use amino acid names exactly as shown (Asp², Thr, Ser, Glu³, etc.)
   - MUST include ALL following columns in this exact order:
     | Sample | Asp² | Thr | Ser | Glu³ | Ala | Cys | Val | Met | Ile | Leu | Tyr | Phe | His | Trp | Lys⁴ | Arg |
   - Keep exact superscript formatting
   - Include "Mean amino acid digestibility" and "Overall SEM⁵" rows
   - Transform protein source columns into rows
   - Extract ALL values for each amino acid and protein source

2. STRICT TABLE STRUCTURE:
   Row 1: Column headers only
   Row 2: First data row FROM IMAGE (including Casein if present)
   Following rows: All data rows FROM IMAGE
   NO blank lines
   NO separator rows
   NO dashes
   NO empty rows

3. SAMPLE NAMES:
   - Use italics for "Red Lentil" and "Green Lentil"
   - Combine with processing methods
   - Keep exact protein source names
   - For Type 3: Use amino acid names from first column
   - For first-column values: Use column headers as sample names
   - Start immediately after header
   - Extract names EXACTLY as shown in image
   - Include ALL protein sources when transforming

4. CRITICAL RULES:
   - Extract ONLY what appears in the image
   - Include Casein row if present
   - Transform ALL tables with values in first column
   - Transform ALL protein source columns to rows
   - NEVER include blank/empty rows
   - NO separator lines
   - Start data immediately after headers
   - Keep exact formatting

5. DO NOT:
   - Skip Casein row if present
   - Skip any protein source values
   - Skip transforming first-column value tables
   - Add blank rows
   - Include separator lines
   - Put space between header and data
   - Skip any values from image

VERIFY BEFORE RESPONDING:
- Is Casein included if present in image?
- Are ALL protein sources transformed to rows?
- Are tables with first-column values transformed?
- Are ALL values present as shown in image?
- Is formatting preserved exactly?
- Are there NO blank rows
 """

                    # Create generation config for Flash model
                    generation_config = {
                        'temperature': 0.1,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }

                    try:
                        # Get response from Gemini
                        response = model.generate_content(
                            contents=[{
                                'parts': [
                                    {'text': prompt},
                                    {'inline_data': {
                                        'mime_type': 'image/jpeg',
                                        'data': base64.b64encode(img_bytes).decode()
                                    }}
                                ]
                            }],
                            generation_config=generation_config
                        )
                        
                        # Process the response and convert to HTML
                        extracted_text = response.text if response.text else "No data extracted"
                        formatted_html = format_table_html(extracted_text)
                        
                        # Add result to list
                        results.append({
                            'image_name': image_name,
                            'extracted_data': formatted_html
                        })

                    except Exception as e:
                        print(f"Gemini API error: {str(e)}")
                        results.append({
                            'image_name': image_name,
                            'extracted_data': f'<p class="error">Error processing image: {str(e)}</p>'
                        })

                    # Clean up
                    buffered.close()

            except Exception as e:
                print(f"Error processing image {image_name}: {str(e)}")
                results.append({
                    'image_name': image_name,
                    'extracted_data': f'<p class="error">Error processing image: {str(e)}</p>'
                })

        if not results:
            return jsonify({'success': False, 'error': 'No results generated'}), 400

        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        print(f"Error during extraction: {str(e)}")  # Server-side logging
        return jsonify({'success': False, 'error': str(e)})







@app.route('/get_saved_images')
def get_saved_images():
    try:
        static_images_path = os.path.join('static', 'images')
        if not os.path.exists(static_images_path):
            os.makedirs(static_images_path)
            
        # Only get image files and sort them
        images = sorted([
            f for f in os.listdir(static_images_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
        ])
        
        # Get file information
        image_data = []
        for image in images:
            file_path = os.path.join(static_images_path, image)
            file_size = os.path.getsize(file_path)
            image_data.append({
                'name': image,
                'size': file_size,
                'modified': os.path.getmtime(file_path)
            })
                 
        return jsonify({
            'success': True,
            'images': images,
        })
    except Exception as e:
        print(f"Error getting saved images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    






# Add this route to serve images
@app.route('/static/extracted_tables/<filename>')
def serve_extracted_table(filename):
    return send_file(os.path.join(app.config['EXTRACTED_FOLDER'], filename))

def extract_tables(pdf_path):

    # Initialize the processor and model
    image_processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection",
        use_fast=True
    )
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection",
        ignore_mismatched_sizes=True
    )
    
    # Convert PDF to images
    pdf_images = convert_from_path(pdf_path)
    extracted_tables = []
    
    # Process each page
    for page_num, image in enumerate(pdf_images, start=1):
        # Process the image
        inputs = image_processor(
            images=image, 
            return_tensors="pt",
            size={
                "shortest_edge": 600,
                "longest_edge": 800
            }
        )
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]
        
        # Process detected tables
        for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            if score >= 0.8:  # High confidence threshold
                box = [int(i) for i in box.tolist()]
                xmin, ymin, xmax, ymax = box
                
                # Sort tables by their vertical position on the page
                y_position = ymin
                
                # Crop and save the table
                cropped_table = image.crop((xmin, ymin, xmax, ymax))
                cropped_table = cropped_table.convert('RGB')
                table_filename = f"table_page_{page_num}_idx_{idx}.jpg"
                save_path = os.path.join(app.config['EXTRACTED_FOLDER'], table_filename)
                cropped_table.save(save_path, 'JPEG', quality=95)
                
                extracted_tables.append({
                    'filename': table_filename,
                    'path': f'/static/extracted_tables/{table_filename}',
                    'page': page_num,
                    'y_position': y_position,
                    'confidence': float(score)
                })
    
    # Sort tables by page number and vertical position
    extracted_tables.sort(key=lambda x: (x['page'], x['y_position']))
    
    return extracted_tables

@app.route('/get_table_captions')
def get_table_captions():
    try:
        # Update the file path to match get_table_data route
        file_path = "C:/web_application/extractinghtml/nutrition-labelling.html"
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")
            return jsonify({'error': f'File not found at: {file_path}'}), 404

        # Try different encodings
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    break
            except UnicodeDecodeError:
                continue

        if content is None:
            print("Failed to read file with any encoding")
            return jsonify({'error': 'Unable to read file with supported encodings'}), 500

        soup = BeautifulSoup(content, 'html.parser')
        
        # Find the ordered list with class 'lst-upr-alph'
        ol = soup.find('ol', class_='lst-upr-alph')
        if not ol:
            print("No ordered list found with class 'lst-upr-alph'")
            return jsonify({'error': 'Categories list not found'}), 404

        # Get all list items with their links
        captions = []
        for li in ol.find_all('li'):
            link = li.find('a')
            if link and link.get('href', '').startswith('#'):
                href = link.get('href')
                text = link.text.strip()
                
                # Extract letter and caption
                parts = text.split('.', 1)
                if len(parts) == 2:
                    letter = parts[0].strip()
                    caption = parts[1].strip()
                    
                    captions.append({
                        'id': href[1:],  # Remove the # from href
                        'letter': letter,
                        'text': text  # Use the full text for display
                    })
        
        print(f"Found {len(captions)} captions: {captions}")  # Debug print
        return jsonify(captions)
        
    except Exception as e:
        print(f"Error getting captions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_table_data', methods=['POST'])
def get_table_data():
    try:
        table_id = request.form.get('table_id')
        if not table_id:
            return jsonify({'error': 'No table ID provided'})
            
        print(f"Received table_id: {table_id}")  # Debug print
            
        file_path = "C:/web_application/extractinghtml/nutrition-labelling.html"

        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            
        # Debug print all tables and their IDs
        all_tables = soup.find_all('table')
        print(f"Found {len(all_tables)} tables")
        for idx, table in enumerate(all_tables):
            table_id_attr = table.get('id', 'no-id')
            print(f"Table {idx}: id='{table_id_attr}'")
            
        # First try exact match (case-insensitive)
        target_table = soup.find('table', id=table_id.lower())
        
        if not target_table:
            # Try finding table by ID without case sensitivity
            for table in all_tables:
                if table.get('id', '').lower() == table_id.lower():
                    target_table = table
                    break
                    
        if not target_table:
            return jsonify({'error': f'Table with ID "{table_id}" not found. Available IDs: {[t.get("id", "no-id") for t in all_tables]}'})
            
        # Get the caption (section title)
        caption = target_table.find('caption')
        section_title = caption.get_text(strip=True) if caption else f'Section {table_id.upper()}'
            
        # Extract data from the table - only second and third columns
        table_data = []
        rows = target_table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:  # Ensure we have at least 3 cells
                description = cells[1].get_text(strip=True)  # Second column
                value = cells[2].get_text(strip=True)       # Third column
                if description and value:
                    table_data.append({
                        'description': description,
                        'value': value
                    })
        
        if not table_data:
            return jsonify({'error': 'No data found in the table'})
            
        # Create HTML response
        html_response = f'''
        <div class="table-container" style="margin: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
            <h3 style="margin-bottom: 15px;">{section_title}</h3>
            <table class="table table-bordered table-striped">
                <thead>
                    <tr>
                        <th>Description</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for item in table_data:
            html_response += f'''
                <tr>
                    <td>{item['description']}</td>
                    <td>{item['value']}</td>
                </tr>
            '''
            
        html_response += '''
                </tbody>
            </table>
        </div>
        '''
        
        return jsonify({'html': html_response})
        
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return jsonify({'error': f'Error processing request: {str(e)}'})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Create directories if they don't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['EXTRACTED_FOLDER'], exist_ok=True)
            
            # Save uploaded file with timestamp
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Extract tables
            extracted_tables = extract_tables(file_path)
            
            # Remove the uploaded PDF file after processing
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove uploaded file: {str(e)}")
            
            if not extracted_tables:
                return jsonify({'error': 'No tables found in the PDF'}), 400
                
            return jsonify({'success': True, 'tables': extracted_tables})
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


    
@app.route('/process_table_data', methods=['POST'])
def process_table_data():
    try:
        # Get the table data from the request
        data = request.json.get('tableData')
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Process the data into a structured format
        rows = data.strip().split('\n')
        headers = ['']  # First column has no header
        table_data = []
        
        # Process headers (first row)
        if rows:
            header_cells = rows[0].split()
            # Add headers starting from second value
            headers.extend(header_cells[1:])
        
        # Process data rows
        for row in rows[1:]:
            cells = row.split()
            if len(cells) > 1:  # Ensure row has enough data
                row_dict = {
                    '': cells[1]  # First column (text)
                }
                # Process remaining cells
                for i, value in enumerate(cells[2:], start=0):
                    if i < len(headers[1:]):
                        try:
                            # Try to convert to number for non-first columns
                            if '+' in value:
                                row_dict[headers[i+1]] = value
                            else:
                                row_dict[headers[i+1]] = float(value.replace(',', ''))
                        except ValueError:
                            row_dict[headers[i+1]] = value
                
                table_data.append(row_dict)

        return jsonify({
            'success': True,
            'data': table_data,
        })

    except Exception as e:
        print(f"Error processing table data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    



@app.route('/download_images', methods=['POST'])
@app.route('/save_image', methods=['POST'])
def save_image():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file'})
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
            
        # Generate unique filename
        filename = f"table_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        
        # Save the image
        image_file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': f'/static/images/{filename}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_images', methods=['POST'])

def download_images():
    try:
        selected_images = request.json.get('selected_images', [])
        
        if not selected_images:
            return jsonify({'success': False, 'error': 'No images selected'})

        # Create a memory file for the zip
        memory_file = io.BytesIO()
        
        # Create the zip file
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for image_name in selected_images:
                image_path = os.path.join(UPLOAD_FOLDER, image_name)
                if os.path.exists(image_path):
                    zf.write(image_path, image_name)
        
        # Prepare the memory file for reading
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='table_images.zip'
        )

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    


    
@app.route('/download_selected', methods=['POST'])
def download_selected():
    try:
        selected_images = request.json.get('selected_images', [])
        
        if not selected_images:
            return jsonify({'success': False, 'error': 'No images selected'})

        # Create a memory file for the zip
        memory_file = io.BytesIO()
        
        # Create the zip file
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for image_name in selected_images:
                image_path = os.path.join('static/images', image_name)
                if os.path.exists(image_path):
                    zf.write(image_path, image_name)
        
        # Prepare the memory file for reading
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='selected_images.zip'
        )

    except Exception as e:
        print(f"Download error: {str(e)}")  # For debugging
        return jsonify({'success': False, 'error': str(e)})
        
@app.route('/download_tables', methods=['POST'])
def download_tables():
    try:
        data = request.json
        filenames = data.get('filenames', [])
        
        if not filenames:
            return jsonify({'success': False, 'error': 'No files selected'})

        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for filename in filenames:
                file_path = os.path.join('extracted_tables', filename)
                if os.path.exists(file_path):
                    zf.write(file_path, filename)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='selected_tables.zip'
        )

    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    

@app.route('/get_tables')
def get_tables():
    try:
        tables = []
        if os.path.exists(app.config['EXTRACTED_FOLDER']):
            for filename in os.listdir(app.config['EXTRACTED_FOLDER']):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # Extract page and index from filename (e.g., table_page_1_idx_2.jpg)
                    parts = filename.split('_')
                    page = int(parts[2])
                    index = int(parts[4].split('.')[0])
                    tables.append({
                        'filename': filename,
                        'page': page,
                        'index': index
                    })
        
        # Sort tables by page and index
        tables.sort(key=lambda x: (x['page'], x['index']))
        return jsonify({'success': True, 'tables': tables})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_table', methods=['POST'])
def process_table():
    try:
        content = request.json.get('content')
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})

        # Process the table content (implement your processing logic here)
        processed_content = process_table_content(content)
        return jsonify({'success': True, 'result': processed_content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/calculate_combined', methods=['POST'])
def calculate_combined():
    try:
        content = request.json.get('content')
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})

        # Calculate MET+CYS and PHE+TYR (implement your calculation logic here)
        result = calculate_amino_acids(content)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Add similar routes for other processing functions
@app.route('/drop_pdcaas', methods=['POST'])
def drop_pdcaas():
    try:
        content = request.json.get('content')
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})

        # Parse the HTML content
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        if not table:
            return jsonify({'success': False, 'error': 'No table found'})

        # Get headers
        headers = [th.text.strip() for th in table.find('tr').find_all(['th', 'td'])]
        
        # Identify columns to keep (exclude PDCAAS-related columns)
        keep_columns = []
        for i, header in enumerate(headers):
            if 'pdcaas' not in header.lower():
                keep_columns.append(i)

        # Create new table
        html = '<table class="table table-bordered table-striped">\n<thead>\n<tr>\n'
        # Add filtered headers
        for i in keep_columns:
            html += f'<th>{headers[i]}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'

        # Process each row
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            html += '<tr>\n'
            for i in keep_columns:
                cell_content = cells[i].text.strip()
                try:
                    value = float(cell_content)
                    html += f'<td>{value:.2f}</td>\n'
                except ValueError:
                    html += f'<td>{cell_content}</td>\n'
            html += '</tr>\n'

        html += '</tbody>\n</table>'
        return jsonify({'success': True, 'result': html})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_cp_values', methods=['POST'])
def save_cp_values():
    try:
        content = request.json.get('content')
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})

        # Parse the HTML content
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        if not table:
            return jsonify({'success': False, 'error': 'No table found'})

        # Get headers
        headers = [th.text.strip() for th in table.find('tr').find_all(['th', 'td'])]
        
        # Find CP column index
        cp_index = None
        for i, header in enumerate(headers):
            if '%cp' in header.lower() or 'crude protein' in header.lower() or 'pro' in header.lower():
                cp_index = i
                break

        if cp_index is None:
            return jsonify({'success': False, 'error': 'No CP column found'})

        # Create new table with highlighted CP column
        html = '<table class="table table-bordered table-striped">\n<thead>\n<tr>\n'
        
        # Add headers with CP column highlighted
        for i, header in enumerate(headers):
            if i == cp_index:
                html += f'<th class="highlighted-column">{header}</th>\n'
            else:
                html += f'<th>{header}</th>\n'
        html += '</tr>\n</thead>\n<tbody>\n'

        # Process each row
        for row in table.find_all('tr')[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            html += '<tr>\n'
            for i, cell in enumerate(cells):
                cell_content = cell.text.strip()
                try:
                    value = float(cell_content)
                    if i == cp_index:
                        html += f'<td class="highlighted-column">{value:.2f}</td>\n'
                    else:
                        html += f'<td>{value:.2f}</td>\n'
                except ValueError:
                    if i == cp_index:
                        html += f'<td class="highlighted-column">{cell_content}</td>\n'
                    else:
                        html += f'<td>{cell_content}</td>\n'
            html += '</tr>\n'

        html += '</tbody>\n</table>\n'
        
        # Add CSS for highlighted column
        html = '''
        <style>
        .highlighted-column {
            background-color: #e8f5e9 !important;
            font-weight: bold;
        }
        </style>
        ''' + html
        
        return jsonify({'success': True, 'result': html})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/calculate_eaa', methods=['POST'])
def calculate_eaa():
    try:
        content = request.json.get('content')
        if not content:
            return jsonify({'success': False, 'error': 'No content provided'})

        # Parse the HTML content
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        if not table:
            return jsonify({'success': False, 'error': 'No table found in content'})

        # Get headers
        headers = [th.text.strip() for th in table.find('tr').find_all(['th', 'td'])]
        
        # Find CP column index
        cp_index = -1
        for i, header in enumerate(headers):
            if '%CP' in header.upper():
                cp_index = i
                break

        if cp_index == -1:
            return jsonify({'success': False, 'error': 'Cannot perform calculation. Missing %CP column.'})

        # All amino acids to look for (in order)
        amino_acids = ['THR', 'SER', 'GLU', 'PRO', 'GLY', 'ALA', 'VAL', 'ILE', 'LEU', 'HIS', 'LYS', 'ARG', 'TRP', 'MET+CYS', 'PHE+TYR']
        aa_indices = {}
        
        # Find amino acid column indices
        for i, header in enumerate(headers):
            header_upper = header.upper()
            for aa in amino_acids:
                if aa in header_upper:
                    aa_indices[aa] = i

        # Create new table with amino acid columns
        html = '''
        <div>
            <h3>Essential Amino Acids Calculation Results</h3>
            <table id="aminoAcidsTable" class="table table-bordered table-striped">
            <thead><tr>
            <th class="text-column">Sample</th>
        '''
        
        for aa in amino_acids:
            html += f'<th class="numeric-column">{aa}</th>'
        html += '</tr></thead><tbody>'

        # Process each row
        rows = table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue

            sample_name = cells[0].text.strip()
            try:
                cp_value = float(cells[cp_index].text.strip())
            except (ValueError, IndexError):
                continue
            
            html += f'<tr><td class="text-column">{sample_name}</td>'
            
            # Calculate values for each amino acid
            for aa in amino_acids:
                if aa in aa_indices:
                    try:
                        acid_value = float(cells[aa_indices[aa]].text.strip())
                        calculated_value = (acid_value * 1000) / cp_value
                        html += f'<td class="numeric-column">{calculated_value:.3f}</td>'
                    except (ValueError, IndexError):
                        html += '<td class="numeric-column">N/A</td>'
                else:
                    html += '<td class="numeric-column">N/A</td>'
            
            html += '</tr>'

        html += '''
            </tbody>
            </table>
            <div style="display: flex; gap: 10px; margin-top: 20px; align-items: center;">
                <label>Select Reference Pattern:</label>
                <div class="requirements-selector">
                    <label>Protein Quality Report 1991:</label>
                    <select id="indianAgeGroupSelect" class="age-group-select">
                        <option value="">Select Age Group</option>
                        <option value="Infant">Infant</option>
                        <option value="PreSchool_Child_2_5_Yrs">PreSchool Child (2-5 Yrs)</option>
                        <option value="School_Child_10_12_Yrs">School Child (10-12 Yrs)</option>
                        <option value="Adult">Adult</option>
                    </select>
                </div>
                <div class="requirements-selector">
                    <label>Protein Quality Report 2007:</label>
                    <select id="whoAgeGroupSelect" class="age-group-select">
                        <option value="">Select Age Group</option>
                        <option value="0.50">0.50 yr</option>
                        <option value="1-2">1-2 yr</option>
                        <option value="3-10">3-10 yr</option>
                        <option value="11-14">11-14 yr</option>
                        <option value="15-18">15-18 yr</option>
                        <option value=">18">>18 yr</option>
                    </select>
                </div>
                <div class="requirements-selector">
                    <label>Protein Quality Report 2013:</label>
                    <select id="faoWhoUnuSelect" class="age-group-select">
                        <option value="">Select Age Group</option>
                        <option value="infant">Infant (0-6 mo)</option>
                        <option value="child">Child (6mo-3 yr)</option>
                        <option value="above3">> 3 yr</option>
                    </select>
                </div>
                <button id="multiplyButton" class="multiply-btn" disabled>
                    Calculate
                </button>
            </div>
            <div id="requirementsTableContainer"></div>
            <div id="multiplicationResultContainer"></div>
        </div>
        <style>
        .text-column {
            text-align: left;
        }
        .numeric-column {
            text-align: right;
        }
        .requirements-selector {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .age-group-select {
            padding: 5px;
            border-radius: 4px;
        }
        .multiply-btn {
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .multiply-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        </style>
        '''

        return jsonify({'success': True, 'result': html})

    except Exception as e:
        print(f"Error calculating essential amino acids: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/calculate_eaa', methods=['POST'])
# def calculate_eaa():
#     try:
#         content = request.json.get('content')
#         if not content:
#             return jsonify({'success': False, 'error': 'No content provided'})

#         # Parse the HTML content
#         soup = BeautifulSoup(content, 'html.parser')
#         table = soup.find('table')
#         if not table:
#             return jsonify({'success': False, 'error': 'No table found in content'})

#         # Get headers
#         headers = [th.text.strip() for th in table.find('tr').find_all(['th', 'td'])]
        
#         # Find CP column index
#         cp_index = -1
#         cp_variants = ['CP (%)', 'CP(%)', 'CRUDE PROTEIN', '%CP', '%CPB', '%CPC', '%CPP', 'CP', '%CP ']
#         for i, header in enumerate(headers):
#             header_upper = header.upper().strip()
#             if any(variant.upper() in header_upper for variant in cp_variants):
#                 cp_index = i
#                 break

#         if cp_index == -1:
#             return jsonify({'success': False, 'error': f'Cannot find CP column. Available headers: {headers}'})

#         # Updated amino acid mappings to include combined values
#         amino_acid_mappings = {
#             'THR': ['THR', 'THREONINE'],
#             'VAL': ['VAL', 'VALINE'],
#             'MET': ['MET', 'METHIONINE'],
#             'ILE': ['ILE', 'ISOLEUCINE'],
#             'LEU': ['LEU', 'LEUCINE'],
#             'PHE': ['PHE', 'PHENYLALANINE'],
#             'HIS': ['HIS', 'HISTIDINE', 'HTS'],
#             'LYS': ['LYS', 'LYSINE'],
#             'TRP': ['TRP', 'TRYPTOPHAN'],
#             'MET+CYS': ['MET+CYS', 'METHIONINE+CYSTEINE', 'MET + CYS'],
#             'PHE+TYR': ['PHE+TYR', 'PHENYLALANINE+TYROSINE', 'PHE + TYR']
#         }

#         # Find amino acid column indices
#         aa_indices = {}
#         for i, header in enumerate(headers):
#             header_upper = header.upper().strip()
#             for short_form, variants in amino_acid_mappings.items():
#                 if any(variant == header_upper for variant in variants):
#                     aa_indices[short_form] = i
#                     break

#         # Create new table
#         html = '''
#         <div>
#             <h3>Essential Amino Acids Calculation Results</h3>
#             <table id="aminoAcidsTable" class="table table-bordered table-striped">
#             <thead><tr>
#             <th class="text-column">Sample</th>
#         '''
        
#         # Define the order of amino acids in the output
#         ordered_aas = ['THR', 'VAL', 'MET', 'ILE', 'LEU', 'PHE', 'HIS', 'LYS', 'TRP', 'MET+CYS', 'PHE+TYR']
        
#         for aa in ordered_aas:
#             html += f'<th class="numeric-column">{aa}</th>'
#         html += '</tr></thead><tbody>'

#         # Process each row
#         rows = table.find_all('tr')[1:]  # Skip header row
#         for row in rows:
#             cells = row.find_all(['td', 'th'])
#             if not cells:
#                 continue

#             sample_name = cells[0].text.strip()
#             if sample_name == '---' or sample_name == '' or all(cell.text.strip() in ['---', ''] for cell in cells):
#                 continue

#             try:
#                 cp_text = cells[cp_index].text.strip()
#                 cp_text = ''.join(c for c in cp_text if c.isdigit() or c == '.')
#                 cp_value = float(cp_text)
#             except (ValueError, IndexError) as e:
#                 print(f"Error processing CP value for {sample_name}: {str(e)}")
#                 continue
            
#             html += f'<tr><td class="text-column">{sample_name}</td>'
            
#             # Calculate values for each amino acid
#             for aa_short in ordered_aas:
#                 if aa_short in aa_indices:
#                     try:
#                         aa_index = aa_indices[aa_short]
#                         acid_text = cells[aa_index].text.strip()
#                         if acid_text and acid_text not in ['N/A', '---']:
#                             acid_text = ''.join(c for c in acid_text if c.isdigit() or c == '.')
#                             acid_value = float(acid_text)
#                             # calculated_value = (acid_value * 1000) / cp_value
#                             calculated_value = acid_value 
#                             html += f'<td class="numeric-column">{calculated_value:.2f}</td>'
#                         else:
#                             html += '<td class="numeric-column">N/A</td>'
#                     except (ValueError, IndexError) as e:
#                         print(f"Error calculating {aa_short} for {sample_name}: {str(e)}")
#                         html += '<td class="numeric-column">N/A</td>'
#                 elif aa_short == 'MET+CYS' and 'MET' in aa_indices and 'CYS' in aa_indices:
#                     # Calculate MET+CYS if individual values are available
#                     try:
#                         met_value = float(cells[aa_indices['MET']].text.strip())
#                         cys_value = float(cells[aa_indices['CYS']].text.strip())
#                         combined_value = ((met_value + cys_value) * 1000) / cp_value
#                         html += f'<td class="numeric-column">{combined_value:.2f}</td>'
#                     except (ValueError, IndexError):
#                         html += '<td class="numeric-column">N/A</td>'
#                 elif aa_short == 'PHE+TYR' and 'PHE' in aa_indices and 'TYR' in aa_indices:
#                     # Calculate PHE+TYR if individual values are available
#                     try:
#                         phe_value = float(cells[aa_indices['PHE']].text.strip())
#                         tyr_value = float(cells[aa_indices['TYR']].text.strip())
#                         combined_value = ((phe_value + tyr_value) * 1000) / cp_value
#                         html += f'<td class="numeric-column">{combined_value:.2f}</td>'
#                     except (ValueError, IndexError):
#                         html += '<td class="numeric-column">N/A</td>'
#                 else:
#                     html += '<td class="numeric-column">N/A</td>'
            
#             html += '</tr>'

#         html += '''
#             </tbody>
#             </table>
#             <div style="display: flex; gap: 10px; margin-top: 20px; align-items: center;">
#                 <label>Select Reference Pattern:</label>
#                 <div class="requirements-selector">
#                     <label>Protein Quality Report 1991:</label>
#                     <select id="indianAgeGroupSelect" class="age-group-select">
#                         <option value="">Select Age Group</option>
#                         <option value="Infant">Infant</option>
#                         <option value="PreSchool_Child_2_5_Yrs">PreSchool Child (2-5 Yrs)</option>
#                         <option value="School_Child_10_12_Yrs">School Child (10-12 Yrs)</option>
#                         <option value="Adult">Adult</option>
#                     </select>
#                 </div>
#                 <div class="requirements-selector">
#                     <label>Protein Quality Report 2007:</label>
#                     <select id="whoAgeGroupSelect" class="age-group-select">
#                         <option value="">Select Age Group</option>
#                         <option value="0.50">0.50 yr</option>
#                         <option value="1-2">1-2 yr</option>
#                         <option value="3-10">3-10 yr</option>
#                         <option value="11-14">11-14 yr</option>
#                         <option value="15-18">15-18 yr</option>
#                         <option value=">18">>18 yr</option>
#                     </select>
#                 </div>
#                 <div class="requirements-selector">
#                     <label>Protein Quality Report 2013:</label>
#                     <select id="faoWhoUnuSelect" class="age-group-select">
#                         <option value="">Select Age Group</option>
#                         <option value="infant">Infant (0-6 mo)</option>
#                         <option value="child">Child (6mo-3 yr)</option>
#                         <option value="above3">> 3 yr</option>
#                     </select>
#                 </div>
#                 <button id="multiplyButton" class="multiply-btn" disabled>
#                     Calculate
#                 </button>
#             </div>
#             <div id="requirementsTableContainer"></div>
#             <div id="multiplicationResultContainer"></div>

#         </div>
#         <style>
#         .text-column {
#             text-align: left;
#             font-weight: bold;
#         }
#         .numeric-column {
#             text-align: right;
#         }
#         .requirements-selector {
#             display: flex;
#             flex-direction: column;
#             gap: 5px;
#         }
#         .age-group-select {
#             padding: 5px;
#             border-radius: 4px;
#         }
#         .multiply-btn {
#             padding: 8px 16px;
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#         }
#         .multiply-btn:disabled {
#             background-color: #cccccc;
#             cursor: not-allowed;
#         }
#         table {
#             border-collapse: collapse;
#             width: 100%;
#         }
#         th, td {
#             padding: 8px;
#             border: 1px solid #ddd;
#         }
#         thead th {
#             background-color: #f5f5f5;
#         }
#         tbody tr:nth-child(even) {
#             background-color: #f9f9f9;
#         }
#         </style>
#         '''

#         return jsonify({'success': True, 'result': html})

#     except Exception as e:
#         print(f"Error calculating essential amino acids: {str(e)}")
#         return jsonify({'success': False, 'error': str(e)})

@app.route('/calculate_pdcaas', methods=['POST'])
def calculate_pdcaas():
    try:
        data = request.json
        content = data.get('content')
        aas_values = data.get('aasValues')
        
        if not content or not aas_values:
            return jsonify({'success': False, 'error': 'Missing required data'})

        print("Received AAS values:", aas_values)  # Debug print

        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return jsonify({'success': False, 'error': 'No table found in content'})

        # Get headers
        headers = [th.text.strip() for th in table.find('tr').find_all(['th', 'td'])]
        
        # Find required column indices
        tpd_index = -1
        ivpd_index = -1
        
        for i, header in enumerate(headers):
            header_upper = header.upper()
            if 'TPD' in header_upper or '%TPD' in header_upper:
                tpd_index = i
            elif 'IVPD' in header_upper or '%IVPD' in header_upper:
                ivpd_index = i

        if tpd_index == -1 or ivpd_index == -1:
            return jsonify({'success': False, 'error': 'Missing required columns (TPD/IVPD)'})

        # Create new table with calculations
        result_html = '''
        <div class="pdcaas-container">
            <table class="table table-bordered table-striped">
                <thead><tr>
        '''
        
        # Add headers
        for header in headers:
            result_html += f'<th>{header}</th>'
        result_html += '<th>PDCAAS</th><th>IVPDCAAS</th></tr></thead><tbody>'

        # Process each row
        rows = table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cells = row.find_all(['td', 'th'])
            sample_name = cells[0].text.strip()
            
            # Function to normalize sample name
            def normalize_name(name):
                return name.lower().replace('*', '').replace(' ', '')
            
            # Get AAS value for the sample
            aas_value = None
            normalized_sample = normalize_name(sample_name)
            
            # Try to find matching AAS value
            for stored_name, value in aas_values.items():
                if normalize_name(stored_name) == normalized_sample:
                    aas_value = value
                    print(f"Found match: {sample_name} -> {stored_name} = {value}")  # Debug print
                    break
            
            result_html += '<tr>'
            
            # Copy existing cell values
            for cell in cells:
                result_html += f'<td>{cell.text.strip()}</td>'
            
            try:
                # Calculate PDCAAS and IVPDCAAS
                if aas_value is not None:
                    tpd = float(cells[tpd_index].text.strip()) / 100
                    ivpd = float(cells[ivpd_index].text.strip()) / 100
                    
                    pdcaas = aas_value * tpd * 100
                    ivpdcaas = aas_value * ivpd * 100
                    
                    result_html += f'<td>{pdcaas:.3f}</td>'
                    result_html += f'<td>{ivpdcaas:.3f}</td>'
                else:
                    print(f"No AAS value found for {sample_name}")  # Debug print
                    result_html += '<td>N/A</td><td>N/A</td>'
            except (ValueError, IndexError) as e:
                print(f"Error processing row {sample_name}: {str(e)}")  # Debug print
                result_html += '<td>N/A</td><td>N/A</td>'
            
            result_html += '</tr>'

        result_html += '''
                </tbody>
            </table>
            
            <div class="reference-selection mt-4">
                <div class="form-group">
                    <label for="tableSelect">Select Table:</label>
                    <select class="form-control" id="tableSelect">
                        <option value="">Select a category...</option>
                        <option value="a">A. Bakery products and substitutes</option>
                        <option value="b">B. Beverages</option>
                        <option value="c">C. Cereals, other grain products and substitutes</option>
                        <option value="d">D. Dairy products and substitutes</option>
                        <option value="e">E. Desserts</option>
                        <option value="f">F. Dessert toppings and fillings</option>
                        <option value="g">G. Eggs and egg substitutes</option>
                        <option value="h">H. Fats and oils</option>
                        <option value="i">I. Marine and fresh water animals and substitutes</option>
                        <option value="j">J. Fruit and fruit juices</option>
                        <option value="k">K. Legumes</option>
                        <option value="l">L. Meat, poultry, their products and substitutes</option>
                        <option value="m">M. Miscellaneous category</option>
                        <option value="n">N. Combination dishes including main dishes</option>
                        <option value="o">O. Nuts and seeds</option>
                        <option value="p">P. Potatoes, sweet potatoes and yams</option>
                    </select>
                </div>
                <div id="selectedTableData"></div>
            </div>

            <style>
            .pdcaas-container {
                margin-bottom: 20px;
            }
            .reference-selection {
                margin-top: 20px;
            }
            .form-control {
                width: 100%;
                max-width: 500px;
                padding: 8px;
                margin: 10px 0;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
            #selectedTableData {
                margin-top: 20px;
            }
            </style>

            <script>
            document.addEventListener('DOMContentLoaded', function() {
                const tableSelect = document.getElementById('tableSelect');
                
                tableSelect.addEventListener('change', function() {
                    if (this.value) {
                        fetch(`/get_table_data/${this.value}`)
                            .then(response => response.json())
                            .then(data => {
                                const container = document.getElementById('selectedTableData');
                                if (data.error) {
                                    container.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                                    return;
                                }
                                
                                if (data.html) {
                                    container.innerHTML = data.html;
                                } else {
                                    container.innerHTML = '<div class="alert alert-warning">No data available</div>';
                                }
                            })
                            .catch(error => {
                                console.error('Error:', error);
                                document.getElementById('selectedTableData').innerHTML = 
                                    '<div class="alert alert-danger">Error loading table data</div>';
                            });
                    } else {
                        document.getElementById('selectedTableData').innerHTML = '';
                    }
                });
            });
            </script>
        </div>
        '''

        return jsonify({'success': True, 'result': result_html})
        
    except Exception as e:
        print(f"Error calculating PDCAAS: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def process_table_content(content):
    try:
        # Split content into lines
        lines = content.strip().split('\n')
        if not lines:
            return '<p class="error">No content to process</p>'

        # Start HTML table
        html = '<table class="table table-bordered table-striped">\n<thead>\n<tr>\n'
        
        # Process headers (first row)
        headers = lines[0].split('\t')
        for header in headers:
            html += f'<th>{header.strip()}</th>\n'
        
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Process data rows
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                cells = line.split('\t')
                html += '<tr>\n'
                for cell in cells:
                    # Remove values in brackets
                    cell_value = cell.strip()
                    if '(' in cell_value and ')' in cell_value:
                        cell_value = cell_value.split('(')[0].strip()
                    
                    # Try to format numbers with 2 decimal places
                    try:
                        value = float(cell_value)
                        html += f'<td>{value:.2f}</td>\n'
                    except ValueError:
                        html += f'<td>{cell_value}</td>\n'
                html += '</tr>\n'
        
        html += '</tbody>\n</table>'
        return html
        
    except Exception as e:
        print(f"Error processing table content: {str(e)}")
        return f'<p class="error">Error processing table: {str(e)}</p>'

def calculate_amino_acids(content):
    try:
        # Parse the HTML content
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        if not table:
            return '<p class="error">No table found in content</p>'

        rows = table.find_all('tr')
        headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
        
        # Find indices for Met, Cys, Phe, and Tyr
        met_index = None
        cys_index = None
        phe_index = None
        tyr_index = None
        last_aa_index = 0
        
        # Find indices and track the last amino acid column
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if 'methionine' in header_lower or 'met' in header_lower:
                met_index = i
                last_aa_index = max(last_aa_index, i)
            elif 'cysteine' in header_lower or 'cys' in header_lower:
                cys_index = i
                last_aa_index = max(last_aa_index, i)
            elif 'phenylalanine' in header_lower or 'phe' in header_lower:
                phe_index = i
                last_aa_index = max(last_aa_index, i)
            elif 'tyrosine' in header_lower or 'tyr' in header_lower:
                tyr_index = i
                last_aa_index = max(last_aa_index, i)

        # Create list of columns to keep (excluding individual amino acids)
        keep_columns = []
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if i not in [met_index, cys_index, phe_index, tyr_index]:
                keep_columns.append(i)

        # Create new table
        html = '<table class="table table-bordered table-striped">\n<thead>\n<tr>\n'
        
        # Add headers up to the last amino acid position
        for i in range(last_aa_index + 1):
            if i in keep_columns:
                html += f'<th>{headers[i]}</th>\n'
        
        # Add combined headers
        if met_index is not None and cys_index is not None:
            html += '<th>MET+CYS</th>\n'
        if phe_index is not None and tyr_index is not None:
            html += '<th>PHE+TYR</th>\n'
        
        # Add remaining headers after the amino acid section
        for i in range(last_aa_index + 1, len(headers)):
            if i in keep_columns:
                html += f'<th>{headers[i]}</th>\n'
        
        html += '</tr>\n</thead>\n<tbody>\n'
        
        # Process each data row
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            values = [cell.text.strip() for cell in cells]
            
            html += '<tr>\n'
            
            # Add values up to the last amino acid position
            for i in range(last_aa_index + 1):
                if i in keep_columns:
                    try:
                        num_value = float(values[i])
                        html += f'<td>{num_value:.2f}</td>\n'
                    except ValueError:
                        html += f'<td>{values[i]}</td>\n'
            
            # Calculate and add MET+CYS
            if met_index is not None and cys_index is not None:
                try:
                    met_value = float(values[met_index])
                    cys_value = float(values[cys_index])
                    combined = met_value + cys_value
                    html += f'<td>{combined:.2f}</td>\n'
                except (ValueError, IndexError):
                    html += '<td>N/A</td>\n'
            
            # Calculate and add PHE+TYR
            if phe_index is not None and tyr_index is not None:
                try:
                    phe_value = float(values[phe_index])
                    tyr_value = float(values[tyr_index])
                    combined = phe_value + tyr_value
                    html += f'<td>{combined:.2f}</td>\n'
                except (ValueError, IndexError):
                    html += '<td>N/A</td>\n'
            
            # Add remaining values after the amino acid section
            for i in range(last_aa_index + 1, len(values)):
                if i in keep_columns:
                    try:
                        num_value = float(values[i])
                        html += f'<td>{num_value:.2f}</td>\n'
                    except ValueError:
                        html += f'<td>{values[i]}</td>\n'
            
            html += '</tr>\n'
        
        html += '</tbody>\n</table>'
        return html
        
    except Exception as e:
        print(f"Error calculating amino acids: {str(e)}")
        return f'<p class="error">Error calculating amino acids: {str(e)}</p>'

# def calculate_amino_acids(content):
#     try:
#         # Parse the HTML content
#         soup = BeautifulSoup(content, 'html.parser')
#         table = soup.find('table')
#         if not table:
#             return '<p class="error">No table found in content</p>'

#         rows = table.find_all('tr')
#         headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
        
#         # Find indices for Met, Cys, Phe, and Tyr
#         met_index = None
#         cys_index = None
#         phe_index = None
#         tyr_index = None
#         last_aa_index = 0
        
#         # Find indices and track the last amino acid column
#         for i, header in enumerate(headers):
#             header_lower = header.lower()
#             if 'methionine' in header_lower or 'met' in header_lower:
#                 met_index = i
#                 last_aa_index = max(last_aa_index, i)
#             elif 'cysteine' in header_lower or 'cys' in header_lower:
#                 cys_index = i
#                 last_aa_index = max(last_aa_index, i)
#             elif 'phenylalanine' in header_lower or 'phe' in header_lower:
#                 phe_index = i
#                 last_aa_index = max(last_aa_index, i)
#             elif 'tyrosine' in header_lower or 'tyr' in header_lower:
#                 tyr_index = i
#                 last_aa_index = max(last_aa_index, i)

#         # Create list of columns to keep (excluding individual amino acids)
#         keep_columns = []
#         for i, header in enumerate(headers):
#             header_lower = header.lower()
#             if i not in [met_index, cys_index, phe_index, tyr_index]:
#                 keep_columns.append(i)

#         # Create new table
#         html = '<table class="table table-bordered table-striped">\n<thead>\n<tr>\n'
        
#         # Add headers up to the last amino acid position
#         for i in range(last_aa_index + 1):
#             if i in keep_columns:
#                 html += f'<th>{headers[i]}</th>\n'
        
#         # Add combined headers
#         if met_index is not None and cys_index is not None:
#             html += '<th>MET+CYS</th>\n'
#         if phe_index is not None and tyr_index is not None:
#             html += '<th>PHE+TYR</th>\n'
        
#         # Add remaining headers after the amino acid section
#         for i in range(last_aa_index + 1, len(headers)):
#             if i in keep_columns:
#                 html += f'<th>{headers[i]}</th>\n'
        
#         html += '</tr>\n</thead>\n<tbody>\n'
        
#         # Process each data row
#         for row in rows[1:]:
#             cells = row.find_all(['td', 'th'])
#             values = [cell.text.strip() for cell in cells]
            
#             html += '<tr>\n'
            
#             # Add values up to the last amino acid position
#             for i in range(last_aa_index + 1):
#                 if i in keep_columns:
#                     try:
#                         num_value = float(values[i])
#                         html += f'<td>{num_value:.2f}</td>\n'
#                     except ValueError:
#                         html += f'<td>{values[i]}</td>\n'
            
#             # Calculate and add MET+CYS
#             if met_index is not None and cys_index is not None:
#                 try:
#                     met_value = float(values[met_index])
#                     cys_value = float(values[cys_index])
#                     combined = met_value + cys_value
#                     html += f'<td>{combined:.2f}</td>\n'
#                 except (ValueError, IndexError):
#                     html += '<td>N/A</td>\n'
            
#             # Calculate and add PHE+TYR
#             if phe_index is not None and tyr_index is not None:
#                 try:
#                     phe_value = float(values[phe_index])
#                     tyr_value = float(values[tyr_index])
#                     combined = phe_value + tyr_value
#                     html += f'<td>{combined:.2f}</td>\n'
#                 except (ValueError, IndexError):
#                     html += '<td>N/A</td>\n'
            
#             # Add remaining values after the amino acid section
#             for i in range(last_aa_index + 1, len(values)):
#                 if i in keep_columns:
#                     try:
#                         num_value = float(values[i])
#                         html += f'<td>{num_value:.2f}</td>\n'
#                     except ValueError:
#                         html += f'<td>{values[i]}</td>\n'
            
#             html += '</tr>\n'
        
#         html += '</tbody>\n</table>'
#         return html
        
#     except Exception as e:
#         print(f"Error calculating amino acids: {str(e)}")
#         return f'<p class="error">Error calculating amino acids: {str(e)}</p>'

@app.route('/cleanup_old_files', methods=['POST'])
def cleanup_old_files():
    try:
        # Path to your extracted_tables folder
        folder_path = os.path.join('static', 'extracted_tables')
        
        # Check if folder exists
        if os.path.exists(folder_path):
            # Remove all files in the folder
            shutil.rmtree(folder_path)
            # Recreate the empty folder
            os.makedirs(folder_path)
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/fetch_pdcaas_values', methods=['POST'])
def fetch_pdcaas_values():
    try:
        # Example URL - replace with your actual data source
        url = "https://example.com/pdcaas_values"
        
        # Make request to the URL
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with PDCAAS values
        # Modify these selectors based on your actual HTML structure
        table = soup.find('table', {'class': 'pdcaas-table'})
        
        values = []
        if table:
            rows = table.find_all('tr')[1:]  # Skip header row
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    values.append({
                        'sample': cols[0].text.strip(),
                        'tpd': cols[1].text.strip(),
                        'ivpd': cols[2].text.strip()
                    })
        
        return jsonify({
            'success': True,
            'values': values
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # Match your port number here
