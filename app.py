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


# Initialize Flask app
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = 'extracted_tables'

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



GOOGLE_API_KEY = 'API key comes here'
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
    # Get list of images from your extracted_tables directory
    image_dir = 'extracted_tables'  # Changed from 'static/images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)  # Create directory if it doesn't exist
    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
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
            'is_dir': os.path.isdir(image_dir),
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
            # Adjust the path to match your actual image directory structure
            image_path = os.path.join('extracted_tables', image_name)
            
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
                   Please analyze this table image and extract the data with the following rules:
1. Use 'Sample' as the first column header (not 'Sample Name')
2. For entries like "Red Lentil":
   - Combine with processing methods (e.g., "Red Lentil Untreated", "Red Lentil Extruded")
   - Include the full combined name in the Sample column
3. Keep all numerical values with exact decimal places
4. Format as a markdown table with | separators
5. Ensure all columns from the original table are preserved

Example format:
| Sample | %DM | %CF | %CP | ... |
|--------|-----|-----|-----|-----|
| Casein | 93.56 | 0.20 | 86.48 | ... |
| Red Lentil Untreated | 92.12 | 1.78 | 25.13 | ... |
                    """

                    # Create generation config for Flash model
                    generation_config = {
                        'temperature': 0.1,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }

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

                    # Clean up
                    buffered.close()

            except Exception as e:
                print(f"Error processing image {image_name}: {str(e)}")
                continue

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
            'image_data': image_data
        })
    except Exception as e:
        print(f"Error getting saved images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    


# Add this route to serve images
@app.route('/extracted_tables/<filename>')
def serve_image(filename):
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
                    'path': save_path,
                    'page': page_num,
                    'y_position': y_position,
                    'confidence': float(score)
                })
    
    # Sort tables by page number and vertical position
    extracted_tables.sort(key=lambda x: (x['page'], x['y_position']))
    
    return extracted_tables

@app.route('/get_table_data/<table_id>')
def get_table_data(table_id):
    try:
        # Use the correct file path


        file_path ="C:\\Users\\Owner\\Desktop\\d5\\extractinghtml\\nutrition-labelling.html"
        
        if not os.path.exists(file_path):
            print(f"File not found at {file_path}")
            return jsonify({'error': 'Tables file not found'}), 404

        # Try different encodings
        encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    break
            except UnicodeDecodeError:
                continue
                
        if content is None:
            return jsonify({'error': 'Unable to read file with supported encodings'}), 500

        soup = BeautifulSoup(content, 'html.parser')
            
        # Find the table with the specified ID
        table = soup.find('table', {'id': table_id, 'class': 'table table-bordered table-condensed'})
        
        if not table:
            print(f"Table with id '{table_id}' not found")
            return jsonify({'error': f'Table {table_id} not found'}), 404
            
        # Extract data from the table
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 2:
                    second_td = tds[1].text.strip() if len(tds) > 1 else ""
                    third_td = tds[2].text.strip() if len(tds) > 2 else ""
                    
                    if second_td or third_td:
                        rows.append({
                            'secondColumn': second_td,
                            'thirdColumn': third_td
                        })
        
        # Get table caption if exists
        caption = table.find('caption')
        table_name = caption.text.strip() if caption else f'Table {table_id.upper()}'
        
        print(f"Found {len(rows)} rows of data")
        return jsonify({
            'tableId': table_id,
            'rows': rows,
            'tableName': table_name
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_table_captions')
def get_table_captions():
    try:
        file_path = "C:\\Users\\Owner\\Desktop\\d5\\extractinghtml\\nutrition-labelling.html"
        
        # Try different encodings
        encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    break
            except UnicodeDecodeError:
                continue

        if content is None:
            return jsonify({'error': 'Unable to read file'}), 500

        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all tables and their captions
        tables = soup.find_all('table', class_='table table-bordered table-condensed')
        captions = []
        
        for table in tables:
            table_id = table.get('id', '')
            caption = table.find('caption')
            caption_text = caption.text.strip() if caption else f'Table {table_id.upper()}'
            
            captions.append({
                'id': table_id,
                'caption': caption_text
            })
            
        return jsonify(captions)
        
    except Exception as e:
        print(f"Error getting captions: {str(e)}")
        return jsonify({'error': str(e)}), 500
    



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
        # Clear previous files
        for folder in [UPLOAD_FOLDER, EXTRACTED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error: {e}")
        
        # Save and process the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract tables
        extracted_tables = extract_tables(file_path)
        return jsonify({'tables': extracted_tables})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/extract_table_data', methods=['POST'])
def extract_table_data():
    try:
        selected_images = request.json.get('selected_images', [])
        results = []

        for image_name in selected_images:
            image_path = os.path.join('static/images', image_name)
            
            # Process image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if needed
            max_size = (1024, 1024)
            img.thumbnail(max_size)
            
            # Convert to bytes
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            
            # Gemini API prompt
            prompt = """
            Analyze this table image and extract the following information:
            1. All column headers exactly as they appear
            2. All numerical values with their exact decimal places
            3. Sample names or row identifiers
            4. Maintain empty cells where present
            
            Format the output as a markdown table with | separators.
            Preserve all numerical precision and formatting.
            Include any units or percentages exactly as shown.
            """

            # Generate content with Gemini
            generation_config = {
                'temperature': 0.1,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            }

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
            
            # Process response
            extracted_text = response.text if response.text else "No data extracted"
            formatted_html = format_table_html(extracted_text)
            
            results.append({
                'image_name': image_name,
                'extracted_data': formatted_html
            })

            # Cleanup
            img.close()
            buffered.close()

        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    
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
            'headers': headers
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

    

@app.route('/get_amino_requirements', methods=['GET'])
def get_amino_requirements():
    try:
        app.logger.info("Received request for amino requirements")
        pattern = request.args.get('pattern')
        app.logger.info(f"Requested pattern: {pattern}")
        
        # Define requirements for different patterns
        requirements = {
            'Infant (0.5 Yrs)': {
                'HIS': 20,
                'ILE': 32,
                'LEU': 66,
                'LYS': 57,
                'MET+CYS': 27,
                'PHE+TYR': 52,
                'THR': 31,
                'TRP': 8.5,
                'VAL': 43
            },
            'PreSchool Child (2-5 Yrs)': {
                'HIS': 19,
                'ILE': 28,
                'LEU': 66,
                'LYS': 58,
                'MET+CYS': 25,
                'PHE+TYR': 63,
                'THR': 34,
                'TRP': 11,
                'VAL': 35
            },
            'School Child (10-12 Yrs)': {
                'HIS': 19,
                'ILE': 28,
                'LEU': 44,
                'LYS': 44,
                'MET+CYS': 22,
                'PHE+TYR': 22,
                'THR': 28,
                'TRP': 9,
                'VAL': 25
            },
            'Adult': {
                'HIS': 16,
                'ILE': 13,
                'LEU': 19,
                'LYS': 16,
                'MET+CYS': 17,
                'PHE+TYR': 19,
                'THR': 9,
                'TRP': 5,
                'VAL': 13
            }
        }
        
        if not pattern:
            return jsonify({'error': 'No pattern selected'}), 400
            
        if pattern not in requirements:
            return jsonify({'error': f'Invalid pattern selected: {pattern}'}), 400
            
        app.logger.info(f"Returning requirements for pattern: {pattern}")
        return jsonify({
            'status': 'success',
            'data': requirements[pattern]
        })
        
    except Exception as e:
        app.logger.error(f"Error in get_amino_requirements: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
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
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # Match your port number here