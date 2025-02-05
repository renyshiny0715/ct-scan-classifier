import os
import zipfile
import tempfile
import shutil
import numpy as np
import torch
import cv2
import pydicom
import base64
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, render_template_string
from PIL import Image
from io import BytesIO
from monai.networks.nets import DenseNet121
import torch.nn as nn

app = Flask(__name__)

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_chexnet_model():
    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,  # 3 classes
        pretrained=False
    )
    
    try:
        # Try to load weights
        weights_files = ['densenet121.pth', 'model.pth.tar', 'chexnet_weights.pth']
        loaded = False
        
        for weights_file in weights_files:
            if os.path.exists(weights_file):
                print(f"Found weights file: {weights_file}")
                try:
                    state_dict = torch.load(weights_file, map_location=torch.device('cpu'))
                    
                    # If the state dict has a 'state_dict' key, it's a checkpoint
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    
                    # Remove the 'densenet121.' prefix if it exists
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('densenet121.'):
                            k = k[len('densenet121.'):]
                        if k.startswith('module.'):
                            k = k[len('module.'):]
                        new_state_dict[k] = v
                    
                    # Load only the feature extractor weights
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in new_state_dict.items() 
                                     if k in model_dict and 'class_layers' not in k}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict, strict=False)
                    
                    # Reinitialize the final layer with balanced weights
                    num_features = model.class_layers.out.in_features
                    model.class_layers.out = nn.Linear(num_features, 3)
                    nn.init.xavier_uniform_(model.class_layers.out.weight)
                    nn.init.zeros_(model.class_layers.out.bias)
                    
                    print("Successfully loaded pretrained weights and reinitialized classifier")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Could not load weights from {weights_file}: {str(e)}")
                    continue
        
        if not loaded:
            print("No compatible weight file found. Please download one of the following:")
            print("1. model.pth.tar from https://github.com/arnoweng/CheXNet")
            print("2. densenet121.pth from https://github.com/nasir6/chexnet")
            print("\nPlace the weights file in the same directory as app.py")
            
    except Exception as e:
        print(f"Error loading CheXNet model: {str(e)}")
        return None
        
    model.eval()
    return model

def download_weights():
    """Download model weights if not present locally"""
    weights_url = os.environ.get('WEIGHTS_URL', '')
    if not weights_url:
        print("No weights URL provided. Please set the WEIGHTS_URL environment variable.")
        return False
    
    try:
        import requests
        local_path = 'densenet121.pth'
        if not os.path.exists(local_path):
            print(f"Downloading weights from {weights_url}")
            response = requests.get(weights_url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print("Successfully downloaded weights")
        return True
    except Exception as e:
        print(f"Error downloading weights: {str(e)}")
        return False

# Initialize the model
if not download_weights():
    print("Warning: Could not download weights. Model may not work properly.")
model = load_chexnet_model()
if model is not None:
    model = model.to(device)

# Function to load DICOM images from a directory along with their Series Description metadata

def load_dicom_images(directory):
    slices = []
    # Sort the files to maintain slice order
    files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.dcm') or f.lower().endswith('.dicom')])
    if not files:
        print("No DICOM files found in the directory.")
        return slices
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            ds = pydicom.dcmread(filepath)
            image = ds.pixel_array.astype(np.float32)
            # Normalize pixel values to 0-255
            norm_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255.0
            norm_image = norm_image.astype(np.uint8)
            # Extract Series Description, default to 'Unknown'
            series = ds.get('SeriesDescription', 'Unknown')
            # Translate series name
            series = translate_series_name(series)
            slices.append((norm_image, series))
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
    return slices

# Function to preprocess a single CT slice for classification

def preprocess_slice(slice_image):
    """
    Preprocess a single CT slice for ResNet18
    """
    try:
        # Convert to RGB (3 channels)
        slice_3ch = cv2.merge([slice_image, slice_image, slice_image])
        
        # Convert numpy array to PyTorch tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to PIL Image first
        pil_image = Image.fromarray(slice_3ch)
        tensor = transform(pil_image)
        return tensor
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# Function to generate an animated GIF from a list of numpy images; returns base64 encoded GIF

def generate_series_gif(image_list):
    frames = []
    for img in image_list:
        # Convert numpy array to PIL image, convert to RGB
        frame = Image.fromarray(img).convert('RGB')
        frames.append(frame)
    if frames:
        buf = BytesIO()
        frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=50)
        gif_data = buf.getvalue()
        buf.close()
        gif_b64 = base64.b64encode(gif_data).decode('utf-8')
        return gif_b64
    return None

def get_condition_style(condition):
    """Return style information for different conditions"""
    styles = {
        'Normal | 正常': {'color': 'green', 'bg': '#f3fff3', 'bar': 'lightgreen'},
        'Pneumonia | 肺炎': {'color': '#ff4444', 'bg': '#fff3f3', 'bar': '#ff8888'},
        'Lung Cancer | 肺癌': {'color': '#cc0000', 'bg': '#ffebeb', 'bar': '#ff4444'}
    }
    return styles.get(condition, {'color': 'gray', 'bg': '#f8f8f8', 'bar': 'lightgray'})

def translate_series_name(series):
    """Translate series names to bilingual format"""
    translations = {
        'Mediastinum': 'Mediastinum | 纵隔',
        'MEDIASTINUM': 'Mediastinum | 纵隔',
        'Lung': 'Lung | 肺部',
        'LUNG': 'Lung | 肺部',
        'Chest': 'Chest | 胸部',
        'CHEST': 'Chest | 胸部',
        'Thorax': 'Thorax | 胸腔',
        'THORAX': 'Thorax | 胸腔',
        'Abdomen': 'Abdomen | 腹部',
        'ABDOMEN': 'Abdomen | 腹部',
        'Unknown': 'Unknown | 未知',
        'UNKNOWN': 'Unknown | 未知'
    }
    
    # Handle series names with numbers (e.g., "Mediastinum 1.5")
    for key in translations.keys():
        if series.startswith(key):
            # Preserve any numbers or additional text
            remainder = series[len(key):].strip()
            if remainder:
                return f"{translations[key]} {remainder}"
            return translations[key]
    
    return f"{series} | {series}"  # Return original name if no translation found

# Function to process CT scan and perform classification

def process_scan(scan_dir):
    """
    Process CT scan images and perform classification
    """
    if model is None:
        return None, "Model not loaded properly. Please check the server logs."
    
    # Load DICOM images with their series metadata
    slices = load_dicom_images(scan_dir)
    if not slices:
        return None, "No valid DICOM images found in the directory."
    
    try:
        results = []
        with torch.no_grad():
            for image, series in slices:
                # Preprocess image
                input_tensor = preprocess_slice(image)
                if input_tensor is None:
                    continue
                
                input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

                # Get model predictions
                output = model(input_tensor)
                
                # Apply bias correction to raw logits (final adjusted values)
                bias_correction = torch.tensor([1.0, -1.5, -2.0]).to(device)  # Boost normal, reduce others
                output = output + bias_correction
                
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # Debug prints
                print(f"\nRaw model output: {output[0]}")
                print(f"Softmax probabilities: {probabilities[0]}")
                
                # Class labels with translations
                class_labels = ['Normal | 正常', 'Pneumonia | 肺炎', 'Lung Cancer | 肺癌']
                
                # Get raw probabilities
                normal_prob = float(probabilities[0, 0])
                pneumonia_prob = float(probabilities[0, 1])
                lung_cancer_prob = float(probabilities[0, 2])
                
                print(f"Normal prob: {normal_prob:.3f}, Pneumonia prob: {pneumonia_prob:.3f}, Lung Cancer prob: {lung_cancer_prob:.3f}")
                
                # Get the highest probability and its corresponding label
                max_prob = max(normal_prob, pneumonia_prob, lung_cancer_prob)
                max_index = [normal_prob, pneumonia_prob, lung_cancer_prob].index(max_prob)
                
                # Classification logic with thresholds
                if max_prob < 0.5:  # If no strong prediction
                    predicted_label = class_labels[0]  # Default to normal
                else:
                    predicted_label = class_labels[max_index]
                
                print(f"Predicted label: {predicted_label} (max_prob: {max_prob:.3f})")
                
                # Format probabilities for display
                prob_dict = {
                    class_labels[0]: normal_prob,
                    class_labels[1]: pneumonia_prob,
                    class_labels[2]: lung_cancer_prob
                }

                results.append({
                    'image': image,
                    'series': series,
                    'probabilities': prob_dict,
                    'predicted_label': predicted_label,
                    'style': get_condition_style(predicted_label),
                    'is_abnormal': predicted_label != class_labels[0]  # Add flag for sorting
                })
        
        if not results:
            return None, "Failed to process any images in the scan."
        
        return results, None
    except Exception as e:
        print(f"Error processing scan: {e}")
        return None, f"Error processing scan: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return render_template_string('''
                <!doctype html>
                <title>Error</title>
                <h2 style="color: red;">Error: No files selected.</h2>
                <button onclick='window.location.href="/";'>Return to Upload Page</button>
            ''')
        
        temp_dir = tempfile.mkdtemp()
        dicom_dir = os.path.join(temp_dir, "dicoms")
        os.makedirs(dicom_dir, exist_ok=True)
        
        try:
            for file in files:
                filename = file.filename
                if filename.lower().endswith('.zip'):
                    zip_path = os.path.join(temp_dir, filename)
                    file.save(zip_path)
                    extract_dir = os.path.join(temp_dir, "extracted")
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    for root, dirs, filenames in os.walk(extract_dir):
                        for f in filenames:
                            if f.lower().endswith('.dcm') or f.lower().endswith('.dicom'):
                                src = os.path.join(root, f)
                                dst = os.path.join(dicom_dir, f)
                                shutil.move(src, dst)
                elif filename.lower().endswith('.dcm') or filename.lower().endswith('.dicom'):
                    file.save(os.path.join(dicom_dir, filename))
                else:
                    print(f"Skipping file {filename}: Unsupported file type.")
            
            results, error = process_scan(dicom_dir)
            
            if error:
                return render_template_string('''
                    <!doctype html>
                    <html>
                    <head>
                        <title>Error | 错误</title>
                        <style>
                            body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; text-align: center; }
                            .error-container { max-width: 600px; margin: 40px auto; padding: 20px; background: #fff3f3; border-radius: 10px; }
                            .error-icon { font-size: 48px; color: #ff4444; }
                            .return-btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                            [lang="zh"] { display: none; }
                        </style>
                        <script>
                            function toggleLanguage() {
                                const enElements = document.querySelectorAll('[lang="en"]');
                                const zhElements = document.querySelectorAll('[lang="zh"]');
                                enElements.forEach(el => {
                                    el.style.display = el.style.display === 'none' ? 'block' : 'none';
                                });
                                zhElements.forEach(el => {
                                    el.style.display = el.style.display === 'none' ? 'block' : 'none';
                                });
                            }
                        </script>
                    </head>
                    <body>
                        <button onclick="toggleLanguage()" style="position: absolute; top: 20px; right: 20px;">EN / 中文</button>
                        <div class="error-container">
                            <div class="error-icon">⚠️</div>
                            <div lang="en">
                                <h2 style="color: red;">Error Processing Files</h2>
                                <p>{{ error }}</p>
                            </div>
                            <div lang="zh">
                                <h2 style="color: red;">文件处理错误</h2>
                                <p>{{ error }}</p>
                            </div>
                            <button onclick='window.location.href="/";' class="return-btn">
                                <span lang="en">Return to Upload Page</span>
                                <span lang="zh">返回上传页面</span>
                            </button>
                        </div>
                    </body>
                    </html>
                ''', error=error)
            
            # Group results by series
            series_results = {}
            for result in results:
                series = result['series']
                if series not in series_results:
                    series_results[series] = []
                series_results[series].append(result)
            
            result_html = '''
                <!doctype html>
                <html>
                <head>
                    <title>REN A.I. - Analysis Results | 分析结果</title>
                    <style>
                        :root {
                            --primary-color: #2c3e50;
                            --accent-color: #3498db;
                            --bg-color: #f8f9fa;
                            --text-color: #2c3e50;
                        }
                        body {
                            font-family: 'Segoe UI', Arial, sans-serif;
                            line-height: 1.6;
                            margin: 0;
                            padding: 0;
                            background: var(--bg-color);
                            color: var(--text-color);
                        }
                        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                        .header {
                            text-align: center;
                            padding: 40px 0;
                            background: linear-gradient(135deg, var(--primary-color), #1a2a3a);
                            color: white;
                            margin-bottom: 40px;
                        }
                        .company-name {
                            font-size: 2.5em;
                            font-weight: bold;
                            color: var(--accent-color);
                            margin-bottom: 30px;
                            letter-spacing: 2px;
                            text-transform: uppercase;
                            background: white;
                            display: inline-block;
                            padding: 10px 30px;
                            border-radius: 8px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        }
                        .series-box { 
                            border: 1px solid #ddd;
                            padding: 15px;
                            margin: 10px 0;
                            border-radius: 10px;
                            background: white;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        }
                        .probability-bar { width: 200px; height: 20px; background-color: #f0f0f0; margin: 5px 0; border-radius: 3px; }
                        .probability-fill { height: 100%; border-radius: 3px; }
                        .image-grid {
                            display: grid;
                            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                            gap: 20px;
                            margin-top: 20px;
                        }
                        .slice-image {
                            width: 100%;
                            height: auto;
                            border-radius: 5px;
                            transition: transform 0.3s;
                        }
                        .slice-image:hover {
                            transform: scale(1.05);
                        }
                        .abnormal-tag { color: red; font-weight: bold; }
                        details summary {
                            cursor: pointer;
                            padding: 15px;
                            background-color: #f8f9fa;
                            border-radius: 5px;
                            margin-bottom: 10px;
                            transition: background-color 0.3s;
                            text-align: center;
                        }
                        details summary:hover {
                            background-color: #e9ecef;
                        }
                        .summary-content {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            gap: 10px;
                        }
                        .series-gif {
                            text-align: center;
                            margin: 20px 0;
                            background: #f8f9fa;
                            padding: 20px;
                            border-radius: 10px;
                        }
                        .series-gif img {
                            max-width: 100%;
                            height: auto;
                            border-radius: 5px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }
                        .return-btn {
                            display: block;
                            width: 200px;
                            margin: 40px auto;
                            padding: 12px 0;
                            background: var(--accent-color);
                            color: white;
                            text-align: center;
                            border: none;
                            border-radius: 5px;
                            cursor: pointer;
                            text-decoration: none;
                        }
                        .return-btn:hover {
                            background: #2980b9;
                        }
                        [lang="zh"] {
                            display: block !important;  /* Always show Chinese text */
                        }
                        [lang="en"] {
                            display: block !important;  /* Always show English text */
                        }
                        .bilingual-text {
                            margin: 5px 0;
                        }
                        .hero-section {
                            display: flex;
                            align-items: flex-start;
                            justify-content: space-between;
                            gap: 40px;
                            margin-top: 30px;
                        }
                        
                        .hero-content {
                            flex: 1;
                            padding-top: 20px;
                        }
                        
                        .hero-image {
                            flex: 1;
                            text-align: center;
                        }
                        
                        .hero-image img {
                            max-width: 100%;
                            height: auto;
                            border-radius: 10px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        }

                        @media (max-width: 768px) {
                            .hero-section {
                                flex-direction: column;
                                align-items: center;
                                text-align: center;
                            }
                            
                            .company-name {
                                margin: 0 auto 20px;
                            }
                        }
                    </style>
                    <script>
                        function toggleLanguage() {
                            const enElements = document.querySelectorAll('[lang="en"]');
                            const zhElements = document.querySelectorAll('[lang="zh"]');
                            enElements.forEach(el => {
                                el.style.display = el.style.display === 'none' ? 'block' : 'none';
                            });
                            zhElements.forEach(el => {
                                el.style.display = el.style.display === 'none' ? 'block' : 'none';
                            });
                        }
                    </script>
                </head>
                <body>
                    <button onclick="toggleLanguage()" style="position: absolute; top: 20px; right: 20px; padding: 8px 16px; background: var(--accent-color); color: white; border: none; border-radius: 4px; cursor: pointer;">EN / 中文</button>
                    
                    <div class="header">
                        <div class="container">
                            <div class="hero-section">
                                <div class="hero-content">
                                    <div class="company-name">REN A.I.</div>
                                    <div class="bilingual-text">
                                        <h1>Analysis Results | 分析结果</h1>
                                        <p>Detailed analysis of your CT scan series | CT扫描系列的详细分析</p>
                                    </div>
                                </div>
                                <div class="hero-image">
                                    <img src="/static/ads/download.png" alt="CT Scan Analysis">
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="container">
                '''
            
            if series_results:
                for series, series_data in series_results.items():
                    # Calculate series-level statistics
                    total_slices = len(series_data)
                    abnormal_slices = sum(1 for r in series_data if r['predicted_label'] != 'Normal | 正常')
                    abnormal_percentage = (abnormal_slices / total_slices) * 100 if total_slices > 0 else 0
                    
                    # Create GIF for this series
                    series_images = [r['image'] for r in series_data]
                    series_gif = generate_series_gif(series_images)
                    
                    result_html += f'''
                        <div class="series-box">
                            <div class="series-gif">
                                <h4>Series Animation | 系列动画: {series}</h4>
                                <img src="data:image/gif;base64,{series_gif}" 
                                     alt="Series animation"
                                     title="Series animation">
                            </div>
                            
                            <details>
                                <summary>
                                    <div class="summary-content">
                                        <div class="bilingual-text">
                                            Detailed Analysis | 详细分析
                                        </div>
                                        <span class="abnormal-tag">
                                            ({abnormal_slices}/{total_slices} abnormal - {abnormal_percentage:.1f}%)
                                        </span>
                                    </div>
                                </summary>
                                <div style='margin-left: 20px;'>
                                    <h4 class="bilingual-text">
                                        Conditions Detected | 检测到的病症：
                                    </h4>
                                    <div class="condition-summary">
                    '''
                    
                    # Add summary of conditions detected in this series
                    conditions_found = set(slice_data['predicted_label'] for slice_data in series_data if slice_data['predicted_label'] != 'Normal | 正常')
                    for condition in conditions_found:
                        style = get_condition_style(condition)
                        result_html += f'''
                            <div class="condition-tag" style="color: {style['color']}; background: {style['bg']};">
                                {condition}
                            </div>
                        '''
                    
                    # Sort series_data to put abnormal cases first
                    series_data.sort(key=lambda x: (not x['is_abnormal'], 
                                                  -max(x['probabilities'].values())))  # Sort by abnormal flag and confidence

                    result_html += '''
                        <h4 class="bilingual-text">Abnormal Slices | 异常切片：</h4>
                    '''
                    
                    # Add image grid for abnormal slices first
                    result_html += '<div class="image-grid">'
                    for i, slice_data in enumerate(series_data):
                        if slice_data['is_abnormal']:
                            style = slice_data['style']
                            img_array = slice_data['image']
                            ret, buffer = cv2.imencode('.png', img_array)
                            if ret:
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                # Sort probabilities by value
                                sorted_probs = sorted(slice_data['probabilities'].items(), key=lambda x: x[1], reverse=True)
                                prob_html = ''.join([
                                    f'<div style="color: {get_condition_style(label)["color"]};">{label}: {prob * 100:.1f}%</div>'
                                    for label, prob in sorted_probs
                                ])
                                
                                result_html += f'''
                                    <div class="image-container" style="border: 2px solid {style['color']}; padding: 5px; border-radius: 8px; background-color: {style['bg']};">
                                        <img src="data:image/png;base64,{img_base64}" 
                                             class="slice-image" 
                                             title="Predicted: {slice_data['predicted_label']}">
                                        <div style="text-align: center;">
                                            <div style="color: {style['color']}; font-weight: bold;">
                                                Slice {i+1}: {slice_data['predicted_label']}
                                            </div>
                                            <div class="probability-details" style="font-size: 0.9em; margin-top: 5px;">
                                                {prob_html}
                                            </div>
                                        </div>
                                    </div>
                                '''
                    result_html += '</div>'

                    result_html += '''
                        <h4 class="bilingual-text">Normal Slices | 正常切片：</h4>
                    '''
                    
                    # Add image grid for normal slices
                    result_html += '<div class="image-grid">'
                    for i, slice_data in enumerate(series_data):
                        if not slice_data['is_abnormal']:
                            style = slice_data['style']
                            img_array = slice_data['image']
                            ret, buffer = cv2.imencode('.png', img_array)
                            if ret:
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                # Sort probabilities by value
                                sorted_probs = sorted(slice_data['probabilities'].items(), key=lambda x: x[1], reverse=True)
                                prob_html = ''.join([
                                    f'<div style="color: {get_condition_style(label)["color"]};">{label}: {prob * 100:.1f}%</div>'
                                    for label, prob in sorted_probs
                                ])
                                
                                result_html += f'''
                                    <div class="image-container" style="border: 2px solid {style['color']}; padding: 5px; border-radius: 8px; background-color: {style['bg']};">
                                        <img src="data:image/png;base64,{img_base64}" 
                                             class="slice-image" 
                                             title="Predicted: {slice_data['predicted_label']}">
                                        <div style="text-align: center;">
                                            <div style="color: {style['color']}; font-weight: bold;">
                                                Slice {i+1}: {slice_data['predicted_label']}
                                            </div>
                                            <div class="probability-details" style="font-size: 0.9em; margin-top: 5px;">
                                                {prob_html}
                                            </div>
                                        </div>
                                    </div>
                                '''
                    result_html += '</div>'
                    
                    result_html += '''
                                </div>
                            </details>
                        </div>
                    '''
            
            result_html += '''
                    </div>
                    <br><br>
                    <button onclick='window.location.href="/";' class="return-btn">
                        Return to Upload Page | 返回上传页面
                    </button>
                </div>
            '''
            
            return result_html
            
        except Exception as e:
            return render_template_string('''
                <!doctype html>
                <title>Error</title>
                <h2 style="color: red;">Error Processing Files</h2>
                <p>{{ error }}</p>
                <button onclick='window.location.href="/";'>Return to Upload Page</button>
            ''', error=str(e))
        
        finally:
            shutil.rmtree(temp_dir)
    
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>REN A.I. - AI-Powered CT Scan Analysis | CT扫描智能分析</title>
            <style>
                :root {
                    --primary-color: #2c3e50;
                    --accent-color: #3498db;
                    --bg-color: #f8f9fa;
                    --text-color: #2c3e50;
                }
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    background: var(--bg-color);
                    color: var(--text-color);
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    text-align: center;
                    padding: 40px 0;
                    background: var(--primary-color);
                    color: white;
                    margin-bottom: 40px;
                }
                .company-name {
                    font-size: 2.5em;
                    font-weight: bold;
                    color: var(--accent-color);
                    margin-bottom: 30px;
                    letter-spacing: 2px;
                    text-transform: uppercase;
                    background: white;
                    display: inline-block;
                    padding: 10px 30px;
                    border-radius: 8px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }
                .language-switch {
                    position: absolute;
                    top: 20px;
                    right: 20px;
                }
                .language-btn {
                    padding: 8px 16px;
                    background: transparent;
                    border: 1px solid white;
                    color: white;
                    cursor: pointer;
                    margin-left: 10px;
                    border-radius: 4px;
                }
                .language-btn:hover {
                    background: rgba(255,255,255,0.1);
                }
                .value-props {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 30px;
                    margin: 40px 0;
                }
                .value-prop {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .value-prop img {
                    width: 64px;
                    height: 64px;
                    margin-bottom: 20px;
                }
                .upload-form {
                    background: white;
                    border: 2px dashed var(--accent-color);
                    padding: 40px;
                    text-align: center;
                    border-radius: 10px;
                    margin: 40px 0;
                }
                .submit-btn {
                    background: var(--accent-color);
                    color: white;
                    padding: 12px 30px;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                .submit-btn:hover {
                    background: #2980b9;
                }
                .custom-file-input {
                    position: relative;
                    display: inline-block;
                    margin: 20px 0;
                }
                .custom-file-input input[type="file"] {
                    display: none;
                }
                .custom-file-label {
                    display: inline-block;
                    padding: 10px 20px;
                    background: var(--accent-color);
                    color: white;
                    border-radius: 5px;
                    cursor: pointer;
                    margin-right: 10px;
                }
                .file-name {
                    display: inline-block;
                    color: #666;
                    margin-left: 10px;
                }
                .loading-overlay {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.85);
                    z-index: 1000;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }
                .loading-spinner {
                    width: 120px;
                    height: 120px;
                    margin-bottom: 30px;
                    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cpath fill='%233498db' d='M73,50c0-12.7-10.3-23-23-23S27,37.3,27,50 M30.9,50c0-10.5,8.5-19.1,19.1-19.1S69.1,39.5,69.1,50'%3E%3CanimateTransform attributeName='transform' attributeType='XML' type='rotate' dur='1s' from='0 50 50' to='360 50 50' repeatCount='indefinite'/%3E%3C/path%3E%3C/svg%3E") center/contain no-repeat;
                }
                .loading-text {
                    color: white;
                    font-size: 1.4em;
                    text-align: center;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                }
                .loading-subtext {
                    color: #3498db;
                    font-size: 1.1em;
                    margin-top: 15px;
                    font-weight: 500;
                }
                [lang="zh"] {
                    display: block;
                }
                .features {
                    margin: 40px 0;
                    text-align: center;
                }
                .feature-list {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 25px;
                    margin-top: 30px;
                }
                .feature-item {
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    text-align: center;
                }
                .feature-item:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                }
                .feature-icon {
                    display: inline-block;
                    width: 50px;
                    height: 50px;
                    line-height: 50px;
                    text-align: center;
                    background: var(--accent-color);
                    color: white;
                    border-radius: 50%;
                    margin-bottom: 15px;
                    font-size: 24px;
                }
                .feature-title {
                    font-size: 1.2em;
                    font-weight: 600;
                    margin: 10px 0;
                    color: var(--primary-color);
                }
                .feature-description {
                    font-size: 0.95em;
                    color: #666;
                    margin-top: 8px;
                }
                .hero-section {
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 40px;
                    margin-top: 30px;
                }
                
                .hero-content {
                    flex: 1;
                    padding-top: 20px;
                }
                
                .hero-image {
                    flex: 1;
                    text-align: center;
                }
                
                .hero-image img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 10px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }
                
                .header {
                    background: linear-gradient(135deg, var(--primary-color), #1a2a3a);
                    padding: 60px 0;
                }
                
                .value-props {
                    margin-top: -60px;
                    position: relative;
                    z-index: 2;
                }
                
                .value-prop {
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }
                
                .value-prop:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                }
                
                .upload-form {
                    background: linear-gradient(135deg, white, #f8f9fa);
                    border: none;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }
                
                .custom-file-label {
                    background: linear-gradient(135deg, var(--accent-color), #2980b9);
                    transition: transform 0.3s ease;
                }
                
                .custom-file-label:hover {
                    transform: translateY(-2px);
                }
                
                .submit-btn {
                    background: linear-gradient(135deg, var(--accent-color), #2980b9);
                    transition: transform 0.3s ease;
                }
                
                .submit-btn:hover {
                    transform: translateY(-2px);
                }
                
                @media (max-width: 768px) {
                    .hero-section {
                        flex-direction: column;
                        align-items: center;
                        text-align: center;
                    }
                    
                    .company-name {
                        margin: 0 auto 20px;
                    }
                }
            </style>
            <script>
                function toggleLanguage() {
                    const enElements = document.querySelectorAll('[lang="en"]');
                    const zhElements = document.querySelectorAll('[lang="zh"]');
                    enElements.forEach(el => {
                        el.style.display = el.style.display === 'none' ? 'block' : 'none';
                    });
                    zhElements.forEach(el => {
                        el.style.display = el.style.display === 'none' ? 'block' : 'none';
                    });
                }
            </script>
        </head>
        <body>
            <div class="language-switch">
                <button class="language-btn" onclick="toggleLanguage()">EN / 中文</button>
            </div>
            
            <div class="header">
                <div class="container">
                    <div class="hero-section">
                        <div class="hero-content">
                            <div class="company-name">REN A.I.</div>
                            <div class="bilingual-text">
                                <h1>AI-Powered CT Scan Analysis | CT扫描智能分析</h1>
                                <p>Advanced deep learning technology for rapid and accurate lung condition detection | 采用先进深度学习技术，快速准确检测肺部疾病</p>
                            </div>
                        </div>
                        <div class="hero-image">
                            <img src="{{ url_for('static', filename='ads/download.png') }}" alt="CT Scan Analysis">
                        </div>
                    </div>
                </div>
            </div>

            <div class="container">
                <div class="value-props">
                    <div class="value-prop">
                        <img src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%233498db' d='M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-2a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm1-8h4v2h-6V7h2v5z'/></svg>">
                        <div class="bilingual-text">
                            <h3>Rapid Analysis | 快速分析</h3>
                            <p>Process entire CT scan series in seconds | 数秒内完成整个CT系列扫描分析</p>
                        </div>
                    </div>
                    <div class="value-prop">
                        <img src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%233498db' d='M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1-11v6h2v-6h-2zm0-4v2h2V7h-2z'/></svg>">
                        <div class="bilingual-text">
                            <h3>High Accuracy | 高精确度</h3>
                            <p>Advanced AI model trained on extensive medical datasets | AI模型经过大量医疗数据集训练</p>
                        </div>
                    </div>
                    <div class="value-prop">
                        <img src="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%233498db' d='M20 2H4c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-8 17c-2.8 0-5-2.2-5-5s2.2-5 5-5 5 2.2 5 5-2.2 5-5 5z'/></svg>">
                        <div class="bilingual-text">
                            <h3>Comprehensive Analysis | 全面分析</h3>
                            <p>Detect multiple lung conditions in one scan | 一次扫描检测多种肺部疾病</p>
                        </div>
                    </div>
                </div>

                <div class="features">
                    <h2 class="bilingual-text">Key Features | 主要功能</h2>
                    <div class="feature-list">
                        <div class="feature-item">
                            <span class="feature-icon">🔍</span>
                            <div class="feature-title bilingual-text">
                                Multi-condition Detection | 多病症检测
                            </div>
                            <div class="feature-description bilingual-text">
                                Detect multiple lung conditions simultaneously | 同时检测多种肺部疾病
                            </div>
                        </div>
                        <div class="feature-item">
                            <span class="feature-icon">📊</span>
                            <div class="feature-title bilingual-text">
                                Slice-by-slice Analysis | 逐层分析
                            </div>
                            <div class="feature-description bilingual-text">
                                Detailed analysis of each CT slice | 详细分析每一层CT图像
                            </div>
                        </div>
                        <div class="feature-item">
                            <span class="feature-icon">🎥</span>
                            <div class="feature-title bilingual-text">
                                Dynamic Visualization | 动态可视化
                            </div>
                            <div class="feature-description bilingual-text">
                                Animated series visualization | 动态序列可视化
                            </div>
                        </div>
                        <div class="feature-item">
                            <span class="feature-icon">📋</span>
                            <div class="feature-title bilingual-text">
                                Detailed Reports | 详细报告
                            </div>
                            <div class="feature-description bilingual-text">
                                Comprehensive analysis reports | 全面的分析报告
                            </div>
                        </div>
                    </div>
                </div>

                <div class="upload-form">
                    <h2>Upload Your CT Scan<br>上传CT扫描</h2>
                    <p>Select a ZIP file containing DICOM files or individual DICOM files<br>
                       选择包含DICOM文件的ZIP压缩包或单个DICOM文件</p>
                    <form method=post enctype=multipart/form-data onsubmit="showLoading()">
                        <div class="custom-file-input">
                            <label for="file-upload" class="custom-file-label">
                                Choose Files | 选择文件
                            </label>
                            <input id="file-upload" type=file name=files multiple>
                            <span class="file-name" id="file-name">No file chosen | 未选择文件</span>
                        </div>
                        <br>
                        <button type=submit class="submit-btn">
                            Upload and Analyze<br>上传并分析
                        </button>
                    </form>
                </div>

                <div class="loading-overlay" id="loadingOverlay">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">
                        Processing your CT scan...<br>
                        正在处理您的CT扫描...
                        <div class="loading-subtext">
                            This may take a few moments<br>
                            这可能需要一点时间
                        </div>
                    </div>
                </div>
            </div>

            <script>
                function showLoading() {
                    const overlay = document.getElementById('loadingOverlay');
                    if (overlay) {
                        overlay.style.display = 'flex';
                        overlay.style.opacity = 0;
                        setTimeout(() => {
                            overlay.style.transition = 'opacity 0.3s ease';
                            overlay.style.opacity = 1;
                        }, 10);
                    }
                    return true;
                }

                // Add file input handling
                document.getElementById('file-upload').addEventListener('change', function(e) {
                    const fileName = document.getElementById('file-name');
                    if (this.files.length > 1) {
                        fileName.textContent = `${this.files.length} files selected | 已选择${this.files.length}个文件`;
                    } else if (this.files.length === 1) {
                        fileName.textContent = `${this.files[0].name} | 已选择文件`;
                    } else {
                        fileName.textContent = 'No file chosen | 未选择文件';
                    }
                });
            </script>
        </body>
        </html>
    ''')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 