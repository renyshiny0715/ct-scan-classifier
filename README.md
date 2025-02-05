# CT Scan Classifier

An AI-powered web application for analyzing and classifying CT scan images, with a focus on detecting lung conditions including COVID-19, pneumonia, and lung cancer.

## Features

- 🔍 Multi-condition Detection: Simultaneously detect multiple lung conditions
- 📊 Slice-by-slice Analysis: Detailed analysis of each CT slice
- 🎥 Dynamic Visualization: Animated series visualization of CT scans
- 📋 Comprehensive Reports: Detailed analysis reports with confidence scores
- 🌐 Bilingual Support: English and Chinese interface

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/renyshiny0715/ct-scan-classifier.git
cd ct-scan-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

Due to file size limitations, CT scan data is not included in the repository. You can:

1. Use your own DICOM files:
   - Place your .dcm files in a directory
   - Upload them through the web interface
   - Supported formats: Individual DICOM files or ZIP archives containing DICOM files

2. Download sample data:
   - Sample COVID-19 CT scans can be obtained from:
     - [TCIA Collections](https://www.cancerimagingarchive.net/)
     - [COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/record/3757476)
   - Place the downloaded DICOM files in your local project directory

3. Generate sample data:
   - Use the provided script to generate sample DICOM files:
```bash
python generate_sample_dicom.py
```

## Project Structure

```
ct-scan-classifier/
├── app.py              # Main Flask application
├── mvp.py             # Core classification logic
├── test_app.py        # Unit tests
├── requirements.txt   # Python dependencies
├── Dockerfile        # Container configuration
├── static/           # Static assets
└── templates/        # HTML templates
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload CT scan files through the web interface:
   - Select individual DICOM files or a ZIP archive
   - Click "Upload and Analyze"
   - View the analysis results

## Model Information

The application uses a modified DenseNet121 architecture pre-trained on chest X-ray datasets and fine-tuned for CT scan analysis. The model can detect:
- Normal conditions
- Pneumonia
- Lung Cancer

## Development

To run tests:
```bash
python -m unittest test_app.py
```

## Docker Support

Build and run with Docker:
```bash
docker build -t ct-scan-classifier .
docker run -p 5000:5000 ct-scan-classifier
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MONAI](https://monai.io/) for medical imaging tools
- [PyTorch](https://pytorch.org/) for deep learning framework
- [Flask](https://flask.palletsprojects.com/) for web framework 