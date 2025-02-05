import unittest
import os
import numpy as np
import torch
from PIL import Image
import pydicom
from app import (
    load_dicom_images,
    preprocess_slice,
    generate_series_gif,
    process_scan,
    app
)

class TestCTScanApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test directory
        cls.test_dir = "test_data"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create a sample DICOM file for testing
        cls.sample_image = np.random.randint(0, 255, (512, 512), dtype=np.uint16)
        cls.dicom_path = os.path.join(cls.test_dir, "test.dcm")
        
        # Create a simple DICOM file
        ds = pydicom.Dataset()
        ds.SeriesDescription = "Test Series"
        ds.Rows = 512
        ds.Columns = 512
        ds.PixelData = cls.sample_image.tobytes()
        ds.save_as(cls.dicom_path)

    def test_load_dicom_images(self):
        """Test DICOM image loading"""
        slices = load_dicom_images(self.test_dir)
        self.assertTrue(len(slices) > 0)
        image, series = slices[0]
        self.assertEqual(series, "Test Series")
        self.assertTrue(isinstance(image, np.ndarray))

    def test_preprocess_slice(self):
        """Test image preprocessing"""
        # Create a test image
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        tensor = preprocess_slice(test_image)
        self.assertIsNotNone(tensor)
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertEqual(tensor.shape, (3, 224, 224))

    def test_generate_series_gif(self):
        """Test GIF generation"""
        test_images = [np.random.randint(0, 255, (100, 100), dtype=np.uint8) for _ in range(3)]
        gif_data = generate_series_gif(test_images)
        self.assertIsNotNone(gif_data)
        self.assertTrue(isinstance(gif_data, str))

    def test_process_scan(self):
        """Test scan processing"""
        results, error = process_scan(self.test_dir)
        if error:
            print(f"Process scan error: {error}")
        else:
            self.assertIsNotNone(results)
            self.assertTrue(len(results) > 0)
            self.assertTrue('predicted_label' in results[0])
            self.assertTrue('probabilities' in results[0])

    def test_flask_app(self):
        """Test Flask application routes"""
        with app.test_client() as client:
            # Test GET request
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            
            # Test POST request without files
            response = client.post('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Error', response.data)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 