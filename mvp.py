import os
import cv2
import pydicom
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def load_dicom_images(directory):
    """Loads DICOM files from the specified directory and returns a list of image slices as numpy arrays."""
    slices = []
    # Sort the files to maintain slice order
    files = sorted([f for f in os.listdir(directory) if f.endswith('.dcm')])
    if not files:
        print("No DICOM files found in the directory.")
        return slices
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            ds = pydicom.dcmread(filepath)
            image = ds.pixel_array.astype(np.float32)
            # Normalize the image pixel values to 0-255
            norm_image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255.0
            norm_image = norm_image.astype(np.uint8)
            slices.append(norm_image)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
    return slices


def preprocess_slice(slice_image):
    """Preprocesses a single CT slice for the model: converts to 3 channels, resizes, and normalizes."""
    # Convert single-channel image to a 3-channel image
    slice_3ch = cv2.merge([slice_image, slice_image, slice_image])
    # Resize to 224x224 as expected by most ImageNet models
    slice_resized = cv2.resize(slice_3ch, (224, 224))
    # Define transformation: convert image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(slice_resized)
    return tensor


def main(scan_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load a pre-trained ResNet18 model and adjust for 2 classes (Normal vs Abnormal)
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)  # Change output to 2 classes
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Load DICOM images from the specified directory
    slices = load_dicom_images(scan_directory)
    if not slices:
        print("No valid DICOM images to process.")
        return
    
    scores_list = []
    with torch.no_grad():
        for slice_image in slices:
            tensor = preprocess_slice(slice_image)
            tensor = tensor.unsqueeze(0).to(device)  # add batch dimension
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            scores_list.append(probs.cpu().numpy()[0])

    # Aggregate the predictions by averaging the probabilities across slices
    aggregated_scores = np.mean(scores_list, axis=0)
    print("Aggregated classification result (probabilities):", aggregated_scores)
    
    # Determine the final class based on the higher probability
    pred_class = np.argmax(aggregated_scores)
    classes = ['Normal', 'Abnormal']
    print("Predicted class for CT scan:", classes[pred_class])
    # Save results to a file
    with open("results.txt", "w") as f:
        f.write("Aggregated classification result (probabilities): " + str(aggregated_scores) + "\n")
        f.write("Predicted class for CT scan: " + classes[pred_class] + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CT Scan Classification MVP')
    parser.add_argument('--scan_dir', type=str, required=True, help='Directory containing DICOM files of a CT scan')
    args = parser.parse_args()
    
    main(args.scan_dir) 