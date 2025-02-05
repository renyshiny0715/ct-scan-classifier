import pydicom
from pydicom.dataset import FileDataset
import datetime
import numpy as np


def create_sample_dicom(filename, pixel_array):
    # Create file meta information
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    # Create the FileDataset instance (initially no data elements, but file_meta supplied)
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S')

    # Set necessary values for this file
    ds.Modality = 'OT'  # Other
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelData = pixel_array.tobytes()

    # Save the file
    ds.save_as(filename)
    print(f"Sample DICOM file '{filename}' created.")


if __name__ == '__main__':
    # Create a sample image of size 512x512 with random pixel values in a typical CT range
    sample_image = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
    create_sample_dicom("example.dcm", sample_image) 