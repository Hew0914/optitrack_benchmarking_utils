import cv2
import os
import glob
import re

def list_camera_files(dev_folder):
    # Use glob to find all image files in the dev folder
    image_files = glob.glob(os.path.join(dev_folder, '/video*'))
    return image_files

def extract_camera_indices(filenames):
    # Extract numbers from filenames using regex
    indices = []
    for filename in filenames:
        indices.append(int(re.sub('[^0-9]', '', filename)))
    return indices

def capture_images(cameras):
    output_dir = "camera_captures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cam_index in cameras:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        ret, frame = cap.read()
        if ret:
            filename = f"{output_dir}/camera_{cam_index}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Captured image from camera {cam_index} and saved to {filename}")
        else:
            print(f"Failed to capture image from camera {cam_index}")
        cap.release()

def main():
    dev_folder = '/dev'  # Change this to the folder where your camera image files are stored
    camera_files = list_camera_files(dev_folder)
    if camera_files:
        print(f"Found camera files: {camera_files}")
        camera_indices = extract_camera_indices(camera_files)
        print(f"Extracted camera indices: {camera_indices}")
        capture_images(camera_indices)
    else:
        print("No camera files found")

if __name__ == "__main__":
    main()
