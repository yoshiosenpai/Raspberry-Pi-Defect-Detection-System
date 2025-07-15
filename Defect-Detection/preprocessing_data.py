import os
import cv2
import numpy as np

def load_data(folder_path, label):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath)
        if image is not None:
            # Resize to 64x64 and flatten the image
            data.append(cv2.resize(image, (64, 64)).flatten())
            labels.append(label)
    return data, labels

# Load defect-free images
defect_free_data, defect_free_labels = load_data("/home/pi/Defect/Dataset/non_defect", 0) #replace with your directory path

# Load defective images
defective_data, defective_labels = load_data("/home/pi/Defect/Dataset/defective", 1) #replace with your directory path

# Load conveyor belt images
conveyor_data, conveyor_labels = load_data("/home/pi/Defect/Dataset/conveyor", 2) #replace with your directory path

# Combine the data
data = np.array(defect_free_data + defective_data)
labels = np.array(defect_free_labels + defective_labels)

# Save data for reuse
np.save("data.npy", data)
np.save("labels.npy", labels)
print("Dataset prepared and saved.")
