import os
import json
from datetime import datetime

# Define root categories based on folder names
CATEGORIES = {
    # "Animated Movies": ["Animation", "Animated Movies"],
    # "Live Action Movies": ["Live Action", "Live Action Movies"],
    # "Animated Shows": ["Animated Shows"],
    # "Live Action Shows": ["Live Action Shows"],
    "Music": ["Music"],
    "Music Videos": ["Music Video", "Music Videos"]
}

# Exclude files inside certain paths (e.g., game folders)
EXCLUDED_FOLDERS = ["Games"]

# Define drives to scan
DRIVES_TO_SCAN = ["E:\\", "F:\\"]  # Add more drives if needed


# Function to categorize files based on folder structure
def categorize_files(drives):
    media_list = []

    for root_dir in drives:
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded folders
            if any(excluded in root for excluded in EXCLUDED_FOLDERS):
                continue

            folders_in_path = root.split(os.sep)

            for category, folder_keywords in CATEGORIES.items():
                if any(folder_name in folders_in_path for folder_name in folder_keywords):
                    for file in files:
                        file_path = os.path.join(root, file)
                        media_list.append({
                            "name": file,
                            "location": file_path,
                            "category": category
                        })
                    break  # Avoid duplicate classification

    return media_list


# Run the script for multiple drives
pc_files = categorize_files(DRIVES_TO_SCAN)

# Save the results
with open("pc_content.json", "w") as f:
    json.dump(pc_files, f, indent=4)

print("Categorization complete! Data saved in pc_content.json.")