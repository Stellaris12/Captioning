import os

# Prompt the user for input, asking for strings separated by commas
user_input = input("Please enter strings separated by commas: ")

# Split the input string into a list based on the comma separator,
# strip leading/trailing spaces from each element,
# and filter out any empty strings resulting from a trailing comma.
input_array = [item.strip() for item in user_input.split(',') if item.strip()]
print("Array =", input_array)

# Prompt the user for the folder path containing the text files
folder_path = input("Please enter the path to the folder containing text files: ")

# Validate if the folder path exists
if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
    print("Invalid folder path or path does not exist.")
    exit()

# Initialize an empty list to store names of files with matching content
matching_files = []

# Iterate over each file in the directory
for filename in os.listdir(folder_path):
    # Construct full file path
    file_path = os.path.join(folder_path, filename)
    # Ensure it's a file
    if os.path.isfile(file_path) and filename.endswith('.txt'):
        try:
            # Open and read the file content with explicit encoding
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Split the content by commas and strip whitespace
                content_array = [item.strip() for item in content.split(',')]
                # Check for any matches and add filename to matching_files if found
                if any(item in input_array for item in content_array):
                    matching_files.append(file_path)
        except UnicodeDecodeError as e:
            print(f"Error reading {filename}: {e}")

# Print or use the list of matching files as needed
print("Files with matching content:", matching_files)

# Function to delete files based on the provided list of file paths
def delete_files(file_paths):
    for file_path in file_paths:
        # Check if the file exists before attempting deletion
        if os.path.exists(file_path):
            # Delete the text file
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
            # Delete image files with the same name as the text file
            base_name = os.path.splitext(file_path)[0]  # Get the base name without extension
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']  # Add more image formats as needed
            for ext in image_extensions:
                image_file_path = base_name + ext
                if os.path.exists(image_file_path):
                    os.remove(image_file_path)
                    print(f"Deleted image file: {image_file_path}")

# Delete the matching files
delete_files(matching_files)

# Delete the matching files
delete_files(matching_files)
