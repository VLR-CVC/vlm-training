import json
from pathlib import Path

# --- Configuration ---
# 1. Set your file names
INPUT_FILE = '/gpfs/scratch/ehpc391/finevision_data_test/converted_dataset.jsonl'
OUTPUT_FILE = '/gpfs/scratch/ehpc391/finevision_data_test/converted_dataset_new.jsonl'

# 2. Set the key you want to change
KEY_TO_CHANGE = 'image'

# 3. Set the new base path you want
NEW_BASE_PATH = '/gpfs/scratch/ehpc391/finevision_data_test/images/'
# ---------------------

# Create a Path object for the new base directory
new_base = Path(NEW_BASE_PATH)

print(f"Starting processing...")
print(f"  Input file: {INPUT_FILE}")
print(f"  Output file: {OUTPUT_FILE}")
print(f"  Key to modify: {KEY_TO_CHANGE}")
print(f"  New base path: {NEW_BASE_PATH}")

try:
    with open(INPUT_FILE, 'r') as infile, open(OUTPUT_FILE, 'w') as outfile:
        
        processed_lines = 0
        for line in infile:
            try:
                # Load the JSON object from the line
                data = json.loads(line)

                # Check if the key we want to change exists in this line
                if KEY_TO_CHANGE in data and data[KEY_TO_CHANGE]:
                    
                    # Get the old path string and convert to a Path object
                    old_path = Path(data[KEY_TO_CHANGE])

                    # Get just the filename (e.g., 'image.jpg')
                    # This is the core of your request
                    filename = old_path.name 

                    # Create the new path by joining the new base and the filename
                    # The '/' operator is used by pathlib to join paths correctly
                    new_path = new_base / filename 

                    # Update the dictionary with the new path as a string
                    data[KEY_TO_CHANGE] = str(new_path)
                
                # Write the (possibly modified) JSON object back to the new file
                json.dump(data, outfile)
                outfile.write('\n') # Add the newline to keep the JSONL format
                processed_lines += 1

            except json.JSONDecodeError:
                print(f"Skipping bad line (not valid JSON): {line.strip()}")

    print(f"\nSuccessfully processed {processed_lines} lines.")
    print(f"Output saved to {OUTPUT_FILE}")

except FileNotFoundError:
    print(f"\nError: Input file not found at {INPUT_FILE}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
