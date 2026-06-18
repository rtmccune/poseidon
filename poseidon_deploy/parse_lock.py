import sys
import yaml
import urllib.parse  # Added this to fix the encoding error

# Input and Output filenames
input_file = "poseidon-linux-64-lock.yml"
output_file = "explicit-spec.txt"

print(f"Parsing {input_file}...")

try:
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}")
    sys.exit(1)

# The explicit spec file must start with this exact header
explicit_lines = ["@EXPLICIT"]
pip_packages = []

# Loop through packages
count = 0
for pkg in data.get('package', []):
    # Only process packages for this platform
    if pkg.get('platform') == 'linux-64':
        
        # Handle CONDA packages
        if pkg.get('manager') == 'conda':
            raw_url = pkg.get('url')
            
            # FIX: Decode characters like %21 back to ! so Conda accepts the version string
            url = urllib.parse.unquote(raw_url)
            
            # Append hash if available (security best practice)
            if 'hash' in pkg and 'md5' in pkg['hash']:
                url += f"#{pkg['hash']['md5']}"
            elif 'hash' in pkg and 'sha256' in pkg['hash']:
                pass 
                
            explicit_lines.append(url)
            count += 1
            
        # Handle PIP packages (Save for later)
        elif pkg.get('manager') == 'pip':
            pip_packages.append(f"{pkg['name']}=={pkg['version']}")

# Write the Conda explicit spec
with open(output_file, 'w') as f:
    f.write("\n".join(explicit_lines))

print(f"Success! Extracted {count} conda packages to '{output_file}'.")

# Alert user about Pip packages
if pip_packages:
    print("\n⚠️  WARNING: Found pip dependencies that cannot be installed via explicit spec.")
    print("Run these commands after activating the environment:")
    print(f"pip install {' '.join(pip_packages)}")
