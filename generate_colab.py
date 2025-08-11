import json
import argparse
import os

# --- Model & URL Mappings ---
# This dictionary will map user-friendly model names to their download URLs and filenames.
MAIN_MODELS_MAP = {
    "stable-diffusion-v1-5": {
        "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
        "filename": "v1-5-pruned-emaonly.ckpt"
    },
    "dreamshaper-8": {
        "url": "https://huggingface.co/lykon/dreamshaper-8/resolve/main/dreamshaper-8.safetensors",
        "filename": "dreamshaper-8.safetensors"
    }
}

CONTROLNET_MODELS_MAP = {
    "canny": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
        "filename": "control_v11p_sd15_canny.pth"
    },
    "openpose": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
        "filename": "control_v11p_sd15_openpose.pth"
    },
    "depth": {
        "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
        "filename": "control_v11f1p_sd15_depth.pth"
    }
}

def create_notebook_structure(config, cell1_code, cell2_code):
    """Creates the final Colab notebook structure as a Python dictionary."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "include_colab_link": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            }
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "colab": {
                        "base_uri": "https://localhost:8080/"
                    },
                    "id": "setup_cell"
                },
                "source": cell1_code.split('\n'),
                "outputs": []
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "colab": {
                        "base_uri": "https://localhost:8080/"
                    },
                    "id": "launch_cell"
                },
                "source": cell2_code.split('\n'),
                "outputs": []
            }
        ]
    }
    return notebook

def generate_setup_cell_code(config):
    """Generates the Python code for the first (setup) cell."""

    # Dynamically generate git clone commands for custom nodes
    custom_nodes_code = ""
    for repo_url in config.get("extra_custom_nodes", []):
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        custom_nodes_code += f"""
print(f"\\n🔄 Checking and installing custom node: {repo_name}...")
node_path = os.path.join(custom_nodes_path, '{repo_name}')
if os.path.exists(node_path):
    print(f"  -> {repo_name} already exists. Updating...")
    %cd {{node_path}}
    !git pull
else:
    print(f"  -> Cloning {repo_name}...")
    %cd {{custom_nodes_path}}
    !git clone {repo_url}
"""

    # Dynamically generate download commands for main models
    main_models_code = ""
    for model_name in config.get("main_models", []):
        model_info = MAIN_MODELS_MAP.get(model_name)
        if model_info:
            main_models_code += f"""
print(f'\\n🔄 Checking main model: {model_name}...')
model_file_path = os.path.join(checkpoints_path, '{model_info["filename"]}')
if not os.path.exists(model_file_path):
    print(f'  -> Downloading {model_info["filename"]}...')
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{model_info["url"]}" -d "{{checkpoints_path}}" -o "{model_info["filename"]}"
else:
    print(f'  -> Model {model_info["filename"]} already exists.')
"""

    # Dynamically generate download commands for controlnet models
    controlnet_models_code = ""
    for model_name in config.get("controlnet_models", []):
        model_info = CONTROLNET_MODELS_MAP.get(model_name)
        if model_info:
            controlnet_models_code += f"""
print(f'\\n🔄 Checking ControlNet model: {model_name}...')
model_file_path = os.path.join(controlnet_path, '{model_info["filename"]}')
if not os.path.exists(model_file_path):
    print(f'  -> Downloading {model_info["filename"]}...')
    !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{model_info["url"]}" -d "{{controlnet_path}}" -o "{model_info["filename"]}"
else:
    print(f'  -> Model {model_info["filename"]} already exists.')
"""

    # Assemble the complete code for the cell using an f-string
    cell_code = f"""
#@title 1. Setup & Caching
#@markdown This cell handles all installations, downloads, and caching.
#@markdown It will connect to your Google Drive and store everything in `/content/drive/MyDrive/AI_Art_Station`.
#@markdown Subsequent runs will be much faster.

import os
import time
from google.colab import drive

print("🚀 Starting setup...")
start_time = time.time()

# --- 1. Mount Google Drive ---
print("\\n🔄 [1/7] Mounting Google Drive...")
drive.mount('/content/drive')
print("✅ Google Drive mounted successfully!")

# --- 2. Define Paths & Create Directories ---
print("\\n🔄 [2/7] Setting up directory structure...")
GDRIVE_ROOT = "/content/drive/MyDrive/AI_Art_Station"
working_dir = GDRIVE_ROOT
comfyui_path = os.path.join(working_dir, 'ComfyUI')
custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')
models_path = os.path.join(comfyui_path, 'models')
checkpoints_path = os.path.join(models_path, 'checkpoints')
controlnet_path = os.path.join(models_path, 'controlnet')

os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(controlnet_path, exist_ok=True)
os.makedirs(custom_nodes_path, exist_ok=True)
print("✅ Directory structure is ready.")

# --- 3. Install/Update ComfyUI ---
print("\\n🔄 [3/7] Checking and installing ComfyUI...")
if os.path.exists(comfyui_path):
    print("  -> ComfyUI already exists. Performing a quick update...")
    %cd {{comfyui_path}}
    !git pull
else:
    print("  -> ComfyUI not found. Performing a fresh installation...")
    %cd {{working_dir}}
    !git clone https://github.com/comfyanonymous/ComfyUI.git
print("✅ ComfyUI is ready.")

# --- 4. Install/Update Custom Nodes ---
{custom_nodes_code}
print("\\n✅ All custom nodes are ready.")

# --- 5. Install Dependencies ---
print("\\n🔄 [5/7] Installing Python dependencies...")
%cd {{comfyui_path}}
!pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -q
print("✅ Python dependencies installed.")

# --- 6. Install Aria2c for Accelerated Downloads ---
print("\\n🔄 [6/7] Installing aria2c downloader...")
!apt-get -y install -qq aria2
print("✅ aria2c is ready.")

# --- 7. Download Models ---
{main_models_code}
{controlnet_models_code}
print("\\n✅ All models are checked and ready.")

# --- Finalization ---
end_time = time.time()
print("\\n" + "="*50)
print(f"🎉 Setup complete! Total time: {{end_time - start_time:.2f}} seconds.")
print("✅ You can now proceed to the next cell to launch the UI.")
print("="*50)
"""
    return cell_code.strip()

def generate_launch_cell_code(config):
    """Generates the Python code for the second (launch) cell."""

    # The user_interface variable from config isn't used yet, but is here for future-proofing
    ui_name = config.get("user_interface", "ComfyUI")

    cell_code = f"""
#@title 2. Launch {ui_name}
#@markdown ### ⚙️ Hardware and Network Options
#@markdown Select your hardware and preferred network tunnel.
#@markdown - **`GPU`** is recommended for generating images.
#@markdown - **`CPU`** is for workflow management without using GPU credits.
#@markdown - **`Cloudflared`** is the recommended, stable, and unlimited tunnel.
#@markdown ---
hardware = "GPU"  #@param ["GPU", "CPU"]
tunnel = "Cloudflared"  #@param ["Cloudflared", "Ngrok"]

import os
import time
import sys

# --- 1. Define Paths ---
GDRIVE_ROOT = "/content/drive/MyDrive/AI_Art_Station"
comfyui_path = os.path.join(GDRIVE_ROOT, 'ComfyUI')

# --- 2. Launch Tunnel Service ---
if tunnel == 'Cloudflared':
    print("🚀 Launching Cloudflared tunnel...")
    !pkill cloudflared
    !nohup cloudflared tunnel --url http://127.0.0.1:8188 > /content/cloudflared.log 2>&1 &
    time.sleep(4) # Wait for log file to be created
    print("🔗 Your Public URL:")
    !grep -o 'https://[a-zA-Z0-9-]*\\.trycloudflare.com' /content/cloudflared.log
    print("✅ Cloudflared tunnel is active. Please use the link above.")
else: # Ngrok
    print("🚀 Launching Ngrok tunnel...")
    !pip install pyngrok -q
    from pyngrok import ngrok
    ngrok.kill()
    try:
        ngrok_auth_token = os.environ.get('NGROK_AUTH_TOKEN', 'YOUR_NGROK_AUTHTOKEN_HERE')
        ngrok.set_auth_token(ngrok_auth_token)
        public_url = ngrok.connect(8188)
        print(f"🔗 Your Public URL: {{public_url}}")
        print("✅ Ngrok tunnel is active. (Note: Free tier has time limits)")
    except Exception as e:
        print(f"❌ Ngrok Error: {{e}}")
        print("Please ensure your NGROK_AUTH_TOKEN is set correctly in Colab secrets.")


# --- 3. Construct and Run Launch Command ---
%cd {{comfyui_path}}

launch_command = "python main.py --listen"
if hardware == 'CPU':
    launch_command += " --cpu-only" # Correct flag for CPU-only mode might vary
else:
    launch_command += " --cuda-device 0" # Or other GPU-specific flags

print(f"\\n🔥 Launching {ui_name} with command: `{{launch_command}}`")
print("🟢 Service is starting. This cell will continue to run.")
!{{launch_command}}
"""
    return cell_code.strip()


def main():
    """Main function to generate the Colab notebook."""
    parser = argparse.ArgumentParser(description="Generate a Google Colab notebook from a JSON config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.config}")
        return

    # Generate code for each cell
    setup_code = generate_setup_cell_code(config)
    launch_code = generate_launch_cell_code(config)

    # Create the notebook structure
    notebook = create_notebook_structure(config, setup_code, launch_code)

    # Write the notebook to a file
    output_filename = config.get("colab_notebook_filename", "generated_colab.ipynb")
    with open(output_filename, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully generated notebook: {output_filename}")


if __name__ == "__main__":
    main()
