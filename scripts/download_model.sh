#!/bin/bash
# Download a model for llamacpp-agentified
# Usage: ./scripts/download_model.sh <model-name>
# Example: ./scripts/download_model.sh qwen3-4b

set -e

MODEL_NAME="${1:?Usage: $0 <model-name>}"
CONFIG_DIR="${CONFIG_DIR:-./config/models}"
MODEL_DIR="${MODEL_DIR:-./data/models}"

CONFIG_FILE="$CONFIG_DIR/$MODEL_NAME.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config not found: $CONFIG_FILE"
    echo "Available models:"
    ls "$CONFIG_DIR"/*.json 2>/dev/null | xargs -I{} basename {} .json
    exit 1
fi

REPO=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['repo'])")
FILENAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['filename'])")

echo "Downloading $MODEL_NAME from $REPO..."
mkdir -p "$MODEL_DIR"

pip install -q huggingface-hub 2>/dev/null
python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='$REPO', filename='$FILENAME', local_dir='$MODEL_DIR')
print(f'Downloaded to: {path}')
"

echo "Done! Model ready at $MODEL_DIR/$FILENAME"
