#!/bin/csh
# Setup script for SteerNet-v2 environment in /tmp (RAM disk)
# Run this script if the environment doesn't exist or after a server reboot

echo "Setting up SteerNet-v2 environment in /tmp..."

# Check if venv already exists
if (-d /tmp/steernet_venv) then
    echo "Environment already exists at /tmp/steernet_venv"
    echo "To recreate, first run: rm -rf /tmp/steernet_venv"
    exit 0
endif

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv /tmp/steernet_venv

# Activate and install packages
echo "Installing PyTorch with CUDA support..."
source /tmp/steernet_venv/bin/activate.csh
pip install --upgrade pip --no-cache-dir
pip install torch --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

echo "Installing gymnasium and highway-env..."
pip install gymnasium highway-env --no-cache-dir

echo ""
echo "Setup complete!"
echo "To activate the environment, run:"
echo "  source /tmp/steernet_venv/bin/activate.csh"
echo ""
echo "To verify GPU is available, run:"
echo "  python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
