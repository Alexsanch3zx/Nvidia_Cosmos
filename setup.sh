#!/bin/bash



echo "🚀 Starting GPU instance setup..."

sudo apt update
sudo apt install -y python3.10-venv

# --- Create virtual environment ---
echo "🌱 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- Upgrade pip ---
pip install --upgrade pip

# --- Install dependencies ---
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install streamlit torchvision accelerate huggingface_hub

# --- Fix known issue (jinja2 error) ---
pip install "jinja2>=3.1.0"



echo "✅ Setup complete!"