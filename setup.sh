#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "  Ensemble Deepfake Detector - Setup Script"
echo "============================================================"

# Get script directory
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# ── 1. Virtual Environment ────────────────────────────────────
echo -e "\n${BLUE}[1/5] Checking virtual environment...${NC}"

if [ -d ".venv" ]; then
    echo -e "${GREEN}✅ Virtual environment exists${NC}"
else
    echo -e "${YELLOW}⚙️  Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# ── 2. Install Dependencies ───────────────────────────────────
echo -e "\n${BLUE}[2/5] Installing dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt --progress-bar on --disable-pip-version-check
    echo -e "${GREEN}✅ All dependencies installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# ── 3. Download Image Models ──────────────────────────────────
echo -e "\n${BLUE}[3/5] Checking image models...${NC}"

# Define model directories
VIT_DIR="ImageDetection/models/vit"
SIGLIP_DIR="ImageDetection/models/siglip"

# Create directories if they don't exist
echo -e "${YELLOW}📁 Creating model directories...${NC}"
mkdir -p "$VIT_DIR"
mkdir -p "$SIGLIP_DIR"

# Image Model URLs - Replace with actual direct download links
VIT_URL="https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/model.safetensors"
SIGLIP_URL="https://huggingface.co/prithivMLmods/deepfake-detector-model-v1/resolve/main/model.safetensors"

if [ -f "$VIT_DIR/model.safetensors" ]; then
    echo -e "${GREEN}✅ ViT model already present${NC}"
else
    echo -e "${YELLOW}⬇️  Downloading ViT model...${NC}"
    wget --quiet --show-progress --progress=bar:force:noscroll -P "$VIT_DIR" "$VIT_URL"
    echo -e "${GREEN}✅ ViT model downloaded${NC}"
fi

if [ -f "$SIGLIP_DIR/model.safetensors" ]; then
    echo -e "${GREEN}✅ SigLIP model already present${NC}"
else
    echo -e "${YELLOW}⬇️  Downloading SigLIP model...${NC}"
    wget --quiet --show-progress --progress=bar:force:noscroll -P "$SIGLIP_DIR" "$SIGLIP_URL"
    echo -e "${GREEN}✅ SigLIP model downloaded${NC}"
fi

# ── 4. Download Video Models ──────────────────────────────────
echo -e "\n${BLUE}[4/5] Checking video models...${NC}"

# Define video weights directory
VIDEO_DIR="VideoDetection/weights"

# Create directory if it doesn't exist
echo -e "${YELLOW}📁 Creating video weights directory...${NC}"
mkdir -p "$VIDEO_DIR"

# Video Model URLs - Replace with actual direct download links
VIDEO_ED_URL="https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_ed_inference.pth"
VIDEO_VAE_URL="https://huggingface.co/Deressa/GenConViT/resolve/main/genconvit_vae_inference.pth"

if [ -f "$VIDEO_DIR/genconvit_ed_inference.pth" ]; then
    echo -e "${GREEN}✅ GenConViT-ED model already present${NC}"
else
    echo -e "${YELLOW}⬇️  Downloading GenConViT-ED model...${NC}"
    wget --quiet --show-progress --progress=bar:force:noscroll -P "$VIDEO_DIR" "$VIDEO_ED_URL"
    echo -e "${GREEN}✅ GenConViT-ED model downloaded${NC}"
fi

if [ -f "$VIDEO_DIR/genconvit_vae_inference.pth" ]; then
    echo -e "${GREEN}✅ GenConViT-VAE model already present${NC}"
else
    echo -e "${YELLOW}⬇️  Downloading GenConViT-VAE model...${NC}"
    wget --quiet --show-progress --progress=bar:force:noscroll -P "$VIDEO_DIR" "$VIDEO_VAE_URL"
    echo -e "${GREEN}✅ GenConViT-VAE model downloaded${NC}"
fi

# ── 5. Download Audio Models ──────────────────────────────────
echo -e "\n${BLUE}[5/5] Checking audio models...${NC}"

# Define video weights directory
AUDIO_DIR="AudioDetection/models"

# Create directory if it doesn't exist
echo -e "${YELLOW}📁 Creating audio models directory...${NC}"
mkdir -p "$AUDIO_DIR"

# Video Model URLs - Replace with actual direct download links
AUDIO_URL="https://huggingface.co/mo-thecreator/Deepfake-audio-detection/resolve/main/model.safetensors"

if [ -f "$AUDIO_DIR/model.safetensors" ]; then
    echo -e "${GREEN}✅ Audio model already present${NC}"
else
    echo -e "${YELLOW}⬇️  Downloading Audio model...${NC}"
    wget --quiet --show-progress --progress=bar:force:noscroll -P "$AUDIO_DIR" "$AUDIO_URL"
    echo -e "${GREEN}✅ Audio model downloaded${NC}"
fi


# ── Summary ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Run: python start.py"
echo ""
echo "Or simply run: ./run.sh"
echo ""
