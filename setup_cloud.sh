#!/bin/bash
# Quick deployment script for RunPod/VM

echo "================================"
echo "DeepSeek OCR Worker Setup"
echo "================================"

# Update system
echo "1. Updating system..."
apt-get update -qq

# Install dependencies if needed
echo "2. Installing system dependencies..."
apt-get install -y git python3-pip libgl1-mesa-glx libglib2.0-0 -qq

# Install Python packages
echo "3. Installing Python packages..."
pip install -q -r requirements.txt

# Check CUDA
echo "4. Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check environment variables
echo "5. Checking environment variables..."
if [ -z "$SUPABASE_URL" ]; then
    echo "⚠️  SUPABASE_URL not set!"
else
    echo "✓ SUPABASE_URL set"
fi

if [ -z "$R2_ENDPOINT" ]; then
    echo "⚠️  R2_ENDPOINT not set!"
else
    echo "✓ R2_ENDPOINT set"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "To run worker:"
echo "  python3 workers/ocr/main.py --poll-seconds 3"
echo ""
echo "To run in background:"
echo "  nohup python3 workers/ocr/main.py --poll-seconds 3 > worker.log 2>&1 &"
echo ""
echo "To check logs:"
echo "  tail -f worker.log"
