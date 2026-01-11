# RunPod Deployment - Step by Step

## 📝 Prerequisites
✅ RunPod account created
✅ Pod deployed with GPU
✅ Environment variables sudah dimasukkan di Pod settings

---

## 🚀 Step-by-Step Commands

### 1. Connect ke Pod
Klik tombol **"Connect"** di RunPod dashboard, lalu pilih **"Start Web Terminal"** atau **"Connect with SSH"**

### 2. Clone Repository
```bash
cd /workspace
git clone https://github.com/YOUR-USERNAME/chapterbridge-deepseek-ocr-worker.git
cd chapterbridge-deepseek-ocr-worker
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Output yang diharapkan:**
- Installing torch, transformers, dll
- Займет sekitar 2-3 menit

### 4. Test Environment Variables
```bash
python3 -c "import os; print('SUPABASE_URL:', os.getenv('SUPABASE_URL')[:30] + '...' if os.getenv('SUPABASE_URL') else 'NOT SET')"
python3 -c "import os; print('R2_ENDPOINT:', os.getenv('R2_ENDPOINT')[:30] + '...' if os.getenv('R2_ENDPOINT') else 'NOT SET')"
```

**Output yang diharapkan:**
```
SUPABASE_URL: https://czkmfderwtnltzlytzig...
R2_ENDPOINT: https://7179c252774c3316da88...
```

### 5. Test CUDA
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Output yang diharapkan:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

### 6. Run Worker (Foreground - untuk testing)
```bash
python3 workers/ocr/main.py --poll-seconds 3
```

**Output yang diharapkan:**
```
2026-01-11 17:00:00 | INFO | Starting OCR worker daemon (poll interval: 3s)
2026-01-11 17:00:00 | INFO | Initializing DeepSeek-OCR engine (this may take a moment)
2026-01-11 17:00:05 | INFO | Loading DeepSeek-OCR model from: deepseek-ai/DeepSeek-OCR
...
2026-01-11 17:00:45 | INFO | DeepSeek-OCR loaded on CUDA with bfloat16
2026-01-11 17:00:45 | INFO | DeepSeek-OCR model initialized successfully
2026-01-11 17:00:45 | INFO | Polling for OCR jobs...
```

⏱️ **Model loading time**: 30-60 detik di GPU (vs 5+ menit di CPU)

### 7. Jika Worker Running dengan Baik (ada log "Polling for OCR jobs..."):
**Tekan Ctrl+C** untuk stop, lalu run di background:

```bash
nohup python3 workers/ocr/main.py --poll-seconds 3 > worker.log 2>&1 &
```

**Check process:**
```bash
ps aux | grep "workers/ocr/main.py"
```

### 8. Monitor Logs
```bash
# Real-time logs
tail -f worker.log

# Last 50 lines
tail -n 50 worker.log

# Search for errors
grep -i error worker.log
```

### 9. Check Job Processing
```bash
# Count processed jobs
python3 -c "
from workers.ocr.supabase_client import SupabaseDB
db = SupabaseDB()
result = db.client.table('pipeline_jobs').select('status').eq('job_type', 'clean').execute()
print(f'Total jobs: {len(result.data)}')
print(f'Queued: {len([j for j in result.data if j[\"status\"] == \"queued\"])}')
print(f'Running: {len([j for j in result.data if j[\"status\"] == \"running\"])}')
print(f'Success: {len([j for j in result.data if j[\"status\"] == \"success\"])}')
print(f'Failed: {len([j for j in result.data if j[\"status\"] == \"failed\"])}')
"
```

---

## 🔧 Troubleshooting

### Problem: Environment variables tidak terbaca

**Solution:**
```bash
# Set manually di terminal
export SUPABASE_URL="https://czkmfderwtnltzlytzig.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="eyJhbGc..."
export R2_ENDPOINT="https://7179c252774c3316da886216d661ac21.r2.cloudflarestorage.com"
export R2_ACCESS_KEY_ID="bdd2a6a..."
export R2_SECRET_ACCESS_KEY="05bd9c5..."
export R2_BUCKET="chapterbridge-data"

# Atau buat .env file
cat > .env << 'EOF'
SUPABASE_URL=https://czkmfderwtnltzlytzig.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
R2_ENDPOINT=https://7179c252774c3316da886216d661ac21.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=bdd2a6a...
R2_SECRET_ACCESS_KEY=05bd9c5...
R2_BUCKET=chapterbridge-data
EOF
```

### Problem: Model loading fails

**Check:**
```bash
# Check disk space
df -h /workspace

# Check internet
curl -I https://huggingface.co

# Check GPU memory
nvidia-smi
```

### Problem: Worker crashes

**Check logs:**
```bash
tail -100 worker.log
```

**Common issues:**
- Out of GPU memory → Use smaller batch size
- Network issues → Check R2/Supabase credentials
- Jobs not found → Check database

---

## 🛑 Stop Worker

```bash
# Find process ID
ps aux | grep "workers/ocr/main.py"

# Kill process
pkill -f "workers/ocr/main.py"

# Verify stopped
ps aux | grep "workers/ocr/main.py"
```

---

## 🔄 Restart Worker

```bash
# Stop
pkill -f "workers/ocr/main.py"

# Start
nohup python3 workers/ocr/main.py --poll-seconds 3 > worker.log 2>&1 &

# Check
tail -f worker.log
```

---

## 💡 Tips

1. **Keep Pod Running**: Jangan stop pod saat worker sedang processing
2. **Monitor Costs**: RunPod charge per minute, stop pod jika tidak digunakan
3. **Backup Logs**: Download `worker.log` sebelum stop pod
4. **Use Persistent Storage**: Simpan model di volume agar tidak download ulang

---

## 📊 Expected Performance

**With GPU (RTX 3060 Ti):**
- Model load: 30-60 seconds
- Per image processing: 5-10 seconds
- 100 images/hour ≈ $0.20 total cost

**Signs of Success:**
✅ "DeepSeek-OCR loaded on CUDA with bfloat16"
✅ "Polling for OCR jobs..."
✅ "[job=xxx] Starting OCR job"
✅ "[job=xxx] OCR completed in X ms"
✅ Jobs status change dari "queued" → "running" → "success"
