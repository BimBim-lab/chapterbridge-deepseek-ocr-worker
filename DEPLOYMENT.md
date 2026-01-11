# Cloud Deployment Guide

## Pilihan Platform Cloud

### 🔥 Recommended: RunPod (GPU Support)

**Kelebihan:**
- GPU support (CUDA)
- Pay per minute
- Fast model loading (30s vs 5+ menit di CPU)
- Starting dari $0.20/hour untuk GPU

**Langkah Deploy:**

1. **Sign up**: https://runpod.io
2. **Deploy Pod**:
   ```
   - Template: PyTorch 2.0
   - GPU: RTX 3060 / RTX 4090 (tergantung budget)
   - Disk: 50GB minimum
   ```

3. **Setup di Pod**:
   ```bash
   cd /workspace
   git clone <your-repo-url>
   cd chapterbridge-deepseek-ocr-worker
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set environment variables
   export SUPABASE_URL="..."
   export SUPABASE_SERVICE_ROLE_KEY="..."
   export R2_ENDPOINT="..."
   export R2_ACCESS_KEY_ID="..."
   export R2_SECRET_ACCESS_KEY="..."
   export R2_BUCKET="chapterbridge-data"
   
   # Run worker
   python workers/ocr/main.py --poll-seconds 3
   ```

4. **Optional - Run as background service**:
   ```bash
   nohup python workers/ocr/main.py --poll-seconds 3 > worker.log 2>&1 &
   ```

---

### Option 2: Google Cloud Run (CPU Only)

**Kelebihan:**
- Auto-scaling
- Pay per request
- Free tier available

**Langkah:**

1. Install Google Cloud CLI
2. Build & Deploy:
   ```bash
   gcloud run deploy deepseek-ocr-worker \
     --source . \
     --region us-central1 \
     --memory 16Gi \
     --cpu 4 \
     --timeout 3600 \
     --set-env-vars="SUPABASE_URL=...,SUPABASE_SERVICE_ROLE_KEY=...,R2_ENDPOINT=...,R2_ACCESS_KEY_ID=...,R2_SECRET_ACCESS_KEY=...,R2_BUCKET=chapterbridge-data"
   ```

⚠️ **Note**: Cloud Run tidak support GPU, loading akan tetap lambat.

---

### Option 3: Railway.app

**Kelebihan:**
- Simple deployment dari GitHub
- Auto-deploy on push

**Langkah:**

1. Push code ke GitHub repository
2. Connect Railway to GitHub: https://railway.app
3. Add environment variables di Railway dashboard
4. Deploy automatically

---

### Option 4: Google Compute Engine (VM)

**Kelebihan:**
- Full control
- Bisa pilih GPU instance

**Langkah:**

1. Create VM dengan GPU:
   ```bash
   gcloud compute instances create deepseek-worker \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --boot-disk-size=100GB
   ```

2. SSH ke VM dan setup seperti RunPod

---

## Environment Variables Yang Diperlukan

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Cloudflare R2
R2_ENDPOINT=https://accountid.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET=chapterbridge-data

# DeepSeek Model (optional)
DEEPSEEK_MODEL_PATH=deepseek-ai/DeepSeek-OCR
DEEPSEEK_MAX_NEW_TOKENS=800
DEEPSEEK_TEMPERATURE=0

# Worker Config (optional)
POLL_SECONDS=3
OCR_DEBUG=0
```

---

## Monitoring

### Check logs:
```bash
# If running in background
tail -f worker.log

# Check process
ps aux | grep python

# Kill worker
pkill -f "workers/ocr/main.py"
```

### Check job status di Supabase:
```sql
SELECT id, status, created_at, error 
FROM pipeline_jobs 
WHERE job_type = 'clean' 
ORDER BY created_at DESC 
LIMIT 10;
```

---

## Cost Estimation

**RunPod (GPU):**
- RTX 3060 Ti: ~$0.20/hour
- RTX 4090: ~$0.60/hour
- Process 100 images/hour ≈ $0.002-0.006 per image

**Google Cloud (CPU only):**
- 16GB RAM, 4 vCPU: ~$0.50/hour
- Much slower processing

**Railway:**
- $5/month free tier
- $0.000463/GB-s after that

---

## Best Practice

1. **Use GPU** for production (10x faster)
2. **Set POLL_SECONDS=5** untuk production (less frequent polling)
3. **Monitor logs** untuk error
4. **Set up auto-restart** jika worker crash
5. **Use Docker** untuk consistent environment

---

## Troubleshooting

**Model loading too slow:**
- ✅ Use GPU instance
- ✅ Check RAM (need 16GB+)

**Out of memory:**
- Increase instance RAM to 16GB+
- Use `low_cpu_mem_usage=True` (already enabled)

**Jobs not processing:**
- Check environment variables
- Check database connection
- Check R2 credentials

**Model download failed:**
- Pre-download model: `huggingface-cli download deepseek-ai/DeepSeek-OCR`
- Or copy from local cache
