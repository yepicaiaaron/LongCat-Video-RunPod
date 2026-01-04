# LongCat-Video RunPod Serverless Deployment Guide

This guide walks you through deploying LongCat-Video as a RunPod Serverless endpoint.

## Prerequisites

- Docker installed locally (for building)
- Docker Hub account or RunPod GitHub integration
- RunPod account with credits
- HuggingFace account (for downloading model weights)

## Architecture Overview

This deployment wraps LongCat-Video in a RunPod Serverless handler that:
- Loads model weights on worker initialization
- Accepts text-to-video generation requests via JSON
- Returns base64-encoded video frames
- Supports both "fast" (16 steps, ~4sec videos) and "high_quality" (50 steps, ~6sec videos) modes

## Step 1: Build Docker Image

### Option A: Local Build & Push

```bash
# Clone this repository
git clone https://github.com/yepicaiaaron/LongCat-Video-RunPod.git
cd LongCat-Video-RunPod

# Build the Docker image
docker build -t your-dockerhub-username/longcat-video-runpod:latest .

# Push to Docker Hub
docker push your-dockerhub-username/longcat-video-runpod:latest
```

### Option B: RunPod GitHub Integration

1. Fork this repository to your GitHub account
2. In RunPod console → "My Templates" → "New Template"
3. Choose "Use GitHub" and connect your repo
4. RunPod will automatically build from the Dockerfile

## Step 2: Create RunPod Serverless Template

1. Go to RunPod Console → Serverless → "My Templates"
2. Click "New Template"
3. Fill in the template details:

**Container Settings:**
- **Container Image**: `your-dockerhub-username/longcat-video-runpod:latest` (or use GitHub integration)
- **Container Disk**: 50 GB minimum (for model weights)
- **Docker Command**: Leave empty (uses CMD from Dockerfile)

**Environment Variables:**
```
MODEL_PATH=/workspace/weights/LongCat-Video
HF_HOME=/workspace/huggingface
HF_TOKEN=your_huggingface_token_here
CONTEXT_PARALLEL_SIZE=1
```

**GPU Requirements:**
- Minimum: 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- Recommended: 40GB+ VRAM (e.g., A100, H100) for better performance
- Resolution and video length are constrained by available VRAM

**Advanced Settings:**
- **Idle Timeout**: 30 seconds (adjust based on your use case)
- **Max Workers**: 1-3 (depending on expected load)
- **GPUs per Worker**: 1

4. Save the template

## Step 3: Create Endpoint from Template

1. Go to "Serverless" → "Endpoints"
2. Click "New Endpoint"
3. Select your newly created template
4. Configure endpoint:
   - **Name**: `longcat-video-production`
   - **GPUs**: Select GPU types (A100, H100, or RTX 4090 recommended)
   - **Workers**: Set min/max workers
   - **Idle Timeout**: 30-60 seconds
5. Create the endpoint

## Step 4: Download Model Weights (First Run)

On first worker initialization, the model weights need to be downloaded from HuggingFace:

```python
# This happens automatically when the worker starts
# Make sure HF_TOKEN is set in your environment variables
```

Alternatively, you can bake weights into your Docker image (faster cold starts):

1. Uncomment the weight download section in the Dockerfile
2. Pass your HuggingFace token as a build arg:

```bash
docker build --build-arg HF_TOKEN=your_token -t your-image .
```

## Step 5: Test Your Endpoint

### Using RunPod Python SDK

```python
import runpod
import base64
import numpy as np

runpod.api_key = "your-runpod-api-key"

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Fast mode (16 steps, 4 seconds)
result = endpoint.run_sync({
    "input": {
        "prompt": "A white cat sitting on a park bench, realistic photography",
        "mode": "fast",
        "seed": 42
    }
})

# Decode video
video_b64 = result["video_base64"]
video_shape = result["shape"]
video_bytes = base64.b64decode(video_b64)
video_np = np.frombuffer(video_bytes, dtype=np.uint8).reshape(video_shape)

print(f"Generated video: {video_np.shape} at {result['fps']} fps")
```

### Using cURL

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A serene lake at sunset with mountains in the background",
      "mode": "high_quality",
      "seed": 123
    }
  }'
```

## Request Format

```json
{
  "input": {
    "prompt": "Text description of the video",
    "negative_prompt": "Optional negative prompt",
    "mode": "fast" or "high_quality",
    "height": 480,
    "width": 832,
    "num_frames": 61,
    "num_inference_steps": 16,
    "guidance_scale": 1.0,
    "seed": 42
  }
}
```

### Mode Presets

**Fast Mode** (default for quick iterations):
- 16 inference steps
- ~61 frames (4 seconds at 15fps)
- Guidance scale: 1.0
- ~30-60 seconds generation time on A100

**High Quality Mode** (better results, slower):
- 50 inference steps
- ~93 frames (6 seconds at 15fps)
- Guidance scale: 4.0
- ~2-4 minutes generation time on A100

## Response Format

```json
{
  "video_base64": "base64_encoded_video_data",
  "shape": [93, 480, 832, 3],
  "fps": 15,
  "frames": 93,
  "resolution": "832x480",
  "seed": 42
}
```

## Cost Optimization

1. **Idle Timeout**: Set to 30-60s to minimize idle costs
2. **Worker Count**: Start with 1-2 workers, scale up based on traffic
3. **GPU Selection**: 
   - Use RTX 4090 for development/testing (cheaper)
   - Use A100 40GB/80GB for production (faster, better quality)
4. **Bake Weights**: Include weights in Docker image to reduce cold start time
5. **Use Fast Mode**: For prototyping, use fast mode to reduce compute time

## Monitoring & Debugging

- Check RunPod console logs for worker output
- Monitor request latency and success rates
- Use RunPod's built-in metrics dashboard
- Test locally with Docker before deploying:

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_PATH=/workspace/weights/LongCat-Video \
  -e HF_TOKEN=your_token \
  your-image
```

## Troubleshooting

### Worker fails to start
- Check environment variables (especially HF_TOKEN)
- Verify Docker image builds successfully
- Ensure sufficient container disk space (50GB+)

### Out of memory errors
- Reduce `num_frames` or resolution
- Use smaller batch sizes
- Upgrade to higher VRAM GPU

### Slow generation
- Use fast mode for testing
- Ensure using high-end GPU (A100, H100)
- Check if context parallelism is configured correctly

### Model weights not downloading
- Verify HF_TOKEN is valid
- Check network connectivity in worker logs
- Try baking weights into Docker image

## Advanced: Multi-GPU Context Parallelism

For faster inference with large videos, use context parallelism:

1. Update environment variable: `CONTEXT_PARALLEL_SIZE=2`
2. Set GPUs per worker: 2
3. This splits computation across 2 GPUs

Note: This increases cost but can reduce latency for long videos.

## Next Steps

- Integrate with your application backend
- Add video post-processing (compression, format conversion)
- Implement queueing for high-volume use cases
- Set up monitoring and alerts
- Consider adding image-to-video and video continuation endpoints

## Support

For issues specific to this deployment:
- GitHub Issues: https://github.com/yepicaiaaron/LongCat-Video-RunPod/issues

For LongCat-Video model questions:
- Original Repository: https://github.com/meituan-longcat/LongCat-Video

For RunPod platform issues:
- RunPod Discord: https://discord.gg/runpod
- RunPod Docs: https://docs.runpod.io
