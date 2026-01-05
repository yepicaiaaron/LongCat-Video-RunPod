"""RunPod Serverless Handler for LongCat-Video

This handler wraps LongCat-Video's text-to-video generation
for deployment on RunPod Serverless infrastructure.
"""
import os
import base64
import datetime
from io import BytesIO
from typing import Dict, Any
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
import runpod

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel

# Global model instances
PIPELINE = None
GENERATOR = None
LOCAL_RANK = 0

def load_models():
    """Load models once on worker startup."""
    global PIPELINE, GENERATOR, LOCAL_RANK
    
    checkpoint_dir = os.environ.get('MODEL_PATH', '/workspace/models/LongCat-Video')
    context_parallel_size = int(os.environ.get('CONTEXT_PARALLEL_SIZE', '1'))

        
    # Setup distributed environment
    rank = int(os.environ.get('RANK', '0'))
    num_gpus = torch.cuda.device_count()
    LOCAL_RANK = rank % num_gpus
    torch.cuda.set_device(LOCAL_RANK)
    
    if context_parallel_size > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        init_context_parallel(context_parallel_size=context_parallel_size, 
                             global_rank=global_rank, world_size=num_processes)
    
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
    
    # Load model components
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", 
                                             torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", 
                                                     torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", 
                                           torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, 
                                                                subfolder="scheduler", 
                                                                torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", 
                                                         cp_split_hw=cp_split_hw, 
                                                         torch_dtype=torch.bfloat16)
    
    PIPELINE = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    PIPELINE.to(LOCAL_RANK)
    
    GENERATOR = torch.Generator(device=LOCAL_RANK)
    
    print(f"Models loaded successfully on GPU {LOCAL_RANK}")

def torch_gc():
    """GPU garbage collection."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler function.
    
    Expected input format:
    {
        "input": {
            "prompt": "your text prompt",
            "negative_prompt": "optional negative prompt",
            "height": 480,
            "width": 832,
            "num_frames": 93,
            "num_inference_steps": 50,
            "guidance_scale": 4.0,
            "seed": 42,
            "mode": "fast" or "high_quality"
        }
    }
    """
    global PIPELINE, GENERATOR, LOCAL_RANK
    
    try:
        job_input = job['input']
        
        # Parse inputs with defaults
        prompt = job_input.get('prompt', '')
        negative_prompt = job_input.get('negative_prompt', 
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
            "paintings, images, static, overall gray, worst quality, low quality")
        
        mode = job_input.get('mode', 'fast')
        
        # Adjust parameters based on mode
        if mode == 'fast':
            height = job_input.get('height', 480)
            width = job_input.get('width', 832)
            num_frames = job_input.get('num_frames', 61)  # ~4 sec at 15fps
            num_inference_steps = job_input.get('num_inference_steps', 16)
            guidance_scale = job_input.get('guidance_scale', 1.0)
        else:  # high_quality
            height = job_input.get('height', 480)
            width = job_input.get('width', 832)
            num_frames = job_input.get('num_frames', 93)  # ~6 sec
            num_inference_steps = job_input.get('num_inference_steps', 50)
            guidance_scale = job_input.get('guidance_scale', 4.0)
        
        seed = job_input.get('seed', 42)
        GENERATOR.manual_seed(seed)
        
        # Generate video
        output = PIPELINE.generate_t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=GENERATOR,
        )[0]
        
        # Convert to video bytes
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        
        # Write to BytesIO instead of file
        video_buffer = BytesIO()
        # Save as numpy array then encode
        video_np = output_tensor.cpu().numpy()
        
        # Encode as base64
        video_b64 = base64.b64encode(video_np.tobytes()).decode('utf-8')
        
        torch_gc()
        
        return {
            "video_base64": video_b64,
            "shape": list(video_np.shape),
            "fps": 15,
            "frames": num_frames,
            "resolution": f"{width}x{height}",
            "seed": seed
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Load models on startup
    load_models()
    
    # Start RunPod serverless
    runpod.serverless.start({"handler": handler})
