import os, subprocess, requests, json, runpod

from omegaconf import OmegaConf
import random
import imageio
import torch
import torchvision
import numpy as np
from utils.lora import collapse_lora, monkeypatch_remove_lora
from utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

MAX_SEED = np.iinfo(np.int32).max
device = torch.device("cuda:0")
config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
model_config = config.pop("model", OmegaConf.create())
pretrained_t2v = instantiate_from_config(model_config)
pretrained_t2v = load_model_checkpoint(pretrained_t2v, "checkpoints/vc2_model.ckpt")
unet_config = model_config["params"]["unet_config"]
unet_config["params"]["time_cond_proj_dim"] = 256
unet = instantiate_from_config(unet_config)
unet.load_state_dict(pretrained_t2v.model.diffusion_model.state_dict(), strict=False)
use_unet_lora = True
lora_manager = LoraHandler(
    version="cloneofsimo",
    use_unet_lora=use_unet_lora,
    save_for_webui=True,
    unet_replace_modules=["UNetModel"],
)
lora_manager.add_lora_to_model(
    use_unet_lora,
    unet,
    lora_manager.unet_replace_modules,
    lora_path="checkpoints/unet_lora.pt",
    dropout=0.1,
    r=64,
)
unet.eval()
collapse_lora(unet, lora_manager.unet_replace_modules)
monkeypatch_remove_lora(unet)
pretrained_t2v.model.diffusion_model = unet
scheduler = T2VTurboScheduler(
    linear_start=model_config["params"]["linear_start"],
    linear_end=model_config["params"]["linear_end"],
)
pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
pipeline.to(device)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def save_video(video_array, video_save_path, fps: int = 16):
    video = video_array.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(
        video_save_path, video, fps=fps, video_codec="h264", options={"crf": "10"}
    )

def generate(input):
    values = input["input"]
    prompt = values["prompt"]
    guidance_scale = values["guidance_scale"]
    num_inference_steps = values["num_inference_steps"]
    num_frames = values["num_frames"]
    fps = values["fps"]
    seed = values["seed"]
    randomize_seed = values["randomize_seed"]

    seed = int(randomize_seed_fn(seed, randomize_seed))
    result = pipeline(
        prompt=prompt,
        frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_videos_per_prompt=1,
    )
    torch.cuda.empty_cache()
    tmp_save_path = "tmp.mp4"
    root_path = "./videos/"
    os.makedirs(root_path, exist_ok=True)
    video_save_path = os.path.join(root_path, tmp_save_path)
    save_video(result[0], video_save_path, fps=fps)
    display_model_info = f"Video size: {num_frames}x320x512, Sampling Step: {num_inference_steps}, Guidance Scale: {guidance_scale}"

    result = video_save_path

    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})