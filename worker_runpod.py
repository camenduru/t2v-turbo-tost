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

@torch.inference_mode()
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
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})