import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf
import requests
import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, \
    convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat
import requests
import os

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path

import logging
import os
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from instagrapi import Client

IG = Client()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)

FILE_DIR = "./Files"
if not os.path.exists(FILE_DIR):
    os.makedirs(FILE_DIR)



def send_file(file_path: Path):  # Use Path type annotation
    try:
        # Verify file extension
        file_extension = file_path.suffix.lower()  # Use the Path object's 'suffix' attribute
        if file_extension != '.mp4':
            print("Only MP4 files are allowed.")
            return False

        api_url = "https://api.reality.org.in/api/v1/t2v"
        headers = {
            "AUTH-KEY": "WEWILLFALLAGAIN",
            "Content-Type": "multipart/form-data",
        }

        with open(file_path, 'rb') as file:
            files = {'file': (file_path.name, file)}  # Use file_path.name to get the filename
            response = requests.post(api_url, headers=headers, files=files, timeout=30)
            if response.status_code == 200:
                return True
            else:
                print(response.text)
                return False
    except Exception as e:
        print(e)
        return False


def initiate_animation(prompt_str:str):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"Output/{time_str}"
    os.makedirs(savedir)
    pretrained_model_path = os.path.join(os.getcwd(), "models", "StableDiffusion")
    inference_config = OmegaConf.load(os.path.join(os.getcwd(), "configs", "inference", "inference.yaml"))

    config_dict = {'ToonYou':
                       {'base': '',
                        'path': os.path.join(os.getcwd(), 'models', 'DreamBooth_LoRA', 'toonyou_beta3.safetensors'),
                        'motion_module': [os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v14.ckpt'),
                                          os.path.join(os.getcwd(), 'models', 'Motion_Module', 'mm_sd_v15.ckpt')],
                        'seed': [10788741199826055526, 6520604954829636163, 6519455744612555650, 16372571278361863751],
                        'steps': 25,
                        'guidance_scale': 7.5,
                        'prompt': [prompt_str],
                        'n_prompt': [
                            'badhandv4,easynegative,ng_deepnegative_v1_75t,verybadimagenegative_v1.3, bad-artist, bad_prompt_version2-neg,']
                        }
                   }
    config = OmegaConf.create(config_dict)
    samples = []

    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:

            ### >>> create validation pipeline >>> ###
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
            unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet",
                                                           unet_additional_kwargs=OmegaConf.to_container(
                                                               inference_config.unet_additional_kwargs))

            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0

            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)

                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)

                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)

                                # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            pipeline.to("cuda")
            ### <<< create validation pipeline <<< ###

            prompts = model_config.prompt
            n_prompts = list(model_config.n_prompt) * len(prompts) if len(
                model_config.n_prompt) == 1 else model_config.n_prompt

            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

                # manually set random seed for reproduction
                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    negative_prompt=n_prompt,
                    num_inference_steps=model_config.steps,
                    guidance_scale=model_config.guidance_scale,
                    width=512,
                    height=512,
                    video_length=16,
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/{sample_idx}-{prompt}.mp4")
                print(f"save to {savedir}/{prompt}.mp4")
                send_file(Path(f"{savedir}/{sample_idx}-{prompt}.mp4"))
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.mp4", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")




class GenT2V:
    def __init__(self,
                 libname="Gustavosta/Stable-Diffusion-Prompts",
                 filename="prompts.xlsx",
                 prompt=None,
                 ):
        self.libname: str = libname
        self.filename: Path = Path(os.path.join(FILE_DIR, filename))
        self.prompt = prompt
        if not self.prompt:
            LOGGER.info("Prompt Not Provided, Getting Random Prompt from Dataset...")
            self.prompt = self.prompt_from_dataset()

    def update_dataset(self):
        LOGGER.info("Updating Prompt dataset...")
        dataset = load_dataset(self.libname)
        p1 = dataset["test"]
        p1_df = pd.DataFrame(p1)
        p1_df.to_excel(self.filename, index=False)
        LOGGER.info(
            f"Prompt Dataset {self.filename}\n Updated With {self.libname} Library\n Total Prompts: {len(p1_df)}")
        return True

    def prompt_from_dataset(self):
        LOGGER.info("Getting Random Prompt from Dataset...")
        if not self.filename.exists() or not self.filename.is_file():
            LOGGER.error(f"No Prompt Dataset Found, Creating New Dataset...")
            self.update_dataset()
            return False

        df = pd.read_excel(self.filename)
        random_index = random.randint(1, len(df) - 1)
        random_prompt = df.loc[random_index, "Prompt"]
        return random_prompt if random_prompt else "AI Bot Crashing, Alert,404,4k"

    def gen_t2v(self):
        LOGGER.info("Generating Text to Video...")
        initiate_animation(self.prompt)
        return True

if __name__ == "__main__":
    t2v = GenT2V()
    t2v.prompt = t2v.prompt_from_dataset()
    LOGGER.info(f"Using Prompt: {t2v.prompt[:20]}...{t2v.prompt[-20:]}")
    LOGGER.info("Running Model...")
    t2v.gen_t2v()
