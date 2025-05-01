# -*- coding: utf-8 -*-
# Copyright (c) 2024 Salvador E. Tropea
# License: MIT
# Project: ComfyUI OmniGen Nodes

# This code is used to call some components of OmniGen in a different order
# Python base modules
import logging
import os
import pprint

# ML modules
from safetensors.torch import load_file
from transformers import AutoTokenizer
import torch

# Comfy_UI modules
import model_management
from latent_formats import SDXL
import latent_preview

# OmniGen
from .OmniGen import OmniGen, OmniGenProcessor, OmniGenPipeline, OmniGenScheduler
from .OmniGen.utils import show_mem, show_shape, VAE_SCALE_FACTOR, flush_mem, free_mem
from .OmniGen.transformer import Phi3Config


class OmniGenProcessorWrapper(OmniGenProcessor):
    @classmethod
    def from_pretrained(cls):
        text_tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), 'tokenizer'))
        return cls(text_tokenizer)


class OmniGenWrapper(OmniGen):
    @classmethod
    def from_pretrained(cls, cfg_name, model_name, quantize=False):
        assert os.path.isfile(os.path.join(cfg_name, 'config.json'))
        assert os.path.isfile(model_name)
        logging.info("Loading config from "+cfg_name)
        config = Phi3Config.from_pretrained(cfg_name)
        logging.info("Initializing model")
        model = cls(config, pre_trained=True)
        logging.info("Loading safetensors from "+model_name)
        ckpt = load_file(model_name)
        model.load_state_dict(ckpt)
        # Only quantize if explicitly requested
        if quantize:
            logging.info("Quantizing weights to 8-bit...")
            model._quantize_module(model.llm)
        model.quantized = quantize

        return model


# A couple of classes with the information needed by Comfy_UI to show the preview
class FakeComfyModelModel(object):
    def __init__(self):
        # We use the SDXL VAE
        self.latent_format = SDXL()


class FakeComfyModel(object):
    def __init__(self, device):
        self.load_device = device
        self.model = FakeComfyModelModel()


class OmniGenPipelineWrapper(OmniGenPipeline):
    @classmethod
    def from_pretrained(cls, model, dtype = torch.bfloat16, device=None):
        return cls(model, None, dtype=dtype, device=device)  # The processor was moved outside

    # TODO: Move to main class? Undo changes in main class? do one of the two
    @torch.no_grad()
    def __call__(self, conditioner: dict,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 3,
                 img_guidance_scale: float = 1.6,
                 offload_model: bool = False,
                 use_kv_cache: bool = True,
                 offload_kv_cache: bool = True,
                 seed: int = None,
                 vae = None,
                 ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            conmditioner (`dict`):
                The output from the processor, plus output image size
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800).
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            offload_kv_cache (`bool`, *optional*, defaults to True): offload the cached key and value to cpu, which can save memory but slow down the generation silightly
            offload_model (`bool`, *optional*, defaults to False): offload the model to cpu, which can save memory but slow down the generation
            seed (`int`, *optional*):
                A random seed for generating output.
            vae: Comfy_UI VAE object
        Examples:

        Returns:
            A list with the generated images.
        """
        logging.info("Starting OmniGen pipeline")
        show_mem()
        use_img_guidance = conditioner['input_images'] is not None

        # set model and processor
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()

        logging.info("- Input data images")
        show_mem()
        logging.debug('Processor output:')
        logging.debug(f'input_ids: {show_shape(conditioner["input_ids"])}')
        # logging.debug(f'attention_mask {conditioner["attention_mask"]}')
        # logging.debug(f'position_ids: {conditioner["position_ids"]}')
        logging.debug(f'input_pixel_values: {show_shape(conditioner["input_pixel_values"])}')
        logging.debug(f'input_image_sizes: {show_shape(conditioner["input_image_sizes"])}')
        # logging.debug(f'padding_images: {conditioner["padding_images"]}')
        logging.debug('---------------------------------------------------')
        logging.debug(pprint.pformat(conditioner))
        logging.info('---------------------------------------------------')

        num_prompt = conditioner['num_conditions']
        num_cfg = 2 if use_img_guidance else 1
        latent_size_h, latent_size_w = conditioner['height']//8, conditioner['width']//8

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        logging.info("- Create latents")
        show_mem()
        # The model works with 2 or 3 latents, the last is the output. One input is for the (+) the other for the images
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
        latents = torch.cat([latents]*(1+num_cfg), 0).to(self.dtype)

        input_img_latents = []
        if conditioner['separate_cfg_infer']:
            logging.info("- Encoding images separately")
            for temp_pixel_values in conditioner['input_pixel_values']:
                logging.info("  - One conditional")
                temp_input_latents = []
                for img in temp_pixel_values:
                    logging.info(show_shape(img))
                    temp_input_latents.append(self.vae_encode(vae, img))
                input_img_latents.append(temp_input_latents)
        else:
            logging.info("- Encoding all images at once")
            for img in conditioner['input_pixel_values']:
                input_img_latents.append(self.vae_encode(vae, img))
        # Stop here if we are skipping the model load
        assert self.model is not None, "Stopped because we didn't load the model"

        callback = latent_preview.prepare_callback(FakeComfyModel(self.device), num_inference_steps)
        model_kwargs = dict(input_ids=self.move_to_device(conditioner['input_ids']),
            input_img_latents=input_img_latents,
            input_image_sizes=conditioner['input_image_sizes'],
            attention_mask=self.move_to_device(conditioner["attention_mask"]),
            position_ids=self.move_to_device(conditioner["position_ids"]),
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model,
            )

        show_mem()
        flush_mem()
        show_mem()

        if conditioner['separate_cfg_infer']:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        # Move main model to gpu
        logging.info("- Model to VRAM")
        self.model.to(self.device, dtype=self.dtype)
        show_mem()

        if self.model_cpu_offload:
            for name, param in self.model.named_parameters():
                if 'layers' in name and 'layers.0' not in name:
                    param.data = param.data.cpu()
                else:
                    param.data = param.data.to(self.device)
            for buffer_name, buffer in self.model.named_buffers():
                setattr(self.model, buffer_name, buffer.to(self.device))

        logging.info("- Inference")
        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        self.model.free_mem = free_mem()
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache,
                            callback=callback)
        show_mem(' (Peak)', self.model.free_mem)
        # Separate the last latents, the one with the result
        samples = samples.chunk((1+num_cfg), dim=0)[0]

        show_mem()
        target = model_management.unet_offload_device()
        logging.info(f"- Model to {target}")
        self.model.to(target)
        show_mem()

        samples = samples.to(torch.float32) / VAE_SCALE_FACTOR
        return samples
