import logging
import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import pprint
import time

from PIL import Image
import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from safetensors.torch import load_file

from ..OmniGen import OmniGen, OmniGenProcessor, OmniGenScheduler
from ..OmniGen.utils import show_mem, show_shape, VAE_SCALE_FACTOR, flush_mem

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model
        ... )
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""


class OmniGenPipeline:
    def __init__(
        self,
        model: OmniGen,
        processor: OmniGenProcessor,
        device: Union[str, torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                logging.info("Don't detect any available GPUs, using CPU instead, this may take long time to generate image!!!")
                self.device = torch.device("cpu")

        if self.model:
            self.model.to(self.dtype)
            self.model.eval()

        self.model_cpu_offload = False

    @classmethod
    def from_pretrained(cls, model_name, Quantization: bool=False):
        if not os.path.exists(model_name) or (not os.path.exists(os.path.join(model_name, 'model.safetensors')) and model_name == "Shitao/OmniGen-v1"):
            logging.info("Model not found, downloading...")
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5', 'model.pt'])
            logging.info(f"Downloaded model to {model_name}")
        logging.info(f"Loading OmniGen Model")
        model = OmniGen.from_pretrained(model_name, quantize=Quantization)

        logging.info(f"Loading OmniGen Processor")
        processor = OmniGenProcessor.from_pretrained(model_name)

        return cls(model, processor)
    
    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.merge_and_unload()

        self.model = model
    
    def to(self, device: Union[str, torch.device]):
        if isinstance(device, str):
            device = torch.device(device)
        self.model.to(device)
        self.device = device

    def vae_encode(self, vae, img):
        """ Encode the image and move it to the device and data type used by the model """
        # Note: the result is in the CPU and using FP32, we move it back to the GPU to be used by the model
        return vae.encode(img).mul_(VAE_SCALE_FACTOR).to(self.device, dtype=self.dtype)
    
    def move_to_device(self, data):
        if isinstance(data, list):
            return [x.to(self.device) for x in data]
        return data.to(self.device)

    def enable_model_cpu_offload(self):
        self.model_cpu_offload = True
        self.model.to("cpu")
    
    def disable_model_cpu_offload(self):
        self.model_cpu_offload = False
        if self.model:
            self.model.to(self.device)
            flush_mem()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        offload_model: bool = False,
        use_kv_cache: bool = True,
        offload_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = None,
        Quantization: bool = False,
        move_to_ram: bool = False,
        vae = None,
        ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation. 
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation, but in the negative sense.
            input_images (`List[str]` or `List[List[str]]`, *optional*):
                The list of input images. We will replace the "<|image_i|>" in prompt with the 1-th image in list.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image. The number must be a multiple of 16.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image. The number must be a multiple of 16.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            use_img_guidance (`bool`, *optional*, defaults to True):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            img_guidance_scale (`float`, *optional*, defaults to 1.6):
                Defined as equation 3 in [Instrucpix2pix](https://arxiv.org/pdf/2211.09800). 
            max_input_image_size (`int`, *optional*, defaults to 1024): the maximum size of input image, which will be used to crop the input image to the maximum size
            separate_cfg_infer (`bool`, *optional*, defaults to False):
                Perform inference on images with different guidance separately; this can save memory when generating images of large size at the expense of slower inference.
            use_kv_cache (`bool`, *optional*, defaults to True): enable kv cache to speed up the inference
            offload_kv_cache (`bool`, *optional*, defaults to True): offload the cached key and value to cpu, which can save memory but slow down the generation silightly
            offload_model (`bool`, *optional*, defaults to False): offload the model to cpu, which can save memory but slow down the generation
            use_input_image_size_as_output (bool, defaults to False): whether to use the input image size as the output image size, which can be used for single-image input, e.g., image editing task
            seed (`int`, *optional*):
                A random seed for generating output. 
            dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                data type for the model
            move_to_ram (`bool`, *optional*, defaults to False):
                Keep in VRAM only the needed models, otherwise move them to RAM.
                Use it if you see allocation problems.
            vae: Comfy_UI VAE object
        Examples:

        Returns:
            A list with the generated images.
        """
        logging.info("Starting OmniGen pipeline")
        show_mem()
        # check inputs:
        if use_input_image_size_as_output:
            assert isinstance(prompt, str) and len(input_images) == 1, "if you want to make sure the output image have the same size as the input image, please only input one image instead of multiple input images"
        else:
            assert height%16 == 0 and width%16 == 0, "The height and width must be a multiple of 16."
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None
        
        # set model and processor
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size)
        if offload_model:
            self.enable_model_cpu_offload()
        else:
            self.disable_model_cpu_offload()

        logging.info("- Input data images")
        show_mem()
        input_data = self.processor(prompt, input_images, height=height, width=width, use_img_cfg=use_img_guidance, separate_cfg_input=separate_cfg_infer,
                                    use_input_image_size_as_output=use_input_image_size_as_output, negative_prompt=negative_prompt)

        logging.debug('Processor output:')
        logging.debug(f'input_ids: {show_shape(input_data["input_ids"])}')
        # logging.debug(f'attention_mask {input_data["attention_mask"]}')
        # logging.debug(f'position_ids: {input_data["position_ids"]}')
        logging.debug(f'input_pixel_values: {show_shape(input_data["input_pixel_values"])}')
        logging.debug(f'input_image_sizes: {show_shape(input_data["input_image_sizes"])}')
        # logging.debug(f'padding_images: {input_data["padding_images"]}')
        logging.debug('---------------------------------------------------')
        logging.debug(pprint.pformat(input_data))
        logging.info('---------------------------------------------------')

        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1
        if use_input_image_size_as_output:
            if separate_cfg_infer:
                height, width = input_data['input_pixel_values'][0][0].shape[-2:]
            else:
                height, width = input_data['input_pixel_values'][0].shape[-2:]
        latent_size_h, latent_size_w = height//8, width//8

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        logging.info("- Create latents")
        show_mem()
        latents = torch.randn(num_prompt, 4, latent_size_h, latent_size_w, device=self.device, generator=generator)
        latents = torch.cat([latents]*(1+num_cfg), 0).to(dtype)

        input_img_latents = []
        if separate_cfg_infer:
            logging.info("- Encoding images separately")
            for temp_pixel_values in input_data['input_pixel_values']:
                logging.info("  - One conditional")
                temp_input_latents = []
                for img in temp_pixel_values:
                    logging.info(show_shape(img))
                    temp_input_latents.append(self.vae_encode(vae, img))
                input_img_latents.append(temp_input_latents)
        else:
            logging.info("- Encoding all images at once")
            for img in input_data['input_pixel_values']:
                input_img_latents.append(self.vae_encode(vae, img))
        # Stop here if we are skipping the model load
        assert self.model is not None, "Stopped because we didn't load the model"

        model_kwargs = dict(input_ids=self.move_to_device(input_data['input_ids']), 
            input_img_latents=input_img_latents, 
            input_image_sizes=input_data['input_image_sizes'], 
            attention_mask=self.move_to_device(input_data["attention_mask"]), 
            position_ids=self.move_to_device(input_data["position_ids"]), 
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache,
            offload_model=offload_model,
            )

        show_mem()
        flush_mem()
        show_mem()
        
        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        # Move main model to gpu
        logging.info("- Model to VRAM")
        self.model.to(self.device, dtype=dtype)
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
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache, offload_kv_cache=offload_kv_cache)
        samples = samples.chunk((1+num_cfg), dim=0)[0]

        show_mem()
        if move_to_ram or self.model_cpu_offload:
            logging.info("- Model to CPU")
            self.model.to('cpu')
        show_mem()

        samples = samples.to(torch.float32)
        samples = samples / 0.13025
        return samples
