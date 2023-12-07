# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler  # noqa
import torch  # noqa
import numpy as np

from PIL import Image  # noqa
from diffusers.utils import load_image  # noqa


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make
        running multiple predictions efficient"""
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # controlnet = ControlNetModel.from_pretrained("./control_v11p_sd15_inpaint",
        #                                              torch_dtype=torch.float16,
        #                                              use_safetensors=True)
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(
            "./epiCRealism/epicrealism_v10-inpainting.safetensors",
            use_safetensors=True,
            torch_dtype=torch.float16,
            # controlnet=controlnet,
            requires_safety_checker=False,
        ).to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_model_cpu_offload()
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def predict(
        self,
        image: Path = Input(description="input image"),
        mask: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt"),
        negative_prompt: str = Input(description="input negative_prompt",
                                     default=''),
        seed: int = Input(description="input seed",
                          default=0),
        num_inference_steps: int = Input(
            description="input num_inference_steps",
            default=20
            ),
        guidance_scale: int = Input(
            description="input guidance_scale",
            default=7
        ),
        strength: float = Input(
            description="input strength",
            default=0.8
        ),
        # width: int = Input(description="input width",
        #                    default=1024),
        # height: int = Input(description="input height",
        #                     default=1024),
        controlnet: bool = Input(description="input bool",
                                 default=False),
    ) -> Path:
        """Run a single prediction on the model"""
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        try:
            if controlnet:
                pass

            if not seed:
                seed = random.randint(0, 99999)
            init_image = load_image(str(image))
            w, h = resize_(init_image)
            init_image.resize((w, h))
            mask_image = load_image(str(mask)).resize((w, h))
            print(prompt)
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            self.pipeline.safety_checker = disabled_safety_checker
            control_image = make_inpaint_condition(init_image, mask_image)
            image = self.pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=init_image,
                                  mask_image=mask_image,
                                  eta=1.0,
                                  generator=generator,
                                  num_inference_steps=int(num_inference_steps),
                                  guidance_scale=int(guidance_scale),
                                  strength=strength,
                                  width=w,
                                  height=h,
                                  control_image=control_image,
                                  ).images[0]
            print(image)
            image.save(out_path)
            return out_path
        except Exception as ex:
            print(ex)
            return str(ex)


def resize_(immage) -> tuple[int, int]:
    w = immage.width
    h = immage.height

    if h < 1024 and w < 1024:
        if h % 8 == 0 and w % 8 == 0:
            return w, h
        w = w - (w % 8)
        h = h - (h % 8)
        return w, h

    while True:
        if h < 1024 and w < 1024:
            if h % 8 == 0 and w % 8 == 0:
                return w, h
            w = w - (w % 8)
            h = h - (h % 8)
            return w, h
        h = int(h / 2)
        w = int(w / 2)
