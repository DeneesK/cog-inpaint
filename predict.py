# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler  # noqa
from controlnet_aux import HEDdetector
import torch  # noqa
import numpy as np

from PIL import Image  # noqa
from diffusers.utils import load_image  # noqa
from mask_gen import generate_mask


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
        controlnet1 = ControlNetModel.from_pretrained("./control_v11p_sd15_inpaint",
                                                      torch_dtype=torch.float16,
                                                      use_safetensors=True)
        controlnet2 = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16
        )
        controlnet = [controlnet1, controlnet2]
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "./epicrealism_pureevolutionv5-inpainting",
            use_safetensors=True,
            torch_dtype=torch.float16,
            controlnet=controlnet,
            requires_safety_checker=False,
        ).to("cuda")
        self.processor2 = HEDdetector.from_pretrained('lllyasviel/Annotators')
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights('./', weight_name='breastinclassBetter.safetensors')
        self.pipeline.enable_model_cpu_offload()
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt", default='nsfw, nude, breast, pussy, 4k, masterpiece, sexy, seductive'),
        negative_prompt: str = Input(description="input negative_prompt",
                                     default='(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'),
        seed: int = Input(description="input seed",
                          default=0),
        num_inference_steps: int = Input(
            description="input num_inference_steps",
            default=31
            ),
        guidance_scale: int = Input(
            description="input guidance_scale",
            default=7
        ),
        strength: float = Input(
            description="input strength",
            default=0.95
        )
    ) -> Path:
        """Run a single prediction on the model"""
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        try:
            if not seed:
                seed = random.randint(0, 99999)
            init_image = load_image(str(image))
            w, h = resize_(init_image)

            out_path = Path(tempfile.mkdtemp()) / "output.png"

            generate_mask(init_image, str(out_path))

            mask_image = load_image(str(out_path)).resize(init_image.size)

            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            self.pipeline.safety_checker = disabled_safety_checker
            control_image = make_inpaint_condition(init_image, mask_image)
            control_image2 = self.processor2(image, scribble=True)
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
                                  control_image=[control_image,
                                                 control_image2]
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

    if w > h:
        c = h / w
        h = int(1024 * c)
        h = h - (h % 8)
        w = 1024
        return w, h

    c = w / h
    w = int(1024 * c)
    w = w - (w % 8)
    h = 1024
    return w, h
