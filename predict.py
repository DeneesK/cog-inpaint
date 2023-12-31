# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, AutoencoderKL, StableDiffusionInpaintPipeline  # noqa
from controlnet_aux import OpenposeDetector
import torch  # noqa
import numpy as np

from PIL import Image  # noqa
from diffusers.utils import load_image, make_image_grid  # noqa
from mask_gen import generate_mask, sum_masks


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
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                            torch_dtype=torch.float16)
        controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_inpaint",
                    torch_dtype=torch.float16
            )
        controlnet1 = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
            )
        self.pipeline: StableDiffusionControlNetInpaintPipeline = \
            StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "./epicrealism_pureevolutionv5-inpainting",
                use_safetensors=True,
                torch_dtype=torch.float16,
                requires_safety_checker=False,
                controlnet=[controlnet, controlnet1],
                vae=vae
                ).to("cuda")
        self.processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        self.pipeline.load_lora_weights('./', weight_name='NSFW_Realism_Stable-09.safetensors')
        self.pipeline.load_textual_inversion('./BadDream.pt')
        self.pipeline.enable_model_cpu_offload()
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt", default='nsfw, nude, breast, pussy, 4k, masterpiece, sexy, seductive'),
        negative_prompt: str = Input(description="input negative_prompt",
                                     default='BadDream, (UnrealisticDream:1.2), (bad anatomy), (inaccurate limb:1.2), bad composition, inaccurate eyes, extra digit,fewer digits,(extra arms:1.2), ((((clothes)), lingerie, underwear, brassiere, hair, hairy genitals))'),
        seed: int = Input(description="input seed",
                          default=0),
        num_inference_steps: int = Input(
            description="input num_inference_steps",
            default=31
            ),
        guidance_scale: int = Input(
            description="input guidance_scale",
            default=5
        ),
        strength: float = Input(
            description="input strength",
            default=0.95
        ),
        lora_scale: float = Input(
            description="lora scale FROM 0 to 1, step 0.1",
            default=1.0
        ),
        mask: str = Input(
            description="""
            !STRICT -> EXAMPLE: 1,2 or 4 ...
            # 0 - background
            # 1 - hair
            # 2 - body - skin
            # 3 - face - skin
            # 4 - clothes
            # 5 - others(accessories)
            """,
            default='0'
            ),
        mask_strength: float = Input(
            description="---origin 0.25---",
            default=0.9
        ),
        face_inver: float = Input(
            description="face mask inv",
            default=0.07
        )
    ) -> Path:
        """Run a single prediction on the model"""
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        try:
            if not seed:
                seed = random.randint(0, 99999)
            init_image = load_image(str(image))
            w, h = resize_(init_image)
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            init_image = init_image.resize((w, h))
            out_path = Path(tempfile.mkdtemp()) / "output.png"

            generate_mask(image=str(image),
                          path=str(out_path),
                          mask_index=mask,
                          strength=mask_strength,
                          inv=True)
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            mask_image = load_image(str(out_path)).resize((w, h))
            sum_masks(str(image), out_path, face_inver)
            new_mask = load_image(str(out_path)).resize((w, h))
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            control_image = make_inpaint_condition(init_image, new_mask)
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            self.pipeline.safety_checker = disabled_safety_checker
            control_image2 = self.processor(
                init_image,
                hand_and_face=True
                )
            image: Image = self.pipeline(prompt=prompt,
                                         negative_prompt=negative_prompt,
                                         image=init_image,
                                         mask_image=new_mask,
                                         control_image=[control_image,
                                                        control_image2],
                                         generator=generator,
                                         eta=1.0,
                                         num_inference_steps=int(num_inference_steps),  # noqa
                                         guidance_scale=int(guidance_scale),
                                         strength=strength,
                                         width=w,
                                         height=h,
                                         cross_attention_kwargs={
                                            "scale": float(lora_scale)
                                            }
                                         ).images[0]
            image = make_image_grid([
                                     image, mask_image,
                                     new_mask,
                                     control_image2
                                     ],
                                    rows=1, cols=4)
            image.save(out_path)
            print(image)
            return out_path
        except Exception as ex:
            print(ex)
            return str(ex)


def resize_(image) -> tuple[int, int]:
    w = image.width
    h = image.height

    if w < 1025 and h < 1025:
        return w, h

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
