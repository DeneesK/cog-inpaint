# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import AutoPipelineForInpainting  # noqa
import torch  # noqa

from PIL import Image  # noqa
from diffusers.utils import load_image  # noqa


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make
        running multiple predictions efficient"""
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            "./epicrealism_pureevolutionv5-inpainting",
            requires_safety_checker=False,
            use_safetensors=True
        ).to("cuda")
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
            image = self.pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=init_image,
                                  mask_image=mask_image,
                                  generator=generator,
                                  num_inference_steps=int(num_inference_steps),
                                  guidance_scale=int(guidance_scale),
                                  strength=strength).images[0]
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
