# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
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
        # self.model = torch.load("./weights.pth")
        self.pipeline = AutoPipelineForInpainting.from_pretrained(
            "./stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        self.pipeline.enable_model_cpu_offload()

    def predict(
        self,
        image: Path = Input(description="input image"),
        mask: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt")
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            out_path = Path(tempfile.mkdtemp()) / "output.png"
            # remove following line if xFormers is not installed or
            # you have PyTorch 2.0 or higher installed
            # pipeline.enable_xformers_memory_efficient_attention()

            # load base and mask image
            init_image = load_image(str(image))
            mask_image = load_image(str(mask))

            generator = torch.Generator("cuda").manual_seed(92)
            prompt = str(prompt)
            image = self.pipeline(prompt=prompt,
                                  image=init_image,
                                  mask_image=mask_image,
                                  generator=generator).images[0]
            image.save(out_path)
            return out_path
        except Exception as ex:
            print(ex)
            return str(ex)
