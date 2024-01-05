from functools import reduce
import cv2
import os
from PIL import Image
import wget
import numpy as np
import mediapipe as mp

import numpy as np
from PIL import Image


BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MASK_OPTION_0_BACKGROUND = 'background'
MASK_OPTION_1_HAIR = 'hair'
MASK_OPTION_2_BODY = 'body (skin)'
MASK_OPTION_3_FACE = 'face (skin)'
MASK_OPTION_4_CLOTHES = 'clothes'


def watermark_with_transparency(input_image_path,
                                watermark_image_path,
                                position=(0, 0)):
    base_image = Image.open(input_image_path).convert('RGBA')
    watermark = Image.open(watermark_image_path).convert('RGBA')
    width, height = base_image.size

    transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    transparent.paste(base_image, (0, 0))
    transparent.paste(watermark, position, mask=watermark)
    transparent.save(input_image_path)


def removeBackground(file, savePath):

    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    mask = 255 - mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    cv2.imwrite(savePath, result)


def get_mediapipe_image(image: Image) -> mp.Image:
    # Convert gr.Image to NumPy array
    numpy_image = np.asarray(image)

    image_format = mp.ImageFormat.SRGB

    # Convert BGR to RGB (if necessary)
    print(numpy_image.shape)
    if numpy_image.shape[-1] == 4:
        image_format = mp.ImageFormat.SRGBA
    elif numpy_image.shape[-1] == 3:
        image_format = mp.ImageFormat.SRGB
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

    return mp.Image(image_format=image_format, data=numpy_image)


def generate_mask(image: Image, path: str, mask_index: str = '', strength: float = 0.15, inv = False):  # noqa
    if image is not None:
        image = Image.open(image)
        model_folder_path = os.path.join('models', 'mediapipe')
        os.makedirs(model_folder_path, exist_ok=True)

        model_path = os.path.join(model_folder_path,
                                  'selfie_multiclass_256x256.tflite')
        model_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'  # noqa
        if not os.path.exists(model_path):
            print("Downloading 'selfie_multiclass_256x256.tflite' model")
            wget.download(model_url, model_path)

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True
            )
        # Create the image segmenter
        with ImageSegmenter.create_from_options(options) as segmenter:

            # Retrieve the masks for the segmented image
            media_pipe_image = get_mediapipe_image(image=image)
            segmented_masks = segmenter.segment(media_pipe_image)

            masks = []
            if mask_index == '':
                mask_index = [2, 4]
            else:
                mask_index = [int(i) for i in mask_index.split(',')]
            # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
            # 0 - background
            # 1 - hair
            # 2 - body - skin
            # 3 - face - skin
            # 4 - clothes
            # 5 - others(accessories)
            for i in mask_index:
                masks.append(segmented_masks.confidence_masks[int(i)])

            image_data = media_pipe_image.numpy_view()
            image_shape = image_data.shape

            # convert the image shape from "rgb"
            # to "rgba" aka add the alpha channel
            if image_shape[-1] == 3:
                image_shape = (image_shape[0], image_shape[1], 4)
            if inv:
                mask_background_array = np.zeros(image_shape, dtype=np.uint8)
                mask_background_array[:] = (255, 255, 255, 255)

                mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
                mask_foreground_array[:] = (0, 0, 0, 255)
            else:
                mask_background_array = np.zeros(image_shape, dtype=np.uint8)
                mask_background_array[:] = (0, 0, 0, 255)

                mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
                mask_foreground_array[:] = (255, 255, 255, 255)

            mask_arrays = []
            for i, mask in enumerate(masks):
                condition = np.stack(
                    (mask.numpy_view(),) * image_shape[-1], axis=-1
                    ) > float(strength)  # default: 0.25
                mask_array = np.where(condition,
                                      mask_foreground_array,
                                      mask_background_array)
                mask_arrays.append(mask_array)

            # Merge our masks taking the maximum from each
            merged_mask_arrays = reduce(np.maximum, mask_arrays)

            # Create the image
            mask_image = Image.fromarray(merged_mask_arrays)
            print(mask_image)
            mask_image.save(path)
    else:
        return None


def sum_masks(origin: str):
    generate_mask(origin, 'hair.png', mask_index='1', inv=True, strength=0.07)
    generate_mask(origin, 'face.png', mask_index='3', inv=True, strength=0.07)
    removeBackground('face.png', 'r1.png')
    removeBackground('hair.png', 'r2.png')
    watermark_with_transparency(origin, 'r1.png')
    watermark_with_transparency(origin, 'r2.png')


if __name__ == '__main__':
    generate_mask('test.jpg', 'result.png', inv=False, strength=0.04)
    generate_mask('test.jpg', 'hair.png', mask_index='1', inv=True, strength=0.07)
    generate_mask('test.jpg', 'face.png', mask_index='3', inv=True, strength=0.07)
    removeBackground('face.png', 'r1.png')
    removeBackground('hair.png', 'r2.png')
    watermark_with_transparency('result.png', 'r1.png')
    watermark_with_transparency('result.png', 'r2.png')
