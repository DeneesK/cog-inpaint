from functools import reduce
import cv2
import os
from PIL import Image
import wget
import numpy as np
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MASK_OPTION_0_BACKGROUND = 'background'
MASK_OPTION_1_HAIR = 'hair'
MASK_OPTION_2_BODY = 'body (skin)'
MASK_OPTION_3_FACE = 'face (skin)'
MASK_OPTION_4_CLOTHES = 'clothes'


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


def generate_mask(image: Image, path: str, mask_index: str = '4'):
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

            if not mask_index:
                mask_index = [4]
            else:
                mask_index = mask_index.split(',')
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

            mask_background_array = np.zeros(image_shape, dtype=np.uint8)
            mask_background_array[:] = (0, 0, 0, 255)

            mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
            mask_foreground_array[:] = (255, 255, 255, 255)

            mask_arrays = []
            for i, mask in enumerate(masks):
                condition = np.stack(
                    (mask.numpy_view(),) * image_shape[-1], axis=-1
                    ) > 0.25
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


if __name__ == '__main__':
    print('Starting')
    image = generate_mask(Image.open('test.jpg'), 'result.png')
