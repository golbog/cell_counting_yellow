# TODO: check imports and function definitions to exclude the ones that are no longer relevant

import argparse
from pathlib import Path

import cv2
import keras.preprocessing
import numpy as np
from tqdm import tqdm

import evaluation_utils

parser = argparse.ArgumentParser(description='Run evaluation pipeline for specified model name')
parser.add_argument('threshold', metavar='threshold', type=float, help='Threshold for filtering the predicted images')
parser.add_argument('--model_name', metavar='name', type=str,
                    help='Name of the model to evaluate.', default="model/c-ResUnet")
parser.add_argument('--out_folder', metavar='out_folder', type=str, default="results",
                    help='Output folder')
parser.add_argument('--input_folder', metavar='input_folder', type=str, default="data",
                    help='Output folder')
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                    help='Batch size for generator used for predictions')
args = parser.parse_args()

root_path = Path("./")
IMG_WIDTH = 1600  # 1400
IMG_HEIGHT = 1200  # 1040

if __name__ == "__main__":
    model_name = "{}.h5".format(args.model_name)
    save_path = root_path / args.out_folder
    input_path = root_path / args.input_folder
    threshold = args.threshold
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading images from: {input_path}")
    print(f"Output folder set to: {save_path}")
    print(f"Threshold set to: {threshold}")

    WeightedLoss = evaluation_utils.create_weighted_binary_crossentropy(1, 1.5)
    model = keras.models.load_model(model_name, custom_objects={'mean_iou': evaluation_utils.mean_iou,
                                                                'dice_coef': evaluation_utils.dice_coef,
                                                                'weighted_binary_crossentropy': WeightedLoss})

    # predict with generator

    image_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    image_generator = image_datagen.flow_from_directory(input_path,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=args.batch_size,
                                                        color_mode="rgb", class_mode=None, shuffle=False)
    filenames = image_generator.filenames
    predicts = model.predict_generator(image_generator, steps=np.ceil(len(filenames) / args.batch_size))

    for full_path, pred_im in tqdm(zip(filenames, predicts), total=len(filenames)):
        # TODO: put in separate function
        filename = Path(full_path).stem
        path = save_path / Path(full_path).parent
        path = path.parent / (path.name + f"_threshold={threshold}")
        path.mkdir(parents=True, exist_ok=True)

        pred_mask = evaluation_utils.predict_mask_from_map(pred_im, threshold)

        orig_im = cv2.resize(cv2.imread(str(input_path / full_path), cv2.IMREAD_COLOR), (IMG_WIDTH, IMG_HEIGHT))
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours)
        joined_im = cv2.drawContours(orig_im, contours, -1, (0, 255, 0), 3)

        filename_base = path / filename
        cv2.imwrite(f"{filename_base}_pred.png", joined_im)
        with open(f"{filename_base}_count.txt", "w") as f:
            f.write(str(num_contours))
