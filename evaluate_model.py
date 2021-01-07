# TODO: check imports and function definitions to exclude the ones that are no longer relevant

import argparse

parser = argparse.ArgumentParser(description='Run evaluation pipeline for specified model name')
parser.add_argument('model_name', metavar='name', type=str, default="ResUnet",  # nargs='+',
                    help='Name of the model to evaluate.')
parser.add_argument('--out_folder', metavar='folder', type=str, default="results",
                    help='Output folder')
parser.add_argument('--mode', metavar='mode', type=str, default="eval",
                    help="""Running mode. Valid values:
                            - eval (default) --> optimise threshold (train_val folder, full size images)                            
                            - test --> validate on test images (test folder, full size images)                     
                            - test_code --> for testing changes in the code
                            """)
args = parser.parse_args()

from pathlib import Path

# setup paths --> NOTE: CURRENT PATHS ARE TO BE UPDATED
repo_path = Path("/storage/gpfs_maestro/hpc/user/rmorellihpc/cell_counting_yellow")
repo_path = Path("/home/luca/PycharmProjects/cell_counting_yellow")
if args.mode == "eval":
    IMG_PATH = repo_path / "DATASET/train_val/full_size/images"
    MASKS_PATH = repo_path / "DATASET/train_val/full_size/masks"
elif args.mode == "test":
    IMG_PATH = repo_path / "DATASET/test/all_images/images"
    MASKS_PATH = repo_path / "DATASET/test/all_masks/masks"
else:
    IMG_PATH = repo_path / "DATASET/test_tr_opt/sample_valid/images"
    MASKS_PATH = repo_path / "DATASET/test_tr_opt/sample_valid/masks"

if __name__ == "__main__":

    from skimage.segmentation import watershed
    from math import hypot
    import pandas as pd
    import numpy as np

    from tqdm import tqdm
    import cv2
    # import matplotlib
    # matplotlib.use('Agg')

    from keras.models import load_model
    from evaluation_utils import *

    model_name = "{}.h5".format(args.model_name)
    model_path = "{}/model_results/{}".format(repo_path, model_name)
    save_path = repo_path / args.out_folder / args.mode
    save_path.mkdir(parents=True, exist_ok=True)
    save_path.chmod(16886)  # chmod 776
    text = "\nReading images from: {}".format(str(IMG_PATH))
    print("#" * len(text))
    print(text)
    print("Output folder set to: {}\n".format(str(save_path)))

    print("#" * len(text))
    print(f"\nModel: {model_name}\n\n")
    WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)
    model = load_model(model_path, custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef,
                                                   'weighted_binary_crossentropy': WeightedLoss})  # , compile=False)

    # predict with generator
    from keras.preprocessing.image import ImageDataGenerator
    image_datagen = ImageDataGenerator(rescale=1. / 255)

    IMG_HEIGHT = 1200
    IMG_WIDTH = 1600
    BATCH_SIZE = 3
    image_generator = image_datagen.flow_from_directory(IMG_PATH.parent,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
                                                        color_mode="rgb", class_mode=None, shuffle=False)
    filenames = image_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict_generator(image_generator, steps=np.ceil(nb_samples / BATCH_SIZE))
    threshold_seq = np.arange(start=0.5, stop=1, step=0.1)
    metrics_df_validation_rgb = pd.DataFrame(None, columns=["F1", "MAE", "MedAE", "MPE", "accuracy",
                                                            "precision", "recall"])

    for _, threshold in tqdm(enumerate(threshold_seq), total=len(threshold_seq)):

        print(f"Running for threshold: {threshold:.3f}")
        # create dataframes for storing performance measures
        validation_metrics_rgb = pd.DataFrame(
            columns=["TP", "FP", "FN", "Target_count", "Predicted_count"])
        # loop on masks
        for idx, img_path in enumerate(filenames):
            mask_path = MASKS_PATH / img_path.split("/")[1]
            pred_mask_rgb = predict_mask_from_map(predict[idx], threshold)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            compute_metrics(pred_mask_rgb, mask,
                            validation_metrics_rgb, img_path.split("/")[1])
        metrics_df_validation_rgb.loc[threshold] = F1Score(validation_metrics_rgb)
    outname = save_path / 'metrics_{}.csv'.format(model_name[:-3])
    metrics_df_validation_rgb.to_csv(outname, index=True, index_label='Threshold')
    _ = plot_thresh_opt(metrics_df_validation_rgb, model_name, save_path)

