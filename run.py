from pathlib import Path

import torch.utils.data as data

from config import (
    CRITERION,
    EPOCHS,
    FINAL_SUBMISSION,
    LARGE_PATCH_SIZE,
    LEARNING_RATE,
    MODEL_WEIGHTS_LAST_EPOCH,
    NUMBER_PATCH_PER_IMAGE,
    PATCH_SIZE,
    PREDICTIONS_DIR,
    SAVE_MODEL_EVERY_X_EPOCH,
    SUBMISSION_DIR,
    TEST_BATCH_SIZE,
    TEST_DATASET_DIR,
    TEST_IMAGE_SIZE,
    TEST_NUMBER_PATCH_PER_IMAGE,
    TRAIN_BATCH_SIZE,
)
from config import TRAIN_CHECKPOINTS_DIR as CHECKPOINTS_DIR
from config import TRAIN_DATASET_DIR, TRAIN_IMAGE_INITIAL_SIZE
from config import TRAIN_MODEL as MODEL
from datasets import RoadsDatasetTest, RoadsDatasetTrain
from mask_to_submission import masks_to_submission
from predict import predict
from train import train

if __name__ == "__main__":
    model = MODEL
    dataset = RoadsDatasetTrain(
        patch_size=PATCH_SIZE,
        large_patch_size=LARGE_PATCH_SIZE,
        image_initial_size=TRAIN_IMAGE_INITIAL_SIZE,
        number_patch_per_image=NUMBER_PATCH_PER_IMAGE,
        root_dir=TRAIN_DATASET_DIR,
    )
    train_dataloader = data.DataLoader(
        dataset=dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    train(
        model=model,
        dataloader=train_dataloader,
        epochs=EPOCHS,
        criterion=CRITERION,
        checkpoints_dir=CHECKPOINTS_DIR,
    )
    test_dataset = RoadsDatasetTest(
        patch_size=PATCH_SIZE,
        large_patch_size=LARGE_PATCH_SIZE,
        number_patch_per_image=TEST_NUMBER_PATCH_PER_IMAGE,
        image_initial_size=TEST_IMAGE_SIZE,
        root_dir=TEST_DATASET_DIR,
    )
    test_dataloader = data.DataLoader(dataset=dataset, batch_size=TEST_BATCH_SIZE)
    predict(
        model=model,
        dataloader=test_dataloader,
        model_weights=(Path(CHECKPOINTS_DIR) / MODEL_WEIGHTS_LAST_EPOCH),
    )
    predictions = [
        str(x) for x in Path(PREDICTIONS_DIR).glob("**/*.png") if x.is_file()
    ]
    masks_to_submission(str(Path(SUBMISSION_DIR) / FINAL_SUBMISSION), predictions)