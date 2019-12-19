from models.unet import UNet
from models.resnet import ResNet
import torch.nn as nn

# ----Train Config-----#
TRAIN_MODEL = UNet()
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 1
CRITERION = nn.BCELoss()
EPOCHS = 100
PATCH_SIZE = 16
LARGE_PATCH_SIZE = 96
TRAIN_IMAGE_INITIAL_SIZE = 400
NUMBER_PATCH_PER_IMAGE = int((TRAIN_IMAGE_INITIAL_SIZE / PATCH_SIZE) ** 2)
TRAIN_DATASET_DIR = "./Datasets/training"
TRAIN_CHECKPOINTS_DIR = "./checkpoints"
SAVE_MODEL_EVERY_X_EPOCH = 5
MODEL_WEIGHTS_LAST_EPOCH = "model_final_final"


#----Test Config-----#
TEST_MODEL = UNet()
TEST_BATCH_SIZE = 1
PADDING = 40
PATCH_SIZE = 16
LARGE_PATCH_SIZE = 96
TEST_IMAGE_SIZE = 608
TEST_NUMBER_PATCH_PER_IMAGE = 1444
TEST_MODEL_WEIGHTS = None  # None for now
TEST_DATASET_DIR = "./Datasets/test_set_images"
PREDICTIONS_DIR = "./Predictions"
SUBMISSION_DIR = "./submission"
FINAL_SUBMISSION = "./final_submission.csv"
