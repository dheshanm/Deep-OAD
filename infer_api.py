import warnings

# warnings.simplefilter(action="ignore", category=FutureWarning)
# warnings.simplefilter(action="ignore", category=DeprecationWarning)
# warnings.simplefilter(action="ignore", category=UserWarning)

import logging
import os
import sys

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow messages
# logging.getLogger("tensorflow").setLevel(logging.FATAL)  # suppress TensorFlow messages

# # Keras outputs warnings using `print` to stderr so let's direct that to devnull temporarily
# stderr = sys.stderr
# sys.stderr = open(os.devnull, "w")

import argparse

from loguru import logger
from models import load_vit_model
from processing import preprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# sys.stderr = stderr


class Inference:
    def __init__(self):
        logger.info("Loading Models")
        self.vit_model = load_vit_model()

    def predict(self, image_path, model_name="vit"):
        X = preprocess(model_name, image_path)

        y = self.vit_model.predict(X)[0][0]

        logger.info(f"Predicted angle is: {y} degree")
        return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()

    args.model_name = "vit"

    if args.image_path is None:
        print("Usage: python infer_api.py --image-path <image_path>")
        sys.exit(1)

    model = Inference()
    angle = model.predict(args.image_path)

    print(f"Predicted angle is: {angle} degree")
