import argparse
from materials_vision.cellpose.training import retrain_cyto


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cyto3_retrained")
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    args = parser.parse_args()
    retrain_cyto(args.model_name)
