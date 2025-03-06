import argparse
from materials_vision.cellpose.training import retrain_cyto


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="cyto3_retrained")
    args = parser.parse_args()

    retrain_cyto(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_name=args.model_name
    )
