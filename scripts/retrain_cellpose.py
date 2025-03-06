import argparse
from materials_vision.cellpose.training import retrain_cyto


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Skrypt do dotrenowania modelu cellpose"
    )
    parser.add_argument("--model_name", type=str, default="cyto3_retrained")
    args = parser.parse_args()
    retrain_cyto(args.model_name)
