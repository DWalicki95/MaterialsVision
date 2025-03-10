import argparse
import wandb
from materials_vision.cellpose.training import retrain_cyto


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="cyto3_retrained")
    parser.add_argument(
        "--verbose", action="store_true", help="Turn on detailed logging.")
    args = parser.parse_args()

    if args.verbose:
        print("Verbose turn on.")

    train_losses, test_losses = retrain_cyto(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_name=args.model_name
    )

    wandb.init(
        project='retrain_cellpose',
        name=args.model_name,
        config={
            'train_dir': args.train_dir,
            'test_dir': args.test_dir,
            'model_name': args.model_name
            }
    )
    wandb.log({'train_losses': train_losses, 'test_losses': test_losses})
    wandb.log(
        {
            'train_losses': wandb.plot.line_series(
                xs=list(range(len(train_losses))),
                ys=[train_losses],
                keys=['Train Loss'],
                title='Training Losses',
                xname='Epoch'
            )
        }
    )
