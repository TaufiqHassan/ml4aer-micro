import argparse
import torch.nn as nn
from train_model import ModelTrainer

def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train a model using ModelTrainer")

    # Required positional argument for data path
    parser.add_argument("data_path", type=str, help="Path to the data directory")

    # Optional arguments
    parser.add_argument("--model_id", type=str, default="3_128_0_50", help="model_id to load")
    parser.add_argument("--action", type=str, default="train", help="'train' or 'eval'")
    parser.add_argument("--natmos", type=int, default=20, help="Number of atmospheric variables")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--depths", nargs='+', type=int, default=[2], help="List of model depths")
    parser.add_argument("--widths", nargs='+', type=int, default=[128], help="List of model widths")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--n_models", type=int, default=1, help="Number of models to train")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--loss_func", type=str, default="MSE", help="Loss function to use (default: MSE)")
    parser.add_argument("--varlist", nargs='+', default=['bc_a1', 'bc_a4', 'pom_a1', 'pom_a4',
                                                         'so4_a1', 'so4_a2', 'so4_a3', 'mom_a1', 'mom_a2', 'mom_a4',
                                                         'ncl_a1', 'ncl_a2', 'soa_a1', 'soa_a2', 'soa_a3',
                                                         'num_a1', 'num_a2', 'num_a4', 'H2SO4', 'SOAG'],
                        help="List of variables")
    parser.add_argument("--transformation_type", type=str, default="standardization", help="Type of transformation (default: standardization)")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the loss function argument to the corresponding PyTorch loss function
    loss_func_map = {
        "Huber": nn.SmoothL1Loss(),
        "MSE": nn.MSELoss(),
    }
    loss_func = loss_func_map.get(args.loss_func, nn.MSELoss())
    print('\nLoss Function: ',args.loss_func)

    # Create the ModelTrainer instance with parsed arguments
    trainer = ModelTrainer(
        data_path=args.data_path,
        natmos=args.natmos,
        batch_size=args.batch_size,
        depths=args.depths,
        widths=args.widths,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_models=args.n_models,
        n_epochs=args.n_epochs,
        loss_func=loss_func,
        varlist=args.varlist,
        transformation_type=args.transformation_type,
        model_id=args.model_id
    )

    # Start the training and evaluation process
    if args.action=="train":
        trainer._train_and_eval()
    elif args.action=="eval":
        trainer._evalonly()

if __name__ == "__main__":
    main()
