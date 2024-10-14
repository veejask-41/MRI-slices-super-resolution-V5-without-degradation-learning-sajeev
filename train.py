import os
import time
import torch
from torch.utils.data import DataLoader
from model.create_model import create_model
from data import create_dataset
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.checkpointing import save_checkpoint, load_checkpoint


def main():
    # Set manual seed for reproducibility
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    # Parse options
    opt = TrainOptions().parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = create_dataset(opt)
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    # Create model
    model = create_model(opt)
    model.setup(opt)
    model = model.to(device)

    # Create visualizer
    visualizer = Visualizer(opt)

    # Optionally resume training
    if opt.continue_train:
        load_checkpoint(model, opt.checkpoint_dir, opt.which_epoch, device)

    # Training Loop
    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            current_batch_size = data["input"].size(
                0
            )  # Assuming 'input' is a key in your data dictionary
            total_iters += current_batch_size
            epoch_iter += current_batch_size

            model.set_input(data)  # Prepare input data
            model.optimize_parameters()  # Forward, backward pass, and optimize

            if total_iters % opt.display_freq == 0:  # Display visuals
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result=True
                )

            if total_iters % opt.print_freq == 0:  # Print losses
                losses = model.get_current_losses()
                t_comp = (time.time() - epoch_start_time) / epoch_iter
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)

            if total_iters % opt.save_latest_freq == 0:  # Save latest model
                print(
                    "Saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_checkpoint(model, opt.checkpoint_dir, "latest", epoch, total_iters)

            if (
                epoch % opt.save_epoch_freq == 0 and i == len(dataloader) - 1
            ):  # Save model at the end of every epoch
                print(
                    "Saving the model at the end of epoch %d, iters %d"
                    % (epoch, total_iters)
                )
                save_checkpoint(
                    model, opt.checkpoint_dir, "epoch_%d" % epoch, epoch, total_iters
                )

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally add code to handle specific exceptions and perform cleanup
