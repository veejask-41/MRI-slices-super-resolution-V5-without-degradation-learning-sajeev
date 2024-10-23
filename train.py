import os
import time
import torch
from torch.utils.data import DataLoader
from model.create_model import create_model

# from data import create_dataset
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
from utils.checkpointing import save_checkpoint, load_checkpoint

from data.dataloader import MRIDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Parse options
    opt = TrainOptions().parse()
    opt.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Create a model based on the options
    model = create_model(opt)

    # Creating dataset instances
    train_dataset = MRIDataset("./dataset/train_filenames.txt")
    val_dataset = MRIDataset("./dataset/val_filenames.txt")

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    # dataset = create_dataset(opt)
    # dataloader = DataLoader(
    #     dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers
    # )
    dataset_size = len(train_dataset)
    print(f"The number of training images = {dataset_size}")

    # Create visualizer
    visualizer = Visualizer(opt)

    # Optionally resume training
    if opt.continue_train:
        load_checkpoint(model, opt.checkpoint_dir, opt.which_epoch, str(device))
        print(f"Loading checkpoint on device: {device}")

    # Training loop
    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader, 0):
            epoch_iter += 1
            low_res_images, high_res_images = data[0][i], data[1][i]

            if high_res_images.shape[2:] != low_res_images.shape[2:]:
                print(
                    f"Mismatched shapes in batch {i}: HR shape {high_res_images.shape}, LR shape {low_res_images.shape}"
                )
                continue

            current_batch_size = len(data[0])
            total_iters += current_batch_size

            mri_vol = {"LR": low_res_images, "HR": high_res_images}

            model.set_input(mri_vol)  # Prepare input data by slicing the MRI volume

            # Process each slice in the current volume
            num_slices = len(model.lr_slices)
            for slice_index in range(num_slices):
                lr_slice, hr_slice = model.get_slice_pair(slice_index)

                angle = 45  # degrees
                translation = (10, 5)  # x and y translation in pixels

                # Forward, backward pass, and optimize with additional parameters
                model.optimize_parameters(
                    lr_images=lr_slice,
                    hr_images=hr_slice,
                    lambda_tv=1.0,
                    angle=angle,
                    translation=translation,
                )

                # Print loss information at the specified frequency
                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - epoch_start_time) / epoch_iter
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)

            # Save the latest model at the specified frequency
            if total_iters % opt.save_latest_freq == 0:
                print(
                    "Saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                # save_checkpoint(
                #     model, opt.checkpoint_dir, "latest", epoch, total_iters
                # )
                model.save_checkpoint(
                    opt.checkpoint_dir,
                    ["sr epoch_%d" % epoch, "vgg_patchgan epoch_%d" % epoch],
                    epoch,
                    total_iters,
                )

            # Display visuals at the specified frequency of the slices of a certain MRI Volume
            # if total_iters % opt.display_freq == 0:
            # model.save_volume(epoch=epoch)

            # Save the model at the end of every epoch
            if epoch % opt.save_epoch_freq == 0 and i == len(train_loader) - 1:
                print(
                    "Saving the model at the end of epoch %d, iters %d"
                    % (epoch, total_iters)
                )
                # save_checkpoint(
                #     model, opt.checkpoint_dir, "epoch_%d" % epoch, epoch, total_iters
                # )
                model.save_checkpoint(
                    opt.checkpoint_dir,
                    ["sr epoch_%d" % epoch, "vgg_patchgan epoch_%d" % epoch],
                    epoch,
                    total_iters,
                )

            print(
                "End of epoch %d / %d \t Time Taken: %d sec"
                % (
                    epoch,
                    opt.n_epochs + opt.n_epochs_decay,
                    time.time() - epoch_start_time,
                )
            )

    model.save_final_models()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally add code to handle specific exceptions and perform cleanup
