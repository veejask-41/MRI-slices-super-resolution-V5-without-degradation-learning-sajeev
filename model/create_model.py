from Generator import SRUNet
from Discriminator import MultiGDNModel


def create_model(opt):
    if opt.model_type == "sr_unet":
        # Create a Super-Resolution U-Net model
        return SRUNet(
            image_size=opt.image_size,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
            freeze_encoder=opt.freeze_encoder,
        )
    elif opt.model_type == "gdn":
        # Create a Multi-GDN Model with custom parameters for each GDN
        # opt.gdn_params = [{'size': 9, 'sigma': 1.0, 'sinc_scale': 0.3}, {'size': 11, 'sigma': 1.5, 'sinc_scale': 0.5}]
        return MultiGDNModel(
            gdn_params=opt.gdn_params,
            in_channels=opt.in_channels,
            out_channels=opt.out_channels,
        )
    else:
        raise ValueError(f"Model type '{opt.model_type}' not recognized.")
