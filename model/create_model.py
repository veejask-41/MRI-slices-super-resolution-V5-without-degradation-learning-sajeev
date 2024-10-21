from .SRUNet import SimpleNet
from .degradation_network import DegradationNetwork
from .VGGStylePatchGAN import VGGStylePatchGAN
from .SuperResolutionModel import SuperResolutionModel


def create_model(opt):
    if opt.model_type == "super_resolution_model":
        return SuperResolutionModel(opt)
    elif opt.model_type == "sr_unet":
        return SimpleNet()
    elif opt.model_type == "multi_gdn":
        return DegradationNetwork(image_size=opt.image_size)
    elif opt.model_type == "vgg_patch_gan":
        return VGGStylePatchGAN(patch_size=70)
    else:
        raise ValueError(f"Unknown model type: {opt.model_type}")


# SRUNet(
#     image_size=opt.image_size,
#     in_channels=opt.in_channels,
#     out_channels=opt.out_channels,
#     freeze_encoder=opt.freeze_encoder,
# )
