"""One experiment registry shared by training, evaluation and notebooks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    architecture: str
    channels: int
    loss: str
    best_checkpoint: str
    last_checkpoint: str


EXPERIMENTS = {
    "pawl": Experiment("unet", 10, "auto", "best_unet.pth", "last_unet.pth"),
    "swin": Experiment("swin", 10, "auto", "best_model.pth", "last_model.pth"),
    "unet-2ch-l1": Experiment("unet", 2, "l1", "best_2ch_l1.pth", "last_2ch_l1.pth"),
    "unet-2ch-auto": Experiment("unet", 2, "auto", "best_2ch_auto.pth", "last_2ch_auto.pth"),
    "unet-10ch-l1": Experiment("unet", 10, "l1", "best_10ch_l1.pth", "last_10ch_l1.pth"),
}


def build_model(experiment, config, *, pretrained=False):
    """Keep optional timm imports out of the main PAWL-Net path."""
    if experiment.architecture == "unet":
        from pawl_net.models.unet import StandardUNet

        return StandardUNet(in_channels=experiment.channels)
    from pawl_net.models.swin import PhysicsSwinUNet

    return PhysicsSwinUNet(
        in_channels=experiment.channels,
        img_size=config["model"]["img_size"],
        pretrained=pretrained,
    )


def select_channels(inputs, experiment):
    return inputs[:, [0, 9]] if experiment.channels == 2 else inputs
