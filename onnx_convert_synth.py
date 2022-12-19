import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time

from models import build_model


class PredictModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(self.hparams)

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)

@torch.no_grad()
def export(model :pl.LightningModule, width, args):
    left = torch.rand(1, 1, args.height, args.width)
    right = torch.rand(1, 1, args.height, args.width)

    output_name = args.model_name

    torch.onnx.export(
        model,
        (left, right),
        # f"/home/bz/hitnet_out/onnxs/{time.strftime('%Y_%m_%d-%H_%M_%S')}_gather_fixed.onnx",
        f"/home/bz/hitnet_out/onnxs/{output_name}.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['left', 'right']
    )

    # model.to_onnx(
    #     "/home/bz/hitnet_out/onnxs/ours.onnx",
    #     (left, right),
    #     # input_names = ['left', 'right'],
    #     # dynamic_axes={  'left'  : {0 : 'batch_size'},
    #     #                 'right' : {0 : 'batch_size'}}
    # )

    # pred = model(left, right)

    return

if __name__ == "__main__":
    import cv2
    import argparse
    import torchvision
    from pathlib import Path

    from dataset.utils import np2torch
    from colormap import apply_colormap, dxy_colormap

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs=2, required=False)
    parser.add_argument("--model", type=str, default="HITNet")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=180, required=False)
    parser.add_argument("--height", type=int, default=100, required=False)
    parser.add_argument("--output", default="./")
    parser.add_argument("--model_name", default="HitNET_generic")
    args = parser.parse_args()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model.cpu()

    export(model, args.width, args)

