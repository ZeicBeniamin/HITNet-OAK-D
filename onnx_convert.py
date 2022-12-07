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
def predict(model, lp, rp, width, op):
    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
    if width is not None and width != left.shape[1]:
        height = int(round(width / left.shape[1] * left.shape[0]))
        left = cv2.resize(
            left,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        right = cv2.resize(
            right,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    left = np2torch(left, bgr=True).unsqueeze(0)
    right = np2torch(right, bgr=True).unsqueeze(0)
    pred = model(left, right)

    disp = pred["disp"]
    disp = torch.clip(disp / 192 * 255, 0, 255).long()
    disp = apply_colormap(disp)

    output = [left, disp]
    if "slant" in pred:
        dxy = dxy_colormap(pred["slant"][-1][1])
        output.append(dxy)

    output = torch.cat(output, dim=0)
    torchvision.utils.save_image(output, op, nrow=1)

    print("output: {}".format(op))

    return

@torch.no_grad()
def export(model :pl.LightningModule, lp, rp, width, op):
    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
    if width is not None and width != left.shape[1]:
        height = int(round(width / left.shape[1] * left.shape[0]))
        left = cv2.resize(
            left,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        right = cv2.resize(
            right,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    left = np2torch(left, bgr=True).cpu().unsqueeze(0)
    right = np2torch(right, bgr=True).cpu().unsqueeze(0)

    output_name = "oak_sized_model"

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
    parser.add_argument("--images", nargs=2, required=True)
    parser.add_argument("--model", type=str, default="HITNet")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", default="./")
    args = parser.parse_args()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model.cpu()

    if "*" in args.images[0]:
        lps = list(sorted(Path(".").glob(args.images[0])))
        rps = list(sorted(Path(".").glob(args.images[1])))

        for ids, (lp, rp) in enumerate(zip(lps, rps)):
            op = Path(args.output) / f"{lp.stem}_{ids}.png"
            predict(model, lp, rp, args.width, op)
    else:
        lp = Path(args.images[0])
        rp = Path(args.images[1])
        op = Path(args.output)
        if op.is_dir():
            op = op / lp.name
        # predict(model, lp, rp, args.width, op)
        export(model, lp, rp, args.width, op)

