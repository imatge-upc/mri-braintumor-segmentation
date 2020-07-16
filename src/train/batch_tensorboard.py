import torch
import torchvision
from matplotlib import pyplot as plt
import numpy as np



def write_image_batch_tensorboard(images, writer, name="batch"):
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    writer.add_image(name, img_grid)

def matplotlib_imshow(img, normalized=False):
    if normalized:
        img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def write_segmentations(segmentation_batch, writer, slice=30, pred=True):
    seg_maps_2d = []
    for seg_map in segmentation_batch:
        seg_slice = seg_map[:, slice, :].unsqueeze(0)
        seg_maps_2d.append(seg_slice)

    name = "Predictions" if pred else "GroundTruth"
    write_image_batch_tensorboard(torch.stack(seg_maps_2d), writer, name)