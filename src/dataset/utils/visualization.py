import os
from matplotlib import pyplot as plt
import matplotlib
import time
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting import plot_anat
from matplotlib import cm
from skimage.transform import resize
import io


def plot_batch(batch, seg: bool = False, slice: int = 32, batch_size: int=4):

    def unnorm(data, epsilon=1e-8):
        non_zero = data[data > 0.0]
        mean = non_zero.mean()
        std = non_zero.std() + epsilon
        out = data * std + mean
        out[data == 0] = 0
        return out

    plt.figure(figsize=(10, 3.5))

    for i, volume in enumerate(batch):
        plt.subplot(1, batch_size + 1, i + 1)

        img = volume[:, slice, :].T if seg else volume[0, :, 32, :].T

        npimg = img.cpu().detach().numpy()
        img = npimg if seg else unnorm(npimg)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    # name = f"{round(time.time())}_batch_pred.png" if seg else f"{round(time.time())}_batch_.png"
    # fig.savefig(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf




def plot_3_view(modal: str, vol: np.ndarray, s: int=100, discrete: bool=False,
                color_map: str="gray", save: bool=True):

    views = [vol[s, :, :], vol[:, s, :], vol[:, :, s]]
    fig = plt.figure(figsize=(10, 3.5))

    for position in range(1, len(views) + 1):
        plt.subplot(1, len(views), position)
        plt.imshow(views[position - 1].T, cmap=color_map)
        plt.axis("off")
        if discrete:
            plt.clim(0, 4)

    if discrete:
        plt.colorbar(ticks=range(5))
    else:
        plt.colorbar()

    if save:
        fig.savefig(f'plot_{modal}_{time.time()}.png')
    else:
        plt.show()


def plot_3_view_uncertainty(modal: str, vol: np.ndarray, s: int=100, color_map: str="gray", save: bool=True):

    views = [vol[s, :, :], vol[:, s, :], vol[:, :, s]]
    fig = plt.figure(figsize=(10, 3.5))

    for position in range(1, len(views) + 1):
        plt.subplot(1, len(views), position)
        plt.imshow(views[position - 1].T, cmap=color_map)
        plt.axis("off")
        plt.clim(0, 100)

    plt.colorbar()

    if save:
        fig.savefig(f'plot_unc_{modal}_{time.time()}.png')
    else:
        plt.show()


def plot_axis_overlayed(modalities: dict, segmentation_mask: str, subject: int, axis: str = 'x', save: bool=False):
    """Save or show figure of provided axis"""
    fig, axes = plt.subplots(len(modalities), 1)

    for i, (modality_name, modality_path) in enumerate(modalities.items()):
        display = plot_anat(modality_path, draw_cross=False, display_mode=axis, axes=axes[i], figure=fig, title=modality_name)
        display.add_overlay(segmentation_mask)

    if save:
        fig.savefig(f'results/patient_{subject}.png')
    else:
        matplotlib.use('TkAgg')
        plt.show()


def plot_brain_batch_per_patient(patient_ids, data, save=True):
    for patient in patient_ids:
        patient = data[patient.item()]
        patient_modalities = list(map(lambda x: os.path.join(patient.data_path, patient.patch_name, x), [patient.flair, patient.t2, patient.t1, patient.t1ce]))
        patient_modalities = {"flair": patient_modalities[0],"t2": patient_modalities[1],"t1": patient_modalities[2],"t1ce": patient_modalities[3] }
        patient_seg = os.path.join(patient.data_path, patient.patch_name, patient.seg)
        plot_axis_overlayed(patient_modalities, patient_seg, patient.patch_name, axis='x', save=save)


def plot_batch_slice(images, gt, slice = 10, save=True):
    """Plot, for a given batch, different types of visualizations.
    If paths: plot overlayed axis plot
    If paths=None: plot slice of volume
    """
    for element_index in range(0, len(images)):
        for i, mod_id in enumerate(images):
            patient_mod = images[i][element_index]

            plot_3_view(str(i), patient_mod, slice, save=save)

        patient_seg = gt[element_index]
        plot_3_view('seg', patient_seg, slice, save=save)


def plot_batch_cubes(patient_ids, batch_volumes, batch_gt, patches=1, img_size=30):
    for batch_pos, patient in enumerate(patient_ids[:patches]):
        patient = patient.item()
        modality = batch_volumes[batch_pos][0]
        gt =  batch_gt[batch_pos][0]
        resized_modality = resize(modality,(img_size, img_size, img_size), mode='constant')
        resized_gt = resize(gt, (img_size, img_size, img_size), mode='constant')

        fig = plot_cube(resized_modality, img_size)
        fig.savefig(f'results/3Dplot_{patient}_{batch_pos}.png')

        fig_seg = plot_cube(resized_gt, img_size)
        fig_seg.savefig(f'results/3Dplot_{patient}_gt_{batch_pos}.png')


def plot_cube(cube, dim, gt=False, angle=320):
    def normalize(arr):
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    def explode(data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def expand_coordinates(indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    if gt:
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors, 4)
        plt.register_cmap(cmap=my_cmap)
        cmap = plt.get_cmap('my_cmap')
        cube = np.around(cube)
    else:
        cmap = cm.viridis

    cube = normalize(cube)
    facecolors = cmap(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=dim * 2)
    ax.set_ylim(top=dim * 2)
    ax.set_zlim(top=dim * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    return fig