import os

from matplotlib import pyplot as plt
import matplotlib
import time
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting import plot_anat
from matplotlib import cm
from skimage.transform import resize



def plot_3_view(modal, volume_type: np.ndarray, s: int=100, save: bool=True):
    """ Plot slice of volume seen from each view (x, y, z)"""
    views = [volume_type[s, :, :], volume_type[:, s, :], volume_type[:, :, s]]
    fig, axes = plt.subplots(1, len(views))

    for i, slice in enumerate(views):
        dst = slice # .numpy()
        axes[i].imshow(dst.T, cmap='viridis', origin="lower")
    if save:
        fig.savefig(f'plot_{modal}_{time.time()}.png')
    else:
        plt.show()


def plot_axis_overlayed(modalities: list, segmentation_mask: str, subject: int, axis: str = 'x', save: bool=False):
    """Save or show figure of provided axis"""
    fig, axes = plt.subplots(len(modalities), 1)

    for i, modality in enumerate(modalities):
        display = plot_anat(modality, draw_cross=False, display_mode=axis, axes=axes[i], figure=fig, title=subject)
        display.add_overlay(segmentation_mask)

    if save:
        fig.savefig(f'results/patient_{subject}.png')
    else:
        matplotlib.use('TkAgg')
        plt.show()


def plot_brain_batch_per_patient(patient_ids, data, save=True):
    for patient in patient_ids:
        patient = data[patient.item()]
        patient_modalities = list(map(lambda x: os.path.join(patient.data_path, patient.patient, x), [patient.flair, patient.t2, patient.t1, patient.t1ce]))
        patient_seg = os.path.join(patient.data_path, patient.patient, patient.seg)
        plot_axis_overlayed(patient_modalities, patient_seg, patient, axis='x', save=save)


def plot_batch_slice(images, gt, save=True):
    """Plot, for a given batch, different types of visualizations.
    If paths: plot overlayed axis plot
    If paths=None: plot slice of volume
    """
    slice = 10
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