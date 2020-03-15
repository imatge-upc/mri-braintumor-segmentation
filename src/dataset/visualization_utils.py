from matplotlib import pyplot as plt
import matplotlib
import cv2
import numpy as np
from nilearn.plotting import plot_anat


def plot_3_view(modal, volume_type: np.ndarray, s: int=100, save: bool=True):
    ''' Plot slice of volume seen from each view (x, y, z)'''
    views = [volume_type[s, :, :], volume_type[:, s, :], volume_type[:, :, s]]
    fig, axes = plt.subplots(1, len(views))

    for i, slice in enumerate(views):
        dst = slice.numpy()
        dst = cv2.resize(dst, (200, 200), interpolation=cv2.INTER_CUBIC)
        axes[i].imshow(dst.T, origin="lower")
    if save:
        fig.savefig(f'plot_{modal}.png')

def plot_axis_overlayed(modalities: list, segmentation_mask: str, subject: int, axis: str='x', save: bool=False):
    '''Save or show figure of provided axis'''
    fig, axes = plt.subplots(len(modalities), 1)
    for i, modality in enumerate(modalities):
        display = plot_anat(modality, draw_cross=False, display_mode=axis, axes=axes[i], figure=fig, title=subject)
        if i == 0:
            display.add_overlay(segmentation_mask)
    if save:
        fig.savefig(f'patient_{subject}.png')
    else:
        matplotlib.use('TkAgg')
        plt.show()

def plot_batch(images, gt, paths=None):
    '''Plot, for a given batch, different types of visualizations.
    If paths: plot overlayed axis plot
    If paths=None: plot slice of volume
    '''
    for element_index in range(0, len(images)):
        if paths:
            patient_modalities = [paths[0][element_index], paths[1][element_index] ,paths[2][element_index], paths[3][element_index]]
            patient_seg = paths[4][element_index]
            plot_axis_overlayed(patient_modalities, patient_seg, element_index, axis='x', save=True)
        else:
            for i, mod_id in enumerate(images):
                patient_mod = images[i][element_index]
                slice = int(patient_mod.shape[0] / 2)
                plot_3_view(str(i), patient_mod, slice, save=True)

            patient_seg = gt[element_index]
            slice = int(patient_seg.shape[0] / 2)
            plot_3_view('seg', patient_seg, slice)