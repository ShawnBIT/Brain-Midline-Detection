import numpy as np
from skimage.morphology import convex_hull_image
from dpipe.medim.box import mask2bounding_box
from dpipe.medim.preprocessing import get_greatest_component, crop_to_box
from dpipe.medim.shape_ops import zoom, pad
from dpipe.medim.utils import apply_along_axes

#from .data import PIXEL_SPACING, remove_background

def limits_to_mask(limits, threshold=.3, width_ratio=.7):
    mask = limits >= threshold
    mask = get_greatest_component(mask)

    # take convex hull line by line in order to fill "holes" inside the brain
    limits_mask = apply_along_axes(lambda s: convex_hull_image(s[None])[0] if s.any() else s, mask, -1)

    widths = mask.sum(0)
    # drop the lines that are too thin - these are just outliers
    start, stop = mask2bounding_box(widths >= width_ratio * widths.max()).flatten()
    limits_mask[:, :start] = 0

    return limits_mask.astype(bool)


# Here we use the `predictors` interface from `deep-pipe`
# Check out https://deep-pipe.readthedocs.io/en/latest/tutorials/predict.html for motivation


def rescale_nii(predict):
    def wrapped(image):
        # rescale the image
        x = image.get_fdata()
        scale_factor = image.header.get_zooms()[:2] / PIXEL_SPACING
        x = zoom(x, scale_factor, axes=[0, 1])

        contours, limits = predict(x)
        # rescale the predictions back
        contours = zoom(contours, 1 / scale_factor[0], axes=0) / scale_factor[1]
        limits = zoom(limits, 1 / scale_factor[0], axes=0)
        limits = limits_to_mask(limits)

        contours[~limits] = np.nan
        return contours

    return wrapped


def crop_background(predict):
    def wrapped(x):
        # crop the image
        x = remove_background(x)
        start, stop = box = mask2bounding_box(x != 0)
        padding = np.array([start, x.shape - stop]).T
        padding = padding[[0, 2]]
        cropped = crop_to_box(x, box)

        contours, limits = predict(cropped)
        # pad the predictions pack
        limits = pad(limits, padding)
        contours = pad(contours, padding, padding_values=np.nan) + start[1]

        return contours, limits

    return 


def curves2midline(curves):
    b, h = curves.shape
    midline = np.zeros((b, h, 304), dtype=np.uint8)
    for bb in range(b):
        for y in range(h):
            x = int(np.round(curves[bb][y]))
            if x != 0:
                midline[bb][y][x] = 1

    return midline







