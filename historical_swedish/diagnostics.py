"""Conservative image diagnostics; these do not measure handwriting readability."""
import numpy as np


def image_diagnostics(image):
    pixels = np.asarray(image.convert('L'))
    dark_fraction = float((pixels < 200).mean())
    return dict(pixel_min=int(pixels.min()), pixel_max=int(pixels.max()),
                pixel_std=float(pixels.std()), dark_fraction=dark_fraction,
                near_blank=dark_fraction < 0.001)


def edge_diagnostics(image):
    dark = np.asarray(image.convert('L')) < 96
    band = 24
    coverage = {
        'top': float(dark[:band].mean(axis=1).max()),
        'bottom': float(dark[-band:].mean(axis=1).max()),
        'left': float(dark[:, :band].mean(axis=0).max()),
        'right': float(dark[:, -band:].mean(axis=0).max()),
    }
    return dict(edge_coverage=coverage, horizontal_edge_flag=max(coverage['top'], coverage['bottom']) >= 0.35,
                vertical_edge_flag=max(coverage['left'], coverage['right']) >= 0.65)
