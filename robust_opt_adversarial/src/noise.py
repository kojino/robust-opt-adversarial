import copy

import numpy as np
from PIL import Image


class Noise():
    '''
    Contains functions that apply each noise transformation on any set of input images.
    '''

    def __init__(self):
        return

    def apply_noise(self, data, noise_type, index=None):
        def shrink(p1_shape, p2_shape, resize_shape, stack_fn):
            result = np.zeros(data.shape)
            p1 = np.zeros(p1_shape)
            p2 = np.zeros(p2_shape)
            for i in range(num_data):
                image_array = np.uint8(data[i, :].reshape(imshape) * 255)
                image = Image.fromarray(image_array).resize(
                    resize_shape, Image.ANTIALIAS)
                image = np.array(image) / 255.0
                full_image = stack_fn((p1, image, p2))
                if num_channels == 1:
                    full_image = np.expand_dims(full_image, -1)
                result[i, :] = full_image
            return result

        def noise(f):
            random_values = f(np.random.random(data.shape) * 0.1)
            return np.clip((data + random_values), 0.0, 1.0)

        def format_background(background):
            background = np.expand_dims(background, 0)
            if num_channels == 1:
                background = np.expand_dims(background, -1)
            return np.clip(data + background, 0.0, 1.0)

        data = copy.deepcopy(data)
        noise_type = noise_type
        num_data, num_row_pixels, num_col_pixels, num_channels = data.shape
        if num_channels == 1:
            imshape = (num_row_pixels, num_col_pixels)
            vpadding = (4, num_col_pixels)
            hpadding = (num_row_pixels, 4)
        else:
            imshape = (num_row_pixels, num_col_pixels, num_channels)
            vpadding = (4, num_col_pixels, num_channels)
            hpadding = (num_row_pixels, 4, num_channels)

        if noise_type == 'none':
            return data

        elif noise_type == 'vert_shrink25':
            return shrink(vpadding, vpadding,
                          (num_row_pixels, num_col_pixels - 8), np.vstack)

        elif noise_type == 'horiz_shrink25':
            return shrink(hpadding, hpadding,
                          (num_row_pixels - 8, num_col_pixels), np.hstack)

        elif noise_type == 'both_shrink25':
            data = shrink(hpadding, hpadding,
                          (num_row_pixels - 8, num_col_pixels), np.hstack)
            return shrink(vpadding, vpadding,
                          (num_row_pixels, num_col_pixels - 8), np.vstack)

        elif noise_type == 'light_tint':
            background = 0.2 * np.ones(imshape)
            return format_background(background)

        elif noise_type == 'gradient':
            background = np.zeros(imshape)
            for i in range(num_row_pixels):
                for j in range(num_row_pixels):
                    background[i, j] = ((i + j) / 54.0) * 0.4
            return format_background(background)

        elif noise_type == 'checkerboard':
            background = np.zeros(imshape)
            for i in range(num_row_pixels):
                if (i % 4 == 0) or (i % 4 == 1):
                    background[i, :] = 0.4
                    background[:, i] = 0.2
            return format_background(background)

        elif noise_type == 'pos_noise':
            return noise(lambda x: x + 0.05)

        elif noise_type == 'mid_noise':
            return noise(lambda x: x - 0.05)

        elif noise_type == 'neg_noise':
            return noise(lambda x: -1.0 * (x + 0.05))

        else:
            raise ValueError("This noise type is not currently supported.")
