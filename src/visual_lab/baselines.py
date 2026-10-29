"""Fixed reference transformations using input pixels only."""


def recolor_blue(image):
    """Use only the input pixels: replace exact blue with red everywhere."""
    result = image.copy()
    result.putdata([
        (255, 0, 0) if pixel == (0, 0, 255) else pixel
        for pixel in image.get_flattened_data()
    ])
    return result
