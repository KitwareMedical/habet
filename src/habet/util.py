import numpy as np
import itk

class ITKImageMetadata:
    def __init__(self, im):
        self.spacing = im.GetSpacing()
        self.direction = im.GetDirection()
        self.origin = im.GetOrigin()
        self.size = im.GetLargestPossibleRegion().GetSize()

    def __eq__(self, o):
        self_dir = itk.array_from_matrix(self.direction)
        o_dir = itk.array_from_matrix(o.direction)
        return (
            np.allclose(self.spacing, o.spacing)
            and np.allclose(self_dir, o_dir)
            and np.allclose(self.origin, o.origin)
            and self.size == o.size
        )


def image_from_array(arr, meta):
    # Reshape if we need to
    size = np.flip(meta.size)
    arr = np.reshape(arr, size)

    im = itk.image_from_array(arr)
    im.SetSpacing(meta.spacing)
    im.SetDirection(meta.direction)
    im.SetOrigin(meta.origin)

    return im

def images_in_same_space(im1, im2):
    return ITKImageMetadata(im1) == ITKImageMetadata(im2)