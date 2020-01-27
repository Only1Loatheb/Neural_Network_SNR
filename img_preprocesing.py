# from skimage import io
# img = io.imread("./data/train/Frame_00002.bmp")

# from skimage import img_as_float
# image = img_as_float(img)

# from skimage.color import rgb2gray
# grayscale = rgb2gray(img)

# from skimage.feature import canny
# edges = canny(grayscale)

# from scipy import ndimage as ndi
# fill_coins = ndi.binary_fill_holes(grayscale)


# from skimage.viewer import ImageViewer
# viewer = ImageViewer(fill_coins)
# viewer.show()

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float,img_as_int
from skimage.util import img_as_uint
from skimage.morphology import reconstruction

from skimage import io

from skimage.color import rgb2gray
from skimage.morphology import remove_small_holes

from skimage.util  import invert
from skimage.morphology import convex_hull_image
from skimage.color  import gray2rgb
from skimage.external.tifffile import imsave
from PIL import Image

def prepoocessing(path):
    moon = io.imread(path)
    # Convert to float: Important for subtraction later which won't work with uint8
    image = img_as_float(moon)
    image = gaussian_filter(image, 1)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')

    dilated = rgb2gray(dilated)


    markers = np.zeros_like(dilated,dtype=bool)
    markers[dilated < 0.1] = True

    remove_small_holes(markers, area_threshold=128, connectivity=1, in_place=True)


    # markers = invert(markers)
    # pading = 8
    # markers = np.pad(markers, ((pading,pading), (pading,pading)), mode='constant')
    # markers = convex_hull_image(markers)
    # markers = invert(markers)
    # markers = markers[pading:-pading,pading:-pading]

    dilated = gray2rgb(img_as_float(markers))
    img = np.maximum(image , dilated)

    return img

import glob
paths = list(glob.glob("./data/train/*.bmp"))

for i,path in enumerate(paths[:1]):
    img = prepoocessing(path)
    
    # imsave(f'./data/moded/{i}.tif',img_as_float(img))