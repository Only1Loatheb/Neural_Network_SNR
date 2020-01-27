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
from skimage import img_as_float
from skimage.morphology import reconstruction

from skimage import io
moon = io.imread("./data/train/Frame_00002.bmp")
# Convert to float: Important for subtraction later which won't work with uint8
image = img_as_float(moon)
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image



dilated = reconstruction(seed, mask, method='dilation')

from skimage.color import rgb2gray
dilated = rgb2gray(dilated)

markers = np.zeros_like(dilated)
markers[dilated < 0.1] = 1


from skimage.color  import gray2rgb
dilated = gray2rgb(markers)

# from skimage.viewer import ImageViewer
# viewer = ImageViewer(markers)
# viewer.show()

######################################################################
# Subtracting the dilated image leaves an image with just the coins and a
# flat, black background, as shown below.

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(image, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')

ax2.imshow(image - dilated, cmap='gray')
ax2.set_title('image - dilated')
ax2.axis('off')

fig.tight_layout()
plt.show()