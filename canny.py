import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import abspath
    
def build_gaussian_filter(size: int, sigma: float = 1):
    out = np.zeros((size, size))
    normal = 1 / (2.0 * np.pi * sigma**2)
    dist = size // 2
    for y in range(-dist, dist + 1, 1):
        for x in range(-dist, dist + 1, 1):
            out[y+dist][x+dist] = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return out

def sobel_apply(img):
    sobel_x = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]], np.float32)
    sobel_y = np.array([[1,2,1],
               [0,0,0],
               [-1,-2,-1]], np.float32)
    
    img_x = conv(img, sobel_x)
    img_y = conv(img, sobel_y)

    strength = np.sqrt(img_x ** 2, img_y ** 2)
    strength = strength / strength.max() * 255
    theta = - np.arctan(img_y/img_x)
    return strength, theta

def non_maximum_suppression(img, theta):
    out = np.ndarray(img.shape)
    img = np.pad(img, 1)
    theta = np.pad(theta, 1)
    theta = theta * (180 / np.pi) + 180
    X, Y = img.shape
    for y in range(1, Y - 1):
        for x in range(1, X - 1):
            currT = theta[y][x]
            if (currT < 12.25 or currT >= 347.75):
                pass
        


def load_picture(path: str):
    img = mpimg.imread(path)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2] # Source: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grey

def show_picture(imgs):
    plt.figure(figsize=(20,40))
    for i, img in enumerate(imgs):
        plt.subplot(2,2, i+1)
        plt.imshow(img, 'gray')
        plt.tight_layout()
    plt.show()

def conv(img, filter):
    filter_size = len(filter) // 2
    out = np.ndarray(img.shape)
    img = np.pad(img, filter_size)
    for y in range(filter_size, len(img) - filter_size, 1):
        for x in range(filter_size, len(img[y]) - filter_size, 1):

            curr = 0
            for y_f in range(len(filter)):
                for x_f in range(len(filter)):
                    curr += img[y + (y_f - filter_size)][x + (x_f - filter_size)] * filter[y_f][x_f]
            out[y-filter_size][x-filter_size] = curr
    return out


if __name__ == "__main__":
    img = load_picture(abspath("./cv-parktikum/demo-img.jpg"))
    out = conv(img, build_gaussian_filter(5, 1.4))
    sobel, theta = sobel_apply(out)
    theta = theta * (180 / np.pi) + 180
    print(theta)
    print(theta.min())
    imgs = [img, out, img - out, sobel]
    show_picture(imgs)
    
