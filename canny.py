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
    img_y[img_y == 0] = 0.00000001 # avoid division by zero
    theta = - np.arctan(img_x/img_y)
    return strength, theta

def non_maximum_suppression(img, theta):
    out = np.ndarray(img.shape)
    img = np.pad(img, 1)
    theta = theta * (180 / np.pi)
    Y, X = img.shape
    for y in range(1, Y - 1, 1):
        for x in range(1, X - 1, 1):
            currT = theta[y-1][x-1]
            curr = img[y][x]
            setZero = False
            error = True

            if (currT >= -22.5 and currT < 22.5): # horizontal
                if img[y-1][x] > curr or img[y+1][x] > curr:
                    setZero = True
                error = False

            if (currT >= 22.5 and currT < 67.5): # down left to up right
                if img[y-1][x-1] > curr or img[y+1][x+1] > curr:
                    setZero = True
                error = False
               
            if (currT >= 67.5 or currT < -67.5): # vertical
                if img[y][x-1] > curr or img[y][x+1] > curr:
                    setZero = True
                error = False

            if (currT >= -67.5 and currT < -22.5): # up left to down right
                if img[y+1][x-1] > curr or img[y-1][x+1] > curr:
                    setZero = True
                error = False

            if error:
                print(f"Error, should not reach here. Theta = {currT}")
            
            if setZero:
                out[y-1][x-1] = 0
            else:
                out[y-1][x-1] = curr
    return out

def hysteresis_thresholding(img, Tl = 20, Th = 60):
    out = np.zeros(img.shape)

    # init
    Y, X = img.shape
    for y in range(Y):
        for x in range(X):
            if img[y][x] >= Th:
                out[y][x] = 1
    
    change = True

    while change:
        change = False
        for y in range(Y):
            for x in range(X):
                if out[y][x] != 1 and img[y][x] >= Tl and img[y][x] < Th:
                    set = False
                    for l_y in range(-1, 2):
                        for l_x in range(-1, 2):
                            if x > 0 and y > 0 and x < X-1 and y < Y-1 and out[y + l_y][x + l_x] == 1:
                                set = True
                    if set:
                        change = True
                        out[y][x] = 1

    return out


def load_picture(path: str):
    img = mpimg.imread(path)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2] # Source: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return grey

def show_picture(imgs, dim1=2, dim2=2):
    plt.figure(figsize=(20,40))
    for i, img in enumerate(imgs):
        plt.subplot(dim1, dim2, i+1)
        plt.imshow(img, 'gray')
        # plt.tight_layout()
    plt.show()

def conv(img, filter):
    filter_size = len(filter) // 2
    out = np.ndarray(img.shape)
    img = np.pad(img, filter_size)
    Y, X = img.shape
    for y in range(filter_size, Y - filter_size, 1):
        for x in range(filter_size, X - filter_size, 1):

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
    supp = non_maximum_suppression(sobel, theta)
    Th = supp.max() * 0.1
    Tl = Th * 0.05
    hysteresis = hysteresis_thresholding(supp, Tl, Th)
    imgs = [img, out, supp, hysteresis]
    show_picture(imgs)

    
