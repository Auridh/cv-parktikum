import numpy as np
from os.path import basename, exists
from os import mkdir
from utils import *
from alive_progress import alive_bar
from sklearn.metrics import f1_score



def build_gaussian_filter(size: int, sigma: float = 1):
    out = np.zeros((size, size))
    normal = 1 / (2.0 * np.pi * sigma**2)
    dist = size // 2
    for y in range(-dist, dist + 1, 1):
        for x in range(-dist, dist + 1, 1):
            out[y + dist][x + dist] = (
                np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
            )
    return out


def sobel_apply(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    img_x = conv(img, sobel_x)
    img_y = conv(img, sobel_y)

    strength = np.sqrt(img_x**2, img_y**2)
    strength = strength / strength.max() * 255
    img_y[img_y == 0] = 0.00000001  # avoid division by zero
    theta = -np.arctan(img_x / img_y)
    return strength, theta


def non_maximum_suppression(img, theta):
    out = np.ndarray(img.shape)
    img = np.pad(img, 1)
    theta = theta * (180 / np.pi)
    Y, X = img.shape
    for y in range(1, Y - 1, 1):
        for x in range(1, X - 1, 1):
            currT = theta[y - 1][x - 1]
            curr = img[y][x]
            setZero = False
            error = True

            if currT >= -22.5 and currT < 22.5:  # horizontal
                if img[y - 1][x] > curr or img[y + 1][x] > curr:
                    setZero = True
                error = False

            if currT >= 22.5 and currT < 67.5:  # down left to up right
                if img[y - 1][x - 1] > curr or img[y + 1][x + 1] > curr:
                    setZero = True
                error = False

            if currT >= 67.5 or currT < -67.5:  # vertical
                if img[y][x - 1] > curr or img[y][x + 1] > curr:
                    setZero = True
                error = False

            if currT >= -67.5 and currT < -22.5:  # up left to down right
                if img[y + 1][x - 1] > curr or img[y - 1][x + 1] > curr:
                    setZero = True
                error = False

            if error:
                print(f"Error, should not reach here. Theta = {currT}")

            if setZero:
                out[y - 1][x - 1] = 0
            else:
                out[y - 1][x - 1] = curr
    return out


def hysteresis_thresholding(img, Tl:float=20, Th:float=60):
    out = np.zeros(img.shape)

    # init
    Y, X = img.shape
    for y in range(Y):
        for x in range(X):
            if img[y][x] >= Th:
                out[y][x] = 1

    # while changes still happening
    change = True
    while change:
        change = False
        for y in range(Y):
            for x in range(X):
                if out[y][x] != 1 and img[y][x] >= Tl and img[y][x] < Th:
                    set = False
                    for l_y in range(-1, 2):
                        for l_x in range(-1, 2):
                            if (
                                x > 0
                                and y > 0
                                and x < X - 1
                                and y < Y - 1
                                and out[y + l_y][x + l_x] == 1
                            ):
                                set = True
                    if set:
                        change = True
                        out[y][x] = 1

    return out


if __name__ == "__main__":

    use_gauss = True
    gauss_path = "./predictions/canny-gauss"
    sobel_path = "./predictions/canny-sobel"
    supp_path =  "./predictions/canny-supp"
    hyst_path =  "./predictions/canny-result"


    img = load_picture("demo-img.jpg")
    out = conv(img, build_gaussian_filter(5, 1.4)) if use_gauss else None
    sobel, theta = sobel_apply(out if use_gauss else img)
    supp = non_maximum_suppression(sobel, theta)
    Th = supp.max() * 0.08
    Tl = Th * 0.2
    hysterisis = hysteresis_thresholding(supp, Tl, Th)
    Image.fromarray(hysterisis.astype(np.uint8) * 255).save("demo-img-out.jpg")

    for path in [gauss_path, sobel_path, supp_path, hyst_path]:
        if not exists(path):
            mkdir(path)

    paths = get_image_paths("./BSDS500-master/BSDS500/data/images/test")
    contours = load_contours(
        "./BSDS500-master/BSDS500/data/groundTruth/test/"
    )
    f1_scores = []
    with alive_bar(len(paths)) as bar:
        results = {}
        for i, path in enumerate(paths):
            elem_name = basename(path).split(".")[0]
            img = load_picture(path)
            out = conv(img, build_gaussian_filter(5, 1.4)) if use_gauss else None
            sobel, theta = sobel_apply(out if use_gauss else img)
            supp = non_maximum_suppression(sobel, theta)
            Th = supp.max() * 0.08
            Tl = Th * 0.2
            hysterisis = hysteresis_thresholding(supp, Tl, Th)
            results[basename(path).split(".")[0]] = hysterisis
            labels = contours[elem_name]
            label = np.mean(labels, axis=0)
            
            binary_label = (label > 0.5).astype(np.float32).reshape(-1,1)
            score = f1_score(binary_label, hysterisis.reshape(-1,1))
            f1_scores.append(score)
            print(f"F1 score: {score}")
            
            # save part results
            if use_gauss: Image.fromarray(out.astype(np.uint8)).save(join(gauss_path, elem_name + ".jpg")) # pyright: ignore[reportOptionalMemberAccess]
            Image.fromarray(sobel.astype(np.uint8)).save(join(sobel_path, elem_name + ".jpg"))
            Image.fromarray(supp.astype(np.uint8)).save(join(supp_path, elem_name + ".jpg"))
            Image.fromarray(hysterisis.astype(np.uint8) * 255).save(join(hyst_path, elem_name + ".jpg"))
            bar()

    print("Done loading contours")
    diffs, avg = calc_diffs(results, contours)
    print("Diffs:")
    print(diffs)
    print("#" * 100)
    print(f"Avg: {avg}")
