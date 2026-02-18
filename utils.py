import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from os.path import abspath, basename, join, exists
from os import walk, mkdir

def load_contours(path):
    path = abspath(path)
    contours = {}

    # iterate over .mat files
    for _, _, files in walk(path):
        for file in files:
            if file.endswith('.mat'):
                muf = sio.loadmat(join(path, file))
                mu = muf.get("groundTruth")
                edges = []
                _, n = mu.shape
                for i in range(n):
                    curr = mu[0,i]["Boundaries"][0,0]
                    # edge = np.zeros_like(curr, dtype=bool)
                    # edge[:-1, :] |= (curr[:-1, :] != curr[1:, :])
                    # edge[:, :-1] |= (curr[:, :-1] != curr[:, 1:])
                    edges.append(curr)
                contours[basename(file).split('.')[0]] = edges
    return contours

def get_image_paths(basePath):
    basePath = abspath(basePath)

    paths = []
    for _, _, files in walk(basePath):

        for file in files:
            if file.endswith('.jpg'):
                paths.append(join(basePath, file))

    return paths

def save_contours(contours : dict, savePath, num = 0):
    savePath = abspath(savePath)
    if not exists(savePath):
        mkdir(savePath)

    counter = 0
    for key, bounds in contours.items():
        if counter >= num and num != 0:
            return
        for i, boundary in enumerate(bounds):
            boundary = boundary * 255
            im = Image.fromarray(boundary)
            
            finalPath = join(savePath, key + "_" + str(i) + ".jpg")
            im.save(finalPath)
        counter += 1

            
def calc_diffs(tests: dict, ground_truth: dict):
    out = {}
    avg = 0
    for testname, test in tests.items():
        truths = ground_truth[testname]
        percent = 0
        
        for truth in truths:
            diff = abs(test-truth)
            pos = 0
            total = 0
            Y, X = diff.shape
            for y in range(Y):
                for x in range(X):
                    if diff[y][x] > 0.5:
                        pos += 1
                    total += 1
            percentage = 100 - (pos / total) * 100
            if percentage > percent:
                percent = percentage
        avg += percent
        out[testname] = percent
    avg = avg / len(out)
    return out, avg


                

if __name__ == "__main__":
    contours = load_contours("./cv-parktikum/BSDS500-master/BSDS500/data/groundTruth/test/")
    print(len(contours))
    save_contours(contours, "./cv-parktikum/output")
