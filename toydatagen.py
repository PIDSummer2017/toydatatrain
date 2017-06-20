import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path
import numpy as np
import random
from scipy import optimize


def image(array):
    #  print array.shape
    fig,ax = plt.subplots(figsize=(28,28),facecolor='w')
    plt.imshow(array,interpolation = 'nearest')
    # plt.show()    

def choose_triangle(x, y,array, tracker):
    z = int(random.uniform(3,4))
    
    if (z < 3 or z >=4) and array[x,y] < 10:
        array[x,y] = 0
        return

    if 3 <= z < 4:
        if 3 <= x <= len(array[0])-3:
            if 4 <= y <= len(array[1]) - 4:
                array[x, y] = 180 + array[x, y]
                array[x-1:x+1, y-1] = 180 + array[x-1:x+1, y-1]
                array[x-2:x+2, y-2] = 180 + array[x-2:x+2, y-2]
                array[x-3:x, y-3] = 180 + array[x-3:x, y-3]
                array[x-4:x, y-4] = 180 + array[x-4:x, y-4]
                array[x:x+3, y-3] = 180 + array[x:x+3, y-3]
                array[x:x+4, y-4] = 180 + array[x:x+4, y-4]

                tracker.append([1,0,0,0])

def choose_rectangle(x, y, array, tracker):
    z = random.uniform(0,1)
    if 0 < z <= 1:
        if x+5 <= len(array[0]):
            if y+5 <= len(array[1]):
                array[x:x+5, y:y+5] = 0
                points = [(x, y), (x+5, y), (x, y+5), (x+5, y+5)]
                start_pt, end_pt = min(points), max(points)
                array[start_pt[1]:end_pt[1]+1, start_pt[0]:end_pt[0]+1] = 180 + array[start_pt[1]:end_pt[1]+1, start_pt[0]:end_pt[0]+1]
                tracker.append([0,1,0,0])
    elif array[x,y] < 10:
        array[x,y] = 0

def choose_lines(x, y, array, tracker):
    z = random.uniform(0,2)
    if 0 < z <= 1:
            if y+5 <= len(array[1]):
                points = [(x, y), (x, y+5)]
                array[x, y:y+5] = 180 + array[x, y:y+5]
                tracker.append([0,0,1,0])
    elif 1 < z <= 2:
        if x+5 <= len(array[0]):
            points = [(x, y), (x+5, y)]
            array[x:x+5, y] = 180 + array[x:x+5, y]
            tracker.append([0,0,0,1])
    elif array[x,y] < 10:
        array[x,y] = 0

def add_shapes_to(array, locs, npoints = 1):
    #triangle_coords = []
    # rectangle_coords = []
    #horizontal_coords = []
    #vertical_coords = []
    for i in range(npoints):
        row=int(random.uniform(5,array.shape[0]-5))
        col=int(random.uniform(5,array.shape[1]-5))
        array[row, col] = random.uniform(0,3)
        trianglocs = np.where(np.logical_and(0 < array, array < 1))
        rectlocs = np.where(np.logical_and(array >= 1, array <2))
        linelocs = np.where(array >= 2)

        linexs = linelocs[0]
        lineys = linelocs[1]

        trixs = trianglocs[0]
        triys = trianglocs[1]

        rectxs = rectlocs[0]
        rectys = rectlocs[1]

        for i in range(len(rectxs)):
            x = rectxs[i]
            y = rectys[i]
            choose_rectangle(x,y,array, locs)
        for i in range(len(trixs)):
            x = trixs[i]
            y = triys[i]
            choose_triangle(x,y,array, locs)
        for i in range(len(linexs)):
            x = linexs[i]
            y = lineys[i]
            choose_lines(x,y,array, locs)

def randomize_labels():
    """
    This function returns an array of length 4 where only 1 element
    is set to 1 (randomly chosen) and the rest is set to 0.
    """
    labels = [0,0,0,0]
    z = random.randint(0, 3)
    labels[z] = 1
    return labels

# don't mess with this
class image_gen_counter:
    _counter_ = 0
def make_image_library(num_images=10,debug=0,bad_label=False):

    locations = []
    bad_locations = []
    images = []
                    
    for i in range(num_images):

        if debug:
            print 'Generating image',i

        mat = np.zeros([28,28]).astype(np.float32)
        add_shapes_to(mat, locations)

        if debug>1:
            image(mat)
            plt.savefig('image_%04d.png' % image_gen_counter._counter_)
        
        mat = np.reshape(mat, (784))
        images.append(mat)

        image_gen_counter._counter_ +=1

    if bad_label:
        for loc in locations:
            bad_locations.append(randomize_labels())
        
    if bad_label: 
        return images, bad_locations
    return images, locations
