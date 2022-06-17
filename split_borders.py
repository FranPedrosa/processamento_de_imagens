from skimage import io, color, morphology, filters
import matplotlib.pyplot as plt
import numpy as np
import sys

borders = []
mask = []
paths = []
closed_paths = []
not_visited = []
min_size = 20


def neighbors(center):
    x,y = center[0],center[1]
    return (x+1,y),(x+1,y-1),(x,y-1),(x-1,y-1),(x-1,y),(x-1,y+1),(x,y+1),(x+1,y+1)

def new_path(point):
    paths.append([point])

def split_path(path,p1,p2):
    new_path = list(path)
    add_point(new_path,p1)
    paths.append(new_path)
    return add_point(path,p2)

def add_point(path,p):
    global paths
    if p not in borders:
        return False
    if p in path:
        index = path.index(p)
        closed = path[index:]
        if len(closed) > min_size:
            closed_paths.append(closed)
        return False
    if p not in not_visited:
        new_paths = []
        for other_path in paths:
            if p in other_path:
                index = other_path.index(p)
                new_path = path + other_path[index:]
                new_paths.append(new_path)
        paths += new_paths
        return False
    if p not in borders:
        return False
    path.append(p)
    return True

def process_point(path,p):
    if not p in not_visited:
        return False
    not_visited.remove(p)
    ns = neighbors(p)
    had_white = False
    first_1 = None
    last_1 = None
    first_2 = None
    last_2 = None

    for i in range(16):
        if not mask[ns[i%8]]:
            had_white = True

        if mask[ns[i%8]] and had_white and first_1 is None:
            first_1 = i % 8

        if not mask[ns[i%8]] and not first_1 is None and last_1 is None:
            if ((i-1) % 8) == first_1:
                first_1 = None
            else:
                last_1 = (i -1) % 8

        if mask[ns[i%8]] and not last_1 is None and first_2 is None:
            first_2 = i % 8

        if not mask[ns[i%8]] and not first_2 is None and last_2 is None:
            if ((i-1)%8) == first_2:
                first_2 = None
            else:
                last_2 = (i -1) % 8

    if last_1 is None:
        return False
    
    if last_2 is None or first_1 == first_2:
        p = ns[last_1]
        return add_point(path,p)
    else:
        p1 = ns[last_1]
        p2 = ns[last_2]
        return split_path(path,p1,p2)
            
def process_path(path):
    last = path[-1]
    while process_point(path,last):
        last = path[-1]

def split(m,s):
    global mask, borders, paths, not_visited, min_size, closed_paths
    paths = []
    mask = m
    min_size = s
    cross = [[0,1,0],[1,1,1],[0,1,0]]
    centers = morphology.erosion(mask, cross)
    borders = np.logical_and(mask,np.logical_not(centers))
    plt.imshow(borders)
    borders = [(p[0],p[1]) for p in np.argwhere(borders)]
    last_path = 0
    not_visited = list(borders)
    closed_paths = []

    while len(not_visited) > 0:
        start = not_visited[0]
        new_path(start)
        while(last_path < len(paths)):
            process_path(paths[last_path])
            last_path += 1

    return closed_paths
