import matplotlib
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np
from mpl_toolkits import mplot3d


from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import struct

import argparse
import math

import data_loader_r3d

def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    v2 = [2, 5, 7]
    v1 = [1, 2, 3]
    list(zip(v1, v2)) -> [(1, 2), (2, 5), (3, 7)]
    print([x for x in range(4)]) -> [0, 1, 2, 3]
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1, p2))






# fig = plt.figure()
# ax = plt.axes(projection='3d')

# xdata = 5*np.random.random(50)
# ydata = 5*np.random.random(50)
# zdata = 5*np.random.random(50)
# ax.scatter(xdata, ydata, zdata);





# Adapted From: https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

# Adapted From: https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
def plotCubeAt2(positions, sizes, colors, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append(cuboid_data2(p, size=s))
        alpha = 0.65
        # randColor = np.random.choice([0.3, 0.7])
    return Poly3DCollection(np.concatenate(g),facecolors=np.repeat(colors,6), alpha=alpha, **kwargs)

# positions = [(0,0,0),(5,5,5)]
# sizes = [(10,10,10), (10,15,10)]

# pc = plotCubeAt2(positions,sizes)
# ax.add_collection3d(pc) 
   






plt.show()

colors = ["Orange", "Blue", "Green", "Black", "Red", "Brown", "Olive", "Cyan", "Gray", "Purple"]
grays = ["Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray"]

def main(args):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])

    # ax.set_xlim([0,20])
    # ax.set_ylim([0,20])
    # ax.set_zlim([0,20])

    if args.point_cloud:
        # visualize obstacles as point cloud
        file = args.data_path + f'obs_cloud/obc{args.env_id}.dat'
        obs = []
        temp=np.fromfile(file)
        obs.append(temp)
        obs = np.array(obs).astype(np.float32).reshape(-1,3)
        # plt.scatter(obs[:,0], obs[:,1], obs[:,2], c='gray')
        ax.scatter3D(obs[:,0], obs[:,1], obs[:,2], c='gray')
    else:
        # visualize obstacles as cuboids
        obcSize = [[5,5,10], [5,10,5], [5,10,10], [10,5,5], [10,5,10], [10,10,5], [10,10,10], [5,5,5], [10,10,10], [5,5,5]]
        obcPos = []
        obc = data_loader_r3d.load_obs_list(args.env_id, folder=args.data_path)
        print("=====Obstacles=====")
        print(obc)
        print("==========")
        for i in range(0,10):
            px = obc[i][0]-(obcSize[i][0]/2)
            py = obc[i][1]-(obcSize[i][1]/2)
            pz = obc[i][2]-(obcSize[i][2]/2)
            obcPos.append([px, py, pz])
            # r = mpatches.Rectangle((x-size/2,y-size/2),size,size,fill=True,color='gray')
            # plt.gca().add_patch(r)
            # print(f"obc[{i}]position[0]: {obc[i][0]}, obc[{i}]size[0]: obcSize[i][0]")

        # pc = plotCubeAt2(obcPos, obcSize, grays, edgecolor="k")
        # pc = plotCubeAt2(obcPos, obcSize, grays)
        pc = plotCubeAt2(obcPos, obcSize, colors)
        ax.add_collection3d(pc) 

    pathNum = 1
    legends = []
    pathName = args.path_file[0].split('/')[-1].split('.')[0]

    for path_file in args.path_file:
        # visualize path
        if path_file.endswith('.txt'):
            path = np.loadtxt(path_file)
        else:
            path = np.fromfile(path_file)
        path = path.reshape(-1, 3)
        print(f'path {pathNum}:\n{path}\n')
        path_x = []
        path_y = []
        path_z = []
        for i in range(len(path)):
            path_x.append(path[i][0])
            path_y.append(path[i][1])
            path_z.append(path[i][2])

        # Calculate the cost of the path
        path_cost = 0
        for p in range(len(path)-1):
            print(f"point {p}: {path[p]}")
            print(f"point {p+1}: {path[p+1]}")
            print(f"dist(p, p+1): {dist(path[p], path[p+1])}")
            path_cost += dist(path[p], path[p+1])
        print(f"cost: {path_cost}")

        if path_file.endswith('.txt'):
            subtitle = f"Path {pathNum} - Type: MPNet ({colors[pathNum-1]}), Path Total Cost: {path_cost}"
            # lineColor = "blue"
        else: 
            subtitle = f"Path {pathNum} - Type: RRT* ({colors[pathNum-1]}), Path Total Cost: {path_cost}"
            # lineColor = "orange"

        plt.plot(path_x, path_y, path_z, color=colors[pathNum-1], marker='o')
        legends.append(subtitle)
        pathNum += 1

    for l in legends:
        plt.figtext(0.1, 0.95-(0.05*legends.index(l)), l)

    plt.figtext(0.1, 0.02, f"env-id: {args.env_id}, path-file: {pathName}", wrap=True)

    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../data/simple/')
parser.add_argument('--env-id', type=int, default=0)
parser.add_argument('--point-cloud', default=False, action='store_true')
parser.add_argument('--path-file', nargs='*', type=str, default=[], help='path file')
args = parser.parse_args()
print(args)
main(args)
