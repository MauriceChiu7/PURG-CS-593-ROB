import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct
import numpy as np
import argparse
import math

import data_loader_2d

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

colors = ["orange", "blue", "green", "black", "red"]

def main(args):
    if args.point_cloud:
        # visualize obstacles as point cloud
        file = args.data_path + f'obs_cloud/obc{args.env_id}.dat'
        obs = []
        temp=np.fromfile(file)
        obs.append(temp)
        obs = np.array(obs).astype(np.float32).reshape(-1,2)
        plt.scatter(obs[:,0], obs[:,1], c='gray')
    else:
        # visualize obstacles as rectangles
        size = 5
        print("=====Obstacles=====")
        obc = data_loader_2d.load_obs_list(args.env_id, folder=args.data_path)
        print("==========")
        for (x,y) in obc:
            r = mpatches.Rectangle((x-size/2,y-size/2),size,size,fill=True,color='gray')
            plt.gca().add_patch(r)

    pathNum = 1
    legends = []
    pathName = args.path_file[0].split('/')[-1].split('.')[0]

    for path_file in args.path_file:
        # visualize path
        if path_file.endswith('.txt'):
            path = np.loadtxt(path_file)
        else:
            path = np.fromfile(path_file)
        path = path.reshape(-1, 2)
        print(f'path {pathNum}:\n{path}\n')
        path_x = []
        path_y = []
        for i in range(len(path)):
            path_x.append(path[i][0])
            path_y.append(path[i][1])

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

        plt.plot(path_x, path_y, color=colors[pathNum-1], marker='o')
        legends.append(subtitle)
        pathNum += 1

    # fig = plt.gcf()
    # fig_width, fig_height = fig.get_size_inches()*fig.dpi
    # figx0 = -(fig_width/2)
    # figy0 = -(fig_height/2)
    # print(f"fig_width: {fig_width}")
    # print(f"fig_height: {fig_height}")
    # print(f"figx0: {figx0}")
    # print(f"figy0: {figy0}")

    # plt.suptitle(legends)
    for l in legends:
        # plt.text(figx0, -1*legends.index(l)+14, l)
        # plt.title(l)
        plt.figtext(0.1, 0.95-(0.05*legends.index(l)), l)

    # plt.text(-12, -17.5, f"env-id: {args.env_id}, path-file: {pathName}", wrap=True)
    # plt.title(f"env-id: {args.env_id}, path-file: {pathName}", wrap=True, loc="bottom")
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
