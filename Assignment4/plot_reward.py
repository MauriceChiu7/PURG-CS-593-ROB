# mse_x = np.square(np.subtract(x, actual_x)).mean()
# mse_y = np.square(np.subtract(y, actual_y)).mean()
# plt.plot(np.arange(0, len(errors_x)), errors_x, 'b', np.arange(0, len(errors_y)), errors_y, 'r')
# plt.savefig('err_v_iter_x-y.png')
# plt.show()
# plt.clf()

import argparse
import csv
import matplotlib.pyplot as plt

colors = ["Orange", "Blue", "Green", "Black", "Red"]

def main(args):
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

        if path_file.endswith('.txt'):
            subtitle = f"Path {pathNum} - Type: MPNet ({colors[pathNum-1]}), Path Total Cost: {path_cost}"
            # lineColor = "blue"
        else: 
            subtitle = f"Path {pathNum} - Type: RRT* ({colors[pathNum-1]}), Path Total Cost: {path_cost}"
            # lineColor = "orange"

        plt.plot(path_x, path_y, color=colors[pathNum-1], marker='o')
        legends.append(subtitle)
        pathNum += 1

    for l in legends:
        plt.figtext(0.1, 0.95-(0.05*legends.index(l)), l)

    plt.figtext(0.1, 0.02, f"env-id: {args.env_id}, path-file: {pathName}", wrap=True)

    plt.show()



    # filename = f"./{args.robot}_final_actions.csv"
    # file = open(filename)
    # csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    # finalActions = []
    # for row in csvreader:
    #     finalActions.append(row)
    # file.close()
    # if args.verbose: print(f"\n...final actions read\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 4')
    parser.add_argument('-f', '--file-paths', nargs='*', type=str, default=[], help='File paths')
    
    args = parser.parse_args()
    
    main(args)