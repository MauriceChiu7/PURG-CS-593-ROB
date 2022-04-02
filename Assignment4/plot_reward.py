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
    legends = []

    # pathName = args.path_file[0].split('/')[-1].split('.')[0]

    for i in range(len(args.file_paths)):
        model = args.file_paths[i].split('.')[0].split('_')[3]
        episodes = args.file_paths[i].split('.')[0].split('_')[5]
        # print(model)
        # print(iterations)
        # exit(0)
        file = open(args.file_paths[i])
        csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        y_axis_avgRew = []
        for row in csvreader:
            y_axis_avgRew = row
        file.close()
        # if args.verbose: print(f"\n...final actions read\n")
        x_axis_iter = [i+1 for i in range(len(y_axis_avgRew))]
        # print(len(y_axis_avgRew))
        # print(len(x_axis_iter))
        # print(y_axis_avgRew)
        # print(x_axis_iter)
        

        subtitle = f"Avg Reward with Q-1.{model} Loss Function and {episodes} Episodes ({colors[i]})"
        

        plt.plot(x_axis_iter, y_axis_avgRew, color=colors[i], marker='o')
        legends.append(subtitle)

    for l in legends:
        plt.figtext(0.1, 0.95-(0.05*legends.index(l)), l)

    plt.figtext(0.1, 0.02, f"Average Rewards", wrap=True)

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