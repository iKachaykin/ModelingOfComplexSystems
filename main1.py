import numpy as np
import imageio as imageio
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline as cs
from scipy.integrate import quad


def image_to_matrix(path):
    image = imageio.imread(path)
    return np.where(np.array(np.all(image == 255, 2), dtype=int) == 1, 0, 1)


if __name__ == '__main__':

    data_path = '/Users/ivankachaikin/PycharmProjects/MCS/data.png'
    data = image_to_matrix(data_path)

    figsize = (5.42667, 7.7)
    buff = ''
    grid = True
    line_number = 7
    layer_number = 6
    sub_splines_num = 3
    sub_splines_dim = [3, 5, 9]
    min_dist_between_pixels = 20
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    y_all = []
    x, y = np.linspace(xmin, xmax, data.shape[1]), np.linspace(ymin, ymax, data.shape[0])[::-1]
    xx, yy = np.meshgrid(x, y)
    tmp_lst, tmp_big_lst = [], []
    splines = []
    sub_splines = []
    lines_starting_points = np.arange(data.shape[0])[data[:, 0] == 1]
    annotate_x = 0.05
    annotate_fontsize = 14
    annotate_fontweight = 'bold'
    integrals = np.zeros((sub_splines_num, layer_number))

    i, j = 0, 0
    while i < lines_starting_points.size:
        j = i
        tmp_lst = []
        while j < lines_starting_points.size and \
                np.abs(lines_starting_points[i] - lines_starting_points[j]) < min_dist_between_pixels:
            tmp_lst.append(lines_starting_points[j])
            j += 1
        i = j
        tmp_big_lst.append(tmp_lst)

    tmp_lst = []

    for tmp in tmp_big_lst:
        tmp_lst.append(tmp[(len(tmp)-1)//2])

    lines_starting_points = np.array(tmp_lst)

    for i in range(line_number):
        yi = np.empty_like(x)
        k = lines_starting_points[i]
        for j in range(data.shape[1]):
            if data[k, j] == 1:
                yi[j] = ((1.0 - k / data.shape[0]) - ymin) / (ymax - ymin)
                continue
            if data[k+1, j] == 1:
                k += 1
                yi[j] = ((1.0 - k / data.shape[0]) - ymin) / (ymax - ymin)
                continue
            if data[k-1, j] == 1:
                k -= 1
                yi[j] = ((1.0 - k / data.shape[0]) - ymin) / (ymax - ymin)
                continue
            print('Some problems...')

        y_all.append(yi.copy())

    y_all = np.array(y_all)

    for i in range(line_number):
        splines.append(cs(x, y_all[i]))

    for i in range(sub_splines_num):
        sub_splines.append([])
        x_tmp = np.linspace(xmin, xmax, sub_splines_dim[i])
        for j in range(line_number):
            sub_splines[i].append(cs(x_tmp, splines[j](x_tmp)))

    for i in range(sub_splines_num):
        plt.figure(i+1, figsize=figsize)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.grid(grid)
        x_tmp = np.linspace(xmin, xmax, sub_splines_dim[i])
        for xi_tmp in x_tmp:
            plt.plot([xi_tmp, xi_tmp], [ymin, ymax], 'k--')
        for j in range(line_number):
            plt.plot(x_tmp, splines[j](x_tmp), 'ko')
            plt.plot(x, y_all[j], 'k-')
            plt.plot(x, sub_splines[i][j](x), 'k--')
        plt.xticks(np.linspace(xmin, xmax, 5), ['0', '0.25A', '0.5A', '0.75A', 'A'])
        plt.yticks(np.linspace(xmin, xmax, 5), ['0', '0.25B', '0.5B', '0.75B', 'B'])

    for i in range(sub_splines_num):
        plt.figure(i+1)
        for j in range(layer_number):
            plt.annotate(str(j+1), [annotate_x, (y_all[j, 0] + y_all[j+1, 0]) / 2.0], fontsize=annotate_fontsize,
                         fontweight=annotate_fontweight)
            integrals[i, j] = quad(lambda x: np.abs(sub_splines[i][j+1](x) - sub_splines[i][j](x)), xmin, xmax)[0]

    for i in range(sub_splines_num):
        for j in range(layer_number):
            buff += '%.8f' % integrals[i, j] + '\t'
        buff += '\n'

    print(buff)

    plt.figure(sub_splines_num+1, figsize=figsize)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(grid)
    for j in range(line_number):
        plt.plot(x, y_all[j], 'k-')
    for j in range(layer_number):
        plt.annotate(str(j + 1), [annotate_x, (y_all[j, 0] + y_all[j + 1, 0]) / 2.0], fontsize=annotate_fontsize,
                     fontweight=annotate_fontweight)
    plt.xticks(np.linspace(xmin, xmax, 5), ['0', '0.25A', '0.5A', '0.75A', 'A'])
    plt.yticks(np.linspace(xmin, xmax, 5), ['0', '0.25B', '0.5B', '0.75B', 'B'])

    plt.show()
    plt.close()
