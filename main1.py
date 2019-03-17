import numpy as np
import imageio as imageio
import matplotlib.pyplot as plt


def image_to_matrix(path):
    image = imageio.imread(path)
    return np.where(np.array(np.all(image == 255, 2), dtype=int) == 1, 0, 1)


if __name__ == '__main__':

    data_path = '/Users/ivankachaikin/PycharmProjects/MCS/data.png'
    data = image_to_matrix(data_path)

    figsize = (5.42667, 7.7)
    line_number = 7
    min_dist_between_pixels = 20
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    y_all = []
    x, y = np.linspace(xmin, xmax, data.shape[1]), np.linspace(ymin, ymax, data.shape[0])[::-1]
    xx, yy = np.meshgrid(x, y)
    tmp_lst, tmp_big_lst = [], []
    fig = plt.figure(1, figsize=figsize)

    lines_starting_points = np.arange(data.shape[0])[data[:, 0] == 1]

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

    plt.axis([xmin, xmax, ymin, ymax])
    for yi in y_all:
        plt.plot(x, yi, 'k-')
    plt.xticks([xmin, xmax], ['0', 'A'])
    plt.yticks([ymin, ymax], ['0', 'B'])

    plt.show()
    plt.close()
