import matplotlib.pyplot as plt
import numpy as np

my_data = np.genfromtxt('data/1/data.csv', delimiter=',')[1:]
x_left, y_up = map(int, input("введите левую вверхнюю координату x y через пробел\n").split())
x_right, y_down = map(int, input("введите правую нижнюю координату x y через пробел\n").split())
my_data = my_data[((my_data[:,0] >= x_left) & (my_data[:,0] <= x_right) & (my_data[:,1] >= y_down) & (my_data[:,1] <= y_up))]
x, y = my_data[:, 0], my_data[:, 1]
fig, ax = plt.subplots()
ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.scatter(x, y, s=x.size)
plt.show()