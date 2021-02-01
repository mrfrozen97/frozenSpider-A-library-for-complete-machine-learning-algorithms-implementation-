import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from frozenSpider.spiderAlgorithmResources import Resources as res
import random

class Linear_Regression_Animation():

    def __init__(self, x, y):

        self.plot_x = []
        self.plot_y = []
        self.curr_b = 0
        self.curr_a = 0
        self.x = x
        self.y = y
        self.x_sum = 0
        self.xy_sum = 0
        self.y_sum = 0
        self.xx_sum = 0
        self.max_x = max(self.x)
        self.min_x = min(self.x)
        self.points_color = 'blue'
        self.line_color = 'red'
        self.color_dict = res.get_color_dict()
        self.alpha = 0.6
        self.x_ticks_count = 15





    def set_plot_properties(self, alpha=0.6, x_ticks_count=15, line_color='red', points_color='blue'):

        self.alpha = alpha
        self.line_color = line_color
        self.points_color = points_color
        self.x_ticks_count = x_ticks_count



    def calculate_coords(self, i):

        self.plot_x.append(self.x[i])
        self.plot_y.append(self.y[i])

        self.x_sum += self.x[i]
        self.y_sum += self.y[i]
        self.xx_sum += self.x[i]**2
        self.xy_sum += self.x[i]*self.y[i]

        numerator = self.xy_sum/(i+1) - (self.y_sum/(i+1) * self.x_sum/(i+1))
        denominator = self.xx_sum/(i+1) - (self.x_sum/(i+1))**2

        if denominator!= 0:
            self.curr_b = numerator/denominator
            self.curr_a = self.y_sum/(i+1) - self.curr_b*self.x_sum/(i+1)
        else:
            self.curr_b = numerator
            self.curr_a = self.y_sum / (i + 1) - self.curr_b * self.x_sum / (i + 1)
        plt.clf()
        plt.scatter(self.plot_x, self.plot_y, color=self.color_dict[self.points_color], alpha=self.alpha)
        plt.plot([self.min_x, self.max_x] ,[self.min_x*self.curr_b + self.curr_a, self.max_x*self.curr_b + self.curr_a],
                 color=self.color_dict[self.line_color])
        #plt.yticks(self.y[::10])
        plt.xticks(self.x[::int(len(self.x)/(len(self.x)/self.x_ticks_count))])
        plt.title("a = " + str(self.curr_a) + "  b = " + str(self.curr_b))
        #plt.style.use('dark_background')



        #print(self.curr_b, end=" ")
        #print(self.curr_a)


    def get_color_dict(self):

        return res.get_color_dict()



    def animate(self, interval=300, frames='defalut'):

        plt.yticks(self.y)
        plt.xticks(self.x)
        #plt.style.use('dark_background')
        plt.scatter(self.plot_x, self.plot_y)
        if frames =='defalut':
            frames = len(self.x)-1
        ani = FuncAnimation(plt.gcf(), self.calculate_coords, interval=interval, frames=frames)
        plt.show()












x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [3, 5, 7.5, 9.4, 10.6, 12.5, 15, 17.6, 20]

x = [i for i in range(200)]
y = [2*i+random.random()*random.randint(-5,6)*i**(0.5) for i in x]


animate = Linear_Regression_Animation(x, y)
animate.animate()









