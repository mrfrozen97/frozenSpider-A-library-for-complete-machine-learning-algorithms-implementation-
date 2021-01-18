"""

Implementation of ML introductory algorithms visulisation as ancillary
Author - mr frozen (or just frozen)
Github - github.com/mrfrozen97
Mail - mrfrozenpeak@gmail.com

This file has class that performs linear regression algorithm to the input data and can also visualise it .
The packages used here are numpy(for efficient calculations) and matplotlib(for data visualisation)



Classes:
Linear Regression- It carries out regression model
plot model- It carries out data visualisation

Equation of best fit line y = bx + a
In linear regression we want to minimise the squared error which is ( y(best_fit) - y(original) )^2

y(o) -> y coordinate of original line
x(o) -> x coordinate of original line
y(b) -> y coordinate of best fit line
x(b) -> x coordinate of best fit line
x(mean) -> mean of all x(o) values
y(mean) -> mean of all y(o) values







#Mathematics.........................................................................................................................

Let F = ∑ (y(b) - y(o))^2
F = ∑ (y(0) - y(b))^2

We also know that y(b) = b*x(b) + a
So, F = ∑ (y(o) - b*x(b) - a)^2

Minimising error.....
a and b are our unkowns and so to find them we partially differentiate them w.r.t. F

∂F/∂a = 0     -(1)
∂F/∂b = 0     -(2)

Let y(b) - y(o) = u

∂F/∂a = ∂F/∂u * ∂u/∂a -(3)
∂F/∂b = ∂F/∂u * ∂u/∂b -(4)

F = u^2
∂F/∂u = 2u

u = ∑y(o) - ∑(bx(b) - a)
∂u/∂a = -1

From  3 and 1,
0 = -2*u
0 = -2*(∑y(o) -∑(bx(b) - a))
na = (∑y(o) -∑(bx(b))
a = (y(mean) - b*x(mean)) -(5)

Now,
∂u/∂a = -∑x(b)

From 4 and 1,
0 = -2*u*∑x(b)
0 = (∑y(o)∑x(b)  -∑(bx(b) - a)*∑x(b) )
∑x(b)*na = ∑y(o)∑x(b)  -∑bx(b)
From 5

∑x(b)*n(y(mean) - bx(mean)) = ∑y(o)∑x(b)  -∑bx(b)
Solving this


b =  ∑(y(o)x(a))(mean) -  (y(mean)*x(mean)) /  ∑x(o)*x(mean) - x(mean)**2                        ###############.....IMP
a = y(mean) - b*x(mean)                                                                          ###############.....IMP

squared mean error is defined as
SME = 1 - ( ∑(y(o)-y(mean))^2 / ∑(y(o)-y(b))^2 )                                                 ###############.....IMP

#End of math. Maybe, Never skip maths................................................................







Attributes:
............................
____________________________
b                  -> indicates slope of best fit line
a                  -> indicates x intercept of the  best fit line
x                  -> input x coordinates
y                  -> input y coordinates
x_mean             -> mean of x values
y_mean             -> mean of y values
squared_mean_error -> squared error to check accuracy
showGrid           -> Boolean which indicates weather grid should be shown or not
sizeX              -> Horizontal size of graph in inches
sizeY              -> Vertical size of graph in inches
labelx_count       -> number of labels on x axis
labely_count       -> number of labels on y axis
title_color        -> color of title of plot
label_default      -> Boolean which tells if the label is default or changed by user
xlabel_color       -> color of x label
ylabel_color       -> color of y label
title              -> title text
x_label            -> x label text
y_label            -> y label text
xlabel_size        -> x_label font size of text
ylabel_size        -> y_label font size of text
title_size         -> title font size of text
model              -> this is the model obeject i.e. linear regression object in case we need any parent class variables
unknown_points_size-> this specifies the size of unknown points if any on the graph
unknown_points_color-> this specifies the color of unknown points if any on the graph
unknown_points_apl]ha->  this specifies the transparency of unknown points if any on the graph


#####.....IMP.......####
color_dict -> Dictionary that contains wide range of colors in the form of dictionary. The keys are the names of colors
              , values are the hex codes of that color. You can get this dict using get_color_dict function()
You can print this dict i.e. print(object.get_color_dict()) to know which colors are available.



You are most welcomed to improve this code:
You can pull request for this code at

github.com/mrfrozen97/               (In spider algorithms repo)
or
Email - mrfrozenpeak@gmail.com

"""





import numpy as np
import matplotlib.pyplot as plt
import math








"""

Class that performs linear regression
It takes in the x and y coordinates of the data
It then calculates the value of b and a using numpy array functions
It returns values of a and b
It also returns values of squared mean error in squared_error function

"""

class Linear_regression:


    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.squared_mean_error = 0.0
        self.x_mean = 0
        self.y_mean = 0
        self.size = 1
        self.x = []
        self.y = []
        self.calculated_y = []
        self.calculated_x = []


    #Function to calcute and return a, b values....................................

    def bestFit(self, x_coordinates, y_coordinates):

        size_coordinates = np.size(x_coordinates)
        x_coordinates_mean = np.sum(x_coordinates)/size_coordinates
        y_coordinates_mean = np.sum(y_coordinates)/size_coordinates

        self.x = x_coordinates
        self.y = y_coordinates
        self.size = size_coordinates
        self.x_mean = x_coordinates_mean
        self.y_mean = y_coordinates_mean

        bestFint_numerator = (np.sum(np.multiply(x_coordinates, y_coordinates))/size_coordinates) - (x_coordinates_mean*y_coordinates_mean)
        bestFint_denominator = (np.sum(np.multiply(x_coordinates,x_coordinates))/size_coordinates) - (x_coordinates_mean**2)
        self.b = bestFint_numerator/bestFint_denominator
        self.a = y_coordinates_mean - (self.b*x_coordinates_mean)
        return self.a, self.b




    def bestFit_logrithemic(self, x_coordinates, y_coordinates, base=math.e):

        y_coordinates = [math.log(x, base) for x in x_coordinates]

        size_coordinates = np.size(x_coordinates)
        x_coordinates_mean = np.sum(x_coordinates)/size_coordinates
        y_coordinates_mean = np.sum(y_coordinates)/size_coordinates

        self.x = x_coordinates
        self.y = y_coordinates
        self.size = size_coordinates
        self.x_mean = x_coordinates_mean
        self.y_mean = y_coordinates_mean

        bestFint_numerator = (np.sum(np.multiply(x_coordinates, y_coordinates))/size_coordinates) - (x_coordinates_mean*y_coordinates_mean)
        bestFint_denominator = (np.sum(np.multiply(x_coordinates,x_coordinates))/size_coordinates) - (x_coordinates_mean**2)
        self.b = bestFint_numerator/bestFint_denominator
        self.a = y_coordinates_mean - (self.b*x_coordinates_mean)
        return self.a, self.b








    #function that calculates and returns squarred error which can be attributed to the accuracy og our best fit line............

    def squared_error(self):

        mean_y_array = [self.y_mean for x in range(self.size)]
        bestFit_y = [(self.b*x + self.a) for x in self.x]

       # print(mean_y_array)
       # print(bestFit_y)

        squared_error_denominator = np.sum(np.square(np.subtract(bestFit_y, mean_y_array)))
        squared_error_numerator = np.sum(np.square(np.subtract(bestFit_y, self.y)))


        self.squared_mean_error = 1 - (squared_error_numerator/squared_error_denominator)

        return self.squared_mean_error

    def find_unknowns(self, x):
        x = list(x)
        self.calculated_x = x
        for i in x:
            self.calculated_y.append(self.b * i + self.a)
        return self.calculated_y













"""

This is child class of the linear regression model class
It basically plots the result obtained by performing linear regression vs the original data
The parameters such as title, x/y labels , their colors, their size, plot colors, x/y label values, grid, etc can be set
There are function to set these values as well as to set the size of graph, to save it, etc.
All the default values of the above variables are already set so you can directly plot entire graph with just one line of code......

"""

class plot_model(Linear_regression):


    #All the default value are already set but they can all be reset.........................

    def __init__(self, model):
        self.showGrid = True
        self.sizeX = 10
        self.labelx_count = 10
        self.labely_count = 10
        self.title_color = "#FF0000"
        self.xlabel_color = "#663399"
        self.ylabel_color = "#663399"
        self.label_default = True
        self.sizeY = 6
        self.title = ""
        self.x_label = ""
        self.y_label = ""
        self.xlabel_size = 15
        self.ylabel_size = 15
        self.title_size = 18
        self.model = model
        self.color_dict = {"Red":"#FF0000", "Orange": "#FFA500", "Yellow": "#FFFF00", "DeepPink" :"#FF1493", "LightPink":"#FFB6C1","LightPink":"#FFB6C1"
                           ,"Pink":"#FFC0CB", "Lavender":"#E6E6FA", "Orchid":"#DA70D6", "Violet":"#EE82EE", "DarkOrchid": "#9932CC", "DarkViolet":"#9400D3"
                           ,"BlueViolet":"#8A2BE2", "Purple":"#800080", "Purple":"#800080", "Indigo":"#4B0082", "Salmon":"#FA8072", "Crimson":"#DC143C","DarkRed":"#8B0000",
                           "DarkOrange":"#FF8C00", "Coral":"#FF7F50","OrangeRed":"#FF4500", "Gold":"#FFD700", "GreenYellow":"#ADFF2F", "Lime":"#00FF00", "PaleGreen":"#98FB98"
                           ,"SpringGreen":"#00FF7F", "Green":"#008000", "LightSeaGreen":"#20B2AA", "Cyan":"#00FFFF", "Aquamarine":"#7FFFD4", "SkyBlue":"#87CEEB",
                           "DeepSkyBlue":"#00BFFF", "Blue":"#0000FF", "MediumBlue":"#0000CD", "Navy":"#000080", "Black":"#000000", "Gray":"#808080"}
        self.unknown_points_size = 20
        self.unknown_points_color = "Orange"
        self.unknown_points_alpha = 0.8





    #Function to get dictionary to access the different options avaliable for the colors.........................

    def get_color_dict(self):
        return self.color_dict






    #function to set vlues for title, x/y label and values of labels in case.........................
    def set_marker_properties(self,unknown_points_alpha=0.8, unknown_points_color = "Orange", unknown_points_size = 20, xlabel_count=17, label_default=True, ylabel_count=12, title_size=15, xlabel_size=10, ylabel_size=10, title="Linear Regression", x_label="x coordinates", y_label="y coordinates", title_color="#FF0000", xlabel_color="#663399", ylabel_color="#663399"):
        self.title_size = title_size
        self.xlabel_size = xlabel_size
        self.ylabel_size = ylabel_size
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.xlabel_color = xlabel_color
        self.ylabel_color = ylabel_color
        self.title_color = title_color
        self.labelx_count =xlabel_count
        self.labely_count =ylabel_count
        self.label_default = label_default
        self.unknown_points_size = unknown_points_size
        self.unknown_points_color = unknown_points_color
        self.unknown_points_alpha = unknown_points_alpha









   #Sets the display size which will be equal to size of image if it is being svaed.........................

    def set_display_size(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY






    #function to create x,y values according to the best fit line equation................................
    def create_best_fit_line(self):
        y_coords_line = []
        x_coords_line = list(self.model.x)
        for i in self.model.x:
            y_coords_line.append(self.model.b*i + self.model.a)
        #print(self.model.calculated_x)
        for i in self.model.calculated_x:
            x_coords_line.append(i)
            y_coords_line.append(self.model.b*i + self.model.a)
        #print(len(self.model.x) - len(self.model.y))
        return x_coords_line, y_coords_line







    #This function creates f\default ticks for x and y axis ....
    def generate_ticks(self, label_x, label_y):
        if self.label_default or len(label_x)==0 or len(label_y)==0:
             min_y = min(self.model.y)
             min_x = min(self.model.x)
             max_y = max(max(self.model.y), max(self.model.calculated_y))
             max_y += int(max_y*0.3)
             max_x = max(max(self.model.x), max(self.model.calculated_x))
             max_x += int(max_x * 0.2)
            # dif_y = max(1, int(len(self.model.y)/self.labely_count))
             dif_y = max(1, int((max_y-min_y) / self.labely_count))
             #dif_x = max(1, int(len(self.model.x)/self.labelx_count))
             dif_x = max(1, int((max_x-min_x) / self.labelx_count))
             print(dif_x, dif_y)
             x_labels_default = []
             y_labels_default = []


             while min_x<=max_x:
                 x_labels_default.append(min_x)
                 min_x+=dif_x
             while min_y<max_y:

                 y_labels_default.append(min_y)
                 min_y += dif_y


            # print(x_labels_default)
            # print(y_labels_default)

             plt.xticks(x_labels_default)
             plt.yticks(y_labels_default)

        else:
            plt.xticks(label_x)
            plt.yticks(label_y)









    #main fuction thats plots the graph and saved it if path provided.........................

    def plot_model(self,label="Points",alpha=0.6, point_size=25,label_x=[], label_y=[],line_color='Purple',unknown_points_label="Unknown points",line_label='Best Fit line', point_color='DeepSkyBlue', save_fig_path='dont'):


        #These are the coordinates to plot the line which is inclusion of both known and calculated points
        x_bestFitLine_coords, y_bestFitLine_coords = self.create_best_fit_line()

        plt.plot(x_bestFitLine_coords, y_bestFitLine_coords, color=self.color_dict[line_color], label=line_label)                     #plot the best fit line
        plt.scatter(self.model.x, self.model.y, zorder=3, label=label, s=point_size, color=self.color_dict[point_color], alpha=alpha)       #plot all the points
        plt.scatter(self.model.calculated_x, self.model.calculated_y, color=self.color_dict[self.unknown_points_color], label=unknown_points_label, zorder = 4, alpha=self.unknown_points_alpha, s=self.unknown_points_size)
        plt.title(self.title, fontdict={"fontsize": 15}, color=self.title_color)
        plt.xlabel(self.x_label, fontdict={"fontsize": 15}, color=self.xlabel_color)
        plt.ylabel(self.y_label, fontdict={"fontsize": 15}, color=self.ylabel_color)

        self.generate_ticks(label_x, label_y)



        plt.legend(loc = 'upper left')

        if self.showGrid:
            plt.grid(color='#cfd8dc', zorder=0)

        figure = plt.gcf()
        figure.set_size_inches(self.sizeX, self.sizeY)


        if not(save_fig_path=='dont') :

              plt.savefig("./" + save_fig_path)#, bbox_inches='tight')
        plt.show()








#Sample code to work with this file of the library
"""

x = [-22, 17, 22, 28, 35]
y = [-9.3, 5, 8, 12, 14]
x1 = [13, 47]


le1 = linear_regression()
a,b =le1.bestFit(x, y)
le1.find_unknowns(x1)


print("Slope = " + str(b))
print("x intercept = " + str(a))
print("Squared error = "+ str(le1.squared_error()))


plot2 = plot_model(le1)
plot2.set_marker_properties(unknown_points_size=80, unknown_points_alpha=0.6, unknown_points_color='Red')
plot2.plot_model(save_fig_path="plot2")


"""