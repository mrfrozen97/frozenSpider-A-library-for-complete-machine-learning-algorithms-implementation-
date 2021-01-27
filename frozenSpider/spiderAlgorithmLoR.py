"""


Implementation of ML introductory algorithms visulisation as ancillary
Author - mr frozen (or just frozen)
Github - github.com/mrfrozen97
Mail - mrfrozenpeak@gmail.com



This file has class that performs Logistic regression algorithm to the input data and can also visualise it .
The packages used here are numpy(for efficient calculations) and matplotlib(for data visualisation)

This is a preliminary file that performs binary Logistic regression. This is one of the first versions. So a lot
will be improved in the future.


Classes:
Logistic Regression- It carries out regression model
plot model- It carries out data visualisation.







#Mathematics.........................................................................................................................

Maths behind this algorithm.

Equations
f(x) = y = w1x1 + w2x3 + w3x3 + ...... + b
(w1, w2, w3... are weights and b is bias)

We use Sigmoid function to normalise our values. It makes sure that our values lie between 0 and 1.
There is a term called as decision boundary. It is a threshold output(between 0 and 1) above which our point
belongs to a particular class and below which it belongs to other.

We used squared error cost function in linear regression, but in logistic regression while achieving gradient decent we
would rather choose a cost function which causes decent faster when we are far from minimum and slowly decends when we
move closer to the minimum.
So we use cost function called as cross entropy.


Cost function
When it belongs to class 1
=> -log( hø(x))
When it belongs to class 2
=> -log(1 - hø(x))


Or we can simply write it as,
Cost(hø(x,y)) = -y*log(hø(x)) - (1-y)*log(1 - hø(x))

We can get this formula by following derivation.
Let y' = p(y=1|x)   (It means that probability that y == 1 given the value of x)

so, 1-y' = p(y=0|x)

When y=0 , we have 1-y'  and when y=1 we have y'
So, we can generalise it as,

P(y|x) = y'^(y) * (1-y')^(1-y)
-Log(P(y|x)) = y*log(y') + (1-y)*log(1-y')
Which is denoted by -L(y',y)


z = w1x1 + w2x3 + w3x3 + ...... + b
y' = 1/(1 - e^-(z))


Now, using this we need to find what changes should we make in our weights and biases so that that next time we get
better results.
So, let changes to be made in weights be ∂W

∂L/∂W1 = ∂L/∂y' * ∂y'/∂z * ∂z/dW1

1st
∂L/∂y' = -y/y' + (1-y)/(1-y') = (-y(1-y') + y'(1-y)) / (y')(1-y')

2nd
∂y'/∂z = (y')(1-y')

3rd
∂z/∂W1 = x1

∂L/∂W1 = [(-y(1-y') + y'(1-y)) / (y')(1-y')]  * (y')(1-y') * x1
=> [yy' - y + y' - yy' ] * x1
=> (y' - y)*x1

Similarly, using similar concept
∂L/∂b = (y' - y)


Now since we calculated how much change in weights and biases..........
So, we can update them as follows...........

W1 = W1 - α*x1*(y' - y)
b = b - α*(y' - y)

Where, α is the learning rate or the speed at which we want to reach at the minimum.
It is important to not mess up with this rate, because very large learning rate can surpass/skip the minimum and we
won't get the best results and very small learning rate makes it too difficult to actually reach the minimum.

The above equation displays how we would update one weight, but real life data has n number of dimentions and so we need
to update each and every weight and there is only one bias independent of the number of xs because it is constant.



#End of math. Maybe, Never skip maths................................................................









Attributes:
............................
____________________________
x                  -> input x parameters (list of list or np array)
y                  -> input y for training data
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
x_label            -> List x label text for all parameters/dimentions
y_label            -> y label text
xlabel_size        -> x_label font size of text
ylabel_size        -> y_label font size of text
title_size         -> title font size of text
model              -> this is the model object i.e. logistic regression object in case we need any parent class variables
x_calculated       -> The calculated part of model's x parameters or the test data's x coordinates
y_calculated       -> Calculated classification using model or the test data
learning_rate      -> The rate at with we want to train our model (usually in order of 0.001)
n_iters            -> Number of iterations which we we want our model to pass through the training process
weights            -> List of weights used in our model
bias               -> single float/double containing our bias value
parameters         -> count of x parameters, dimentions
coordinates_size   -> size of training data
calculated_point_class0_label -> train data class 0 color
calculated_point_class1_label -> train data class 1 point color
calculated_point_class0_color -> test data class 0 point color
calculated_point_class1_color -> test data class 1 point color
calculated_point_alpha        -> alpha/transparency of the point


#####.....IMP.......####
display_graph -> Specify weather graph needs to be displayed before saving in plot_graph method
color_dict -> Dictionary that contains wide range of colors in the form of dictionary. The keys are the names of colors
              , values are the hex codes of that color. You can get this dict using get_color_dict function()
You can print this dict i.e. print(object.get_color_dict()) to know which colors are available.






You are most welcomed to improve this code:
You can pull request for this code at

github.com/mrfrozen97/                      (In spiderAlgorithms repo)
or
Email - mrfrozenpeak@gmail.com






"""








import numpy as np
import matplotlib.pyplot as plt
from frozenSpider import spiderAlgorithmResources as res
#from sklearn import datasets
#from sklearn.model_selection import  train_test_split
#from frozenSpider import Data_visualisation as dv





# Class which implements logistic regression...................................................................s

class LogisticRegression:



    #init method takes in some default input values. It is not compulsory to set this variables, everything is predefined
    #using default values......................

    def __init__(self, learning_rate=0.0001, n_iters = 1000 ):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.parameters = 0
        self.coordinates_size = 0
        self.x = []
        self.y = []
        self.y_actual = []
        self.x_calculated = []
        self.y_calculated = []


     #Function which calculates z.......
     #Math formula explained above code...................................


    def calculate_z(self, x_row):

        z = np.dot(x_row, self.weights) + self.bias
       # print(z)
        return z



    #Sigmoid normalisation function

    def sigmoid(self, z):

        return (1/(1+np.exp(-z)))





    #setter function to set decision boundary if any............................................

    def set_decision_boundary(self, x):
        self.decision_boundary = x





    #Following the comman naming convention best fit function calculated all weights and biases.......................

    def best_fit(self, x, y):
        self.x = x
        self.y = y
        x = np.array(x)

        self.coordinates_size, self.parameters = x.shape
        self.decision_boundary = 0.5
        self.weights = []
        self.bias = 3
        self.weights = np.zeros(self.parameters)


        for i in range(self.n_iters):
            alpha = self.sigmoid(self.calculate_z(x))
            dw = (1/self.coordinates_size) * np.dot(x.T, (alpha - y))
            db = (1/self.coordinates_size) * np.sum(alpha - y)

            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

            #print(dw, db)
        self.y_actual = alpha






    #Predict method used to test the test data/ unkown data.............................................


    def predict(self, x, decision_boundary=0.5):
        self.x_calculated = x
        self.decision_boundary = decision_boundary

        alpha = self.sigmoid(self.calculate_z(x))
        self.y_calculated = alpha
        calculated_class = [1 if i>=self.decision_boundary else 0 for i in alpha]
       # print(list(self.sigmoid(self.calculate_z(x))))
        return calculated_class







#This function plots n graphs of n dimentions of the logistic regression model


class plot_model(LogisticRegression):


    #All the default value are already set but they can all be reset.........................
    #Model object need to be passes so that we get other information about the model....................

    def __init__(self, model):

        self.showGrid = True
        self.sizeX = 10
        self.sizeY = 6
        self.labelx_count = 10
        self.labely_count = 10
        self.title_color = "#FF0000"
        self.xlabel_color = "#663399"
        self.ylabel_color = "#663399"
        self.label_default = True
        self.title = ""
        self.x_label = ["Dimention " + str(x) for x in range(1,100)]
        self.y_label = ""
        self.xlabel_size = 15
        self.ylabel_size = 15
        self.title_size = 18
        self.model = model
        self.color_dict = res.Resources.get_color_dict()
        self.calculated_point_size = 10
        self.calculated_point_class0_label = "Calculated class 0"
        self.calculated_point_class1_label = "Calculated class 1"
        self.calculated_point_class0_color = "Red"
        self.calculated_point_class1_color = "Cyan"
        self.calculated_point_alpha = 0.5







    #Function to get dictionary to access the different options avaliable for the colors.........................

    def get_color_dict(self):
        return self.color_dict








     #function to set label of each dimention of the graph................................................

    def set_dimention_labels(self, dimention_label):
        self.x_label = dimention_label






    #function to set vlues for title, x/y label and values of labels in case.........................

    def set_marker_properties(self, label_default=True, calculated_point_class1_label = "Calculated class 1", calculated_point_class0_label="Calculated class 0", calculated_point_alpha = 0.5,  calculated_point_class1_color='Red',calculated_point_class0_color="Cyan", title_size=15, xlabel_size=10, ylabel_size=10, title="Output vs dimention", y_label="y coordinates", title_color="#FF0000", xlabel_color="#663399", ylabel_color="#663399"):
        self.title_size = title_size
        self.xlabel_size = xlabel_size
        self.ylabel_size = ylabel_size
        self.y_label = y_label
        self.title = title
        self.xlabel_color = xlabel_color
        self.ylabel_color = ylabel_color
        self.title_color = title_color
        self.label_default = label_default
        self.calculated_point_size = 10
        self.calculated_point_class0_label = calculated_point_class0_label
        self.calculated_point_class1_label = calculated_point_class1_label
        self.calculated_point_class0_color = calculated_point_class0_color
        self.calculated_point_class1_color = calculated_point_class1_color
        self.calculated_point_alpha = calculated_point_alpha










   #Sets the display size which will be equal to size of image if it is being svaed.........................

    def set_display_size(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY













    # main fuction thats plots the graph and saved it if path provided.........................
    #This function iterates through all the dimentions/parameters and plot each one seperate...............




    def plot_model(self, display_graph = True,display_calculated_points = True, alpha=0.6, point_size=25,label_x=[], label_y=[],class1_color='Purple', class0_color='Orange',unknown_points_label="Unknown points",line_label='Best Fit line', point_color='DeepSkyBlue', save_fig_file='dont'):


        # These are the coordinates to plot the line which is inclusion of both known and calculated points


        multi_dimentions = np.array(self.model.x)
        multi_dimentions_calculated = np.array(self.model.x_calculated)
       # print(len(multi_dimentions[0]))

        for dimention in range(len(multi_dimentions[0])):



            x_bestFitLine_coords = multi_dimentions.T[dimention]
            y_bestFitLine_coords = self.model.y_actual

            x_bestFitLine_coords1 = []
            x_bestFitLine_coords2 = []
            y_bestFitLine_coords1 = []
            y_bestFitLine_coords2 = []
            for i in range(len(y_bestFitLine_coords)):
                if y_bestFitLine_coords[i]>=0.5:
                    x_bestFitLine_coords1.append(x_bestFitLine_coords[i])
                    y_bestFitLine_coords1.append(y_bestFitLine_coords[i])
                else:
                    x_bestFitLine_coords2.append(x_bestFitLine_coords[i])
                    y_bestFitLine_coords2.append(y_bestFitLine_coords[i])

            x_bestFitLine_coords_calculated = multi_dimentions_calculated.T[dimention]
            y_bestFitLine_coords_calculated = self.model.y_calculated

            x_bestFitLine_coords1_calculated = []
            x_bestFitLine_coords2_calculated = []
            y_bestFitLine_coords1_calculated = []
            y_bestFitLine_coords2_calculated = []
            for i in range(len(y_bestFitLine_coords_calculated)):
                if y_bestFitLine_coords_calculated[i] >= 0.5:
                    x_bestFitLine_coords1_calculated.append(x_bestFitLine_coords_calculated[i])
                    y_bestFitLine_coords1_calculated.append(y_bestFitLine_coords_calculated[i])
                else:
                    x_bestFitLine_coords2_calculated.append(x_bestFitLine_coords_calculated[i])
                    y_bestFitLine_coords2_calculated.append(y_bestFitLine_coords_calculated[i])




            plt.scatter(x_bestFitLine_coords1, y_bestFitLine_coords1, color=self.color_dict[class1_color],
                        label="class 1", alpha=alpha, zorder=3)                     #plot the best fit line
            plt.scatter(x_bestFitLine_coords2, y_bestFitLine_coords2, color=self.color_dict[class0_color],
                        label="class 0", alpha=alpha, zorder=3)  # plot the best fit line

            if display_calculated_points:
                plt.scatter(x_bestFitLine_coords1_calculated, y_bestFitLine_coords1_calculated, color=self.color_dict[self.calculated_point_class0_color],
                            label=self.calculated_point_class0_label, alpha=alpha, zorder=3)  # plot the best fit line
                plt.scatter(x_bestFitLine_coords2_calculated, y_bestFitLine_coords2_calculated, color=self.color_dict[self.calculated_point_class1_color],
                            label=self.calculated_point_class1_label, alpha=alpha, zorder=3)  # plot the best fit line



            plt.title(self.title, fontdict={"fontsize": 15}, color=self.title_color)
            plt.xlabel(self.x_label[dimention], fontdict={"fontsize": 15}, color=self.xlabel_color)
            plt.ylabel(self.y_label, fontdict={"fontsize": 15}, color=self.ylabel_color)





            plt.legend(loc = 'upper right')

            if self.showGrid:
                plt.grid(color='#cfd8dc', zorder=0)

            figure = plt.gcf()
            figure.set_size_inches(self.sizeX, self.sizeY)


            if not(save_fig_file=='dont') :

                  plt.savefig("./"+save_fig_file +"/"+self.x_label[dimention])#, bbox_inches='tight')
            if display_graph:
               plt.show()
            plt.close()


















#######Sample code..............................................................................................................








#Example 1
"""
lor = LogisticRegression()
lor.best_fit([[1,2, 4],[2,3, 5],[4,5, 8]], [0, 0, 1])

cal = lor.predict([[1,0,4],[2,3, 5],[4,5, 8]])
print(cal)
"""









#Example 2
#Import sciketlearn before implementing it
"""
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


lor = LogisticRegression(learning_rate=0.01, n_iters=10000)
lor.best_fit(x_train, y_train)
prediction = lor.predict(x_test)

accuracy = np.sum(prediction == y_test)/ len(prediction)

print(accuracy)

plot = plot_model(lor)
plot.set_marker_properties()
plot.plot_model(save_fig_file="./sample_graphs", display_graph=False)

"""





#Exampl2 extention. How to plot a T-sne 3D graph using above Logrithmic regression

"""
datav = dv.Logistic_regression_plot(lor)
datav.plot_3D_visuals(plot_test_data=True, save_fig_path="./sample_graphs")
"""



#Exampl2 extention. How to plot a T-sne 2D graph using above Logrithmic regression

"""
datav = dv.Logistic_regression_plot2D(lor)
datav.plot_2D_visuals(plot_test_data=True, save_fig_path="./sample_graphs")

"""








