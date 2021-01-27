"""







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
model              -> this is the model object i.e. Naive bayes object in case we need any parent class variables
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
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import random
from frozenSpider import spiderAlgorithmResources as res


class NaiveBayes:

    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.x_classified = []
        self.y_classified = []




    def fit(self, x, y):

        self.x_data =x
        self.y_data =y

        n_samples, n_features = x.shape
        self.y_claasses = np.unique(y)
        self.y_class_count = len(self.y_claasses)


        self.x_means = np.zeros((self.y_class_count, n_features), dtype=np.float64)
        self.x_vars = np.zeros((self.y_class_count, n_features), dtype=np.float64)
        self.priors = np.zeros((self.y_class_count), dtype=np.float64)

        for c in self.y_claasses:
            x_c = x[c==y]
            self.x_means[c, :] = x_c.mean(axis=0)
            self.x_vars[c, :] = x_c.var(axis=0)
            self.priors[c] = x_c.shape[0] / float(n_samples)


    def predict(self, x):
        y_predict = [self.predict_sample(i) for i in x]
        self.x_classified = x
        self.y_classified = y_predict

        return y_predict


    def predict_sample(self, x):

        probabilities_calculated = []

        for i, c in enumerate(self.y_claasses):
            prior = np.log(self.priors[i])
            class_conditional = np.sum(np.log(self.gaussian_normal_function(i, x)))
            probabilities_calculated.append(prior+class_conditional)

        return self.y_claasses[np.argmax(probabilities_calculated)]


    def gaussian_normal_function(self, class_index, x):
        mean = self.x_means[class_index]
        var = self.x_vars[class_index]
        numerator = np.exp(-(x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator/denominator









class plot_model():

    def __init__(self, model):
        self.showGrid = False
        self.sizeX = 10
        self.sizeY = 6
        self.labelx_count = 10
        self.labely_count = 10
        self.title_color = "#FF0000"
        self.xlabel_color = "#663399"
        self.ylabel_color = "#663399"
        self.label_default = True
        self.title = ""
        self.x_label = ["Dimention " + str(x) for x in range(1, 100)]
        self.y_label = ""
        self.xlabel_size = 15
        self.ylabel_size = 15
        self.title_size = 18
        self.model = model
        self.color_dict = res.Resources.get_color_dict()
        self.calculated_point_size = 10
        self.calculated_point_alpha = 0.5
        self.train_labels = []
        self.test_labels = []
        self.train_data_color = [key for key in sorted(self.color_dict)[:15]]
        self.test_data_color = [key for key in sorted(self.color_dict)[15:]]
        random.shuffle(self.train_data_color)
        random.shuffle(self.test_data_color)
        self.legend_position = "upper right"
        self.plot_background = 'dark'








    # Function to get dictionary to access the different options avaliable for the colors.........................

    def get_color_dict(self):
        return self.color_dict








    # function to set label of each dimention of the graph................................................

    def set_dimention_labels(self, dimention_label):
        self.x_label = dimention_label







    def set_marker_properties(self,show_grid = False, plot_background='dark', legend_position="upper right", train_labels=[], test_labels=[], label_default=True, calculated_point_class1_label = "Calculated class 1", calculated_point_class0_label="Calculated class 0", calculated_point_alpha = 0.5,  calculated_point_class1_color='Red',calculated_point_class0_color="Cyan", title_size=15, xlabel_size=10, ylabel_size=10, title="Output vs dimention", y_label="y coordinates", title_color="#FF0000", xlabel_color="#663399", ylabel_color="#663399"):
        self.title_size = title_size
        self.xlabel_size = xlabel_size
        self.ylabel_size = ylabel_size
        self.y_label = y_label
        self.title = title
        self.showGrid = show_grid

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
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.legend_position = legend_position
        self.plot_background = plot_background







  #Sets the display size which will be equal to size of image if it is being svaed.........................

    def set_display_size(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY





# main fuction thats plots the graph and saved it if path provided.........................
    #This function iterates through all the dimentions/parameters and plot each one seperate...............




    def plot_model(self, display_graph = True,display_calculated_points = True, alpha=0.6, point_size=25,label_x=[], label_y=[],class1_color='Purple', class0_color='Orange',unknown_points_label="Unknown points",line_label='Best Fit line', point_color='DeepSkyBlue', save_fig_file='dont'):


        # These are the coordinates to plot the line which is inclusion of both known and calculated points
        if self.plot_background=='dark':
              plt.style.use('dark_background')
        multi_dimentions = np.array(self.model.x_data)
        multi_dimentions_calculated = np.array(self.model.x_classified)
       # print(len(multi_dimentions[0]))
        #print(multi_dimentions)

        for dimention in range(len(multi_dimentions[0])):

            #x_train_data = []
            y_train_data = {}
            y_test_data = {}

            for i,a in enumerate(multi_dimentions):
                #print(i)
                if self.model.y_data[i] in y_train_data:
                    y_train_data[self.model.y_data[i]].append(a[dimention])
                else:
                    y_train_data[self.model.y_data[i]] = [a[dimention]]

            for i, a in enumerate(multi_dimentions_calculated):

                if self.model.y_classified[i] in y_test_data:
                    y_test_data[self.model.y_classified[i]].append(a[dimention])
                else:
                    y_test_data[self.model.y_classified[i]] = [a[dimention]]
                #x_train_data.append(i[dimention])
            #print(y_test_data)

           # print(y_train_data)

            index_color = 0
            for group in y_train_data:
                #print(group)
                y_plot_axis = []

                if len(self.train_labels)==0:
                    actual_train_label = "train class" + str(group)
                else:
                    actual_train_label = self.train_labels[index_color]

                for i in range(len(y_train_data[group])):
                    y_plot_axis.append(group)
                plt.scatter(y_train_data[group], y_plot_axis,
                            color=self.color_dict[self.train_data_color[index_color%15]],
                            label=actual_train_label,
                            alpha=alpha,
                            zorder=3)                     #plot the best fit line
                index_color+=1


            index_color = 0

            if display_calculated_points:

                for group in y_test_data:
                    #print(group)
                    y_plot_axis = []

                    if len(self.test_labels) == 0:
                        actual_test_label = "test class" + str(group)
                    else:
                        actual_test_label = self.test_labels[index_color]


                    for i in range(len(y_test_data[group])):
                        y_plot_axis.append(group)

                    plt.scatter(y_test_data[group], y_plot_axis,
                                label=actual_test_label,
                                alpha=alpha,
                                zorder=3,
                                color=self.color_dict[self.test_data_color[index_color%15]])  # plot the best fit line

                    index_color +=1



            plt.title(self.title,
                      fontdict={"fontsize": 15},
                      color=self.title_color)
            plt.xlabel(self.x_label[dimention],
                       fontdict={"fontsize": 15},
                       color=self.xlabel_color)
            plt.ylabel(self.y_label,
                       fontdict={"fontsize": 15},
                       color=self.ylabel_color)





            plt.legend(loc = self.legend_position)

            if self.showGrid:
                plt.grid(color='#cfd8dc', zorder=0)

            figure = plt.gcf()
            figure.set_size_inches(self.sizeX, self.sizeY)


            if not(save_fig_file=='dont') :

                  plt.savefig("./"+save_fig_file +"/"+self.x_label[dimention])#, bbox_inches='tight')
            if display_graph:
               plt.show()
            plt.close()












#Example breast cancer dataset
"""
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


nb = NaiveBayes()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)

accuracy = (np.sum(y_test==y_predict)/len(y_test) ) * 100

print("Accuracy = " + str(accuracy))
"""



#Example 2 Wine dataset

bc = datasets.load_wine()
x, y = bc.data, bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


nb = NaiveBayes()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)

plot1 = plot_model(nb)
plot1.plot_model()


accuracy = (np.sum(y_test==y_predict)/len(y_test) ) * 100
print(list(y_predict))
print(list(y_test))

print("Accuracy = " + str(accuracy))







#Example 3 Iris dataset
"""
bc = datasets.load_iris()
x, y = bc.data, bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


nb = NaiveBayes()
nb.fit(x_train, y_train)
y_predict = nb.predict(x_test)

accuracy = (np.sum(y_test==y_predict)/len(y_test) ) * 100

print("Accuracy = " + str(accuracy))


"""
