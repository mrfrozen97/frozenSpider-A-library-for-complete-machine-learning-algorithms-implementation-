"""
Implementation of ML introductory algorithms visulisation as ancillary
Author - mr frozen (or just frozen)
Github - github.com/mrfrozen97
Mail - mrfrozenpeak@gmail.com



While handling real world data, the major part of the problem arises when we have to visulize the data because the data
has many parameters which we treat as dimensions. It is very difficult to visualise such data and so instead of wrapping
our mind to perplexing multi dimensional space, we use some algorithms which make our lives easier by transforming this
data to a lower dimensional data which can be plotted and easily understood. Surprisingly, this algorithm called as T-sne
is very powerful. Its efficacy can be observed by the accuracy at which it differentiates between the data.




This file has class that can plot 2d as well as 3d model of a object of logarithmic regression class.
The packages used here are matplotlib(for data visualisation)

This is a preliminary file that performs data visualisation. This is one of the first versions. So a lot
will be improved in the future.


Classes:
Logistic Regression plot - It plots 3D T-sne graph of regression model
Logistic Regression plot 2D - It plots 2D T-sne graph of regression model




#Mathematics...........................................................................................................

For time being, I fully do not understand the math behind T-sne so I would just give a overview of the math working
behind T-sne.

It uses distance between 2 points in nth dimentional space that determines how close two points are in lower dimentinal
space.

It uses the normal function to get a exponential decent, so that the far off points are less significant anyways.

It creates a random arrangement of points in lower dimensional space and then by using the closeness factoe of original
distances decides which points should be moved where.


#end math for now.......................................................






Attributes:
............................
____________________________

model = model
train_class0_color -> color of train data class 0 points
train_class1_color -> color of train data class 1 points
test_class0_color  -> color of test data class 0 points
test_class1_color  -> color of test data class 1 points
title              -> set the title of the plot
background_color   -> set the background color of the plot. Default id black
train_class0_label -> label of train data class 0
train_class1_label -> label of train data class 1
test_class0_label  -> label of test data class 0
test_class1_label  -> label of test data class 1
test_alpha         -> alpha/transparency of the test data
train_alpha        -> alpha/transparency of the train data





#####.....IMP.......####
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
from sklearn.manifold import TSNE
import random
from frozenSpider import spiderAlgorithmResources as res
import sklearn
import math
from sklearn import datasets
from sklearn.model_selection import  train_test_split
from frozenSpider import spiderAlgorithmLoR as lore
#from frozenSpider import Data_visualisation as dv






#Class that automatically extracts data from logistical regression model and plots 3D visualisation of the points......

class Logistic_regression_plot():



    #Init method to set default values..................................................

    def __init__(self, model):
        self.model = model
        self.train_class0_color = "#800080"
        self.train_class1_color = "#FFA500"
        self.test_class0_color = "#00BFFF"
        self.test_class1_color = "#FF1493"
        self.title = "Logistic regression visulisation"
        self.background_color = (0, 0, 0)
        self.train_class0_label = "class0 train"
        self.train_class1_label = "class1 train"
        self.test_class0_label = "class0 test"
        self.test_class1_label = "class1 test"
        self.test_alpha = 0.6
        self.train_alpha = 0.9
        self.color_dict = res.Resources.get_color_dict()






    #Method to set the properties of the graph................................
    #Not necessary, because all the default values are already set......................................

    def set_plot_properties(self, train_alpha = 0.9,test_alpha = 0.6,test_class1_label = "class1 test",test_class0_label = "class0 test",train_class1_label = "class1 train", train_class0_label = "class0 train", background_color = (0, 0, 0),title = "Logistic regression visulisation",train_class0_color = "#800080", train_class1_color = "#FFA500", test_class0_color = "#00BFFF", test_class1_color = "#FF1493"):
        self.train_class0_color = train_class0_color
        self.train_class1_color = train_class1_color
        self.test_class0_color = test_class0_color
        self.test_class1_color = test_class1_color
        self.title = title
        self.background_color = background_color
        self.train_class0_label = train_class0_label
        self.train_class1_label = train_class1_label
        self.test_class0_label = test_class0_label
        self.test_class1_label = test_class1_label
        self.test_alpha = test_alpha
        self.train_alpha = train_alpha





    #method that does all the calculatons before plotting the graph.....................................................

    def plot_3D_calculations(self, model1, x, y):
        tsne_data = model1.fit_transform(x)
        class1 = []
        class2 = []

        for i in range(len(y)):
            if y[i] == 0:
                class1.append(tsne_data[i])
            else:
                class2.append(tsne_data[i])

        class1 = np.array(class1).T
        class2 = np.array(class2).T
        return class1, class2







    # Method to plot the 2d graph of the calculated values after apllying T-sne.......................................

    def plot_3D_visuals(self, plot_train_data =True, plot_test_data = False, save_fig_path="dont"):
        model1 = TSNE(n_components=3)

        if plot_train_data:
             class1, class2 = self.plot_3D_calculations(model1, self.model.x, self.model.y)
        if plot_test_data:
             class3, class4 = self.plot_3D_calculations(model1, self.model.x_calculated, self.model.y_calculated)

        plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        ax.text2D(0.30, 0.98, self.title, transform=ax.transAxes)

        ax.set_facecolor(self.background_color)
        if plot_train_data:
            ax.text2D(0.90, 0.98, self.train_class0_label, transform=ax.transAxes, color=self.train_class0_color)
            ax.text2D(0.90, 0.94, self.train_class1_label, transform=ax.transAxes, color=self.train_class1_color)
            ax.scatter3D(class1[0], class1[1], class1[2], color=self.train_class0_color, alpha=self.train_alpha, label = self.train_class0_label)
            ax.scatter3D(class2[0], class2[1], class2[2], color=self.train_class1_color, alpha=self.train_alpha, label= self.train_class1_label)
        if plot_test_data:
            ax.text2D(0.90, 0.90, self.test_class0_label, transform=ax.transAxes, color=self.test_class0_color)
            ax.text2D(0.90, 0.86, self.test_class1_label, transform=ax.transAxes, color=self.test_class1_color)
            ax.scatter3D(class3[0], class3[1], class3[2], color=self.test_class0_color, alpha=self.test_alpha, label=self.test_class0_label)
            ax.scatter3D(class4[0], class4[1], class4[2], color=self.test_class1_color, alpha=self.test_alpha, label=self.test_class1_label)
        if not(save_fig_path=="dont"):
            plt.savefig(save_fig_path + "/3D" + self.title)

        plt.show()
















#Class that automatically extracts data from logistical regression model and plots 2D visualisation of the points......


class Logistic_regression_plot2D():

    # Init method to set default values..................................................

    def __init__(self, model):
        self.model = model
        self.train_class0_color = "#800080"
        self.train_class1_color = "#FFA500"
        self.test_class0_color = "#00BFFF"
        self.test_class1_color = "#FF1493"
        self.title = "Logistic regression visulisation"
        self.background_color = (0, 0, 0)
        self.train_class0_label = "class0 train"
        self.train_class1_label = "class1 train"
        self.test_class0_label = "class0 test"
        self.test_class1_label = "class1 test"
        self.test_alpha = 0.6
        self.train_alpha = 0.9
        self.x_label = "X axis"
        self.y_label = "Y axis"
        self.color_dict = res.Resources.get_color_dict()





    # Method to set the properties of the graph................................
    # Not necessary, because all the default values are already set......................................

    def set_plot_properties(self, x_label = "X axis", y_label = "Y axis", train_alpha = 0.9,test_alpha = 0.6,test_class1_label = "class1 test",test_class0_label = "class0 test",train_class1_label = "class1 train", train_class0_label = "class0 train", background_color = (0, 0, 0),title = "Logistic regression visulisation",train_class0_color = "#800080", train_class1_color = "#FFA500", test_class0_color = "#00BFFF", test_class1_color = "#FF1493"):
        self.train_class0_color = train_class0_color
        self.train_class1_color = train_class1_color
        self.test_class0_color = test_class0_color
        self.test_class1_color = test_class1_color
        self.title = title
        self.background_color = background_color
        self.train_class0_label = train_class0_label
        self.train_class1_label = train_class1_label
        self.test_class0_label = test_class0_label
        self.test_class1_label = test_class1_label
        self.test_alpha = test_alpha
        self.train_alpha = train_alpha
        self.x_label = x_label
        self.y_label = y_label






    # method that does all the calculatons before plotting the graph.....................................................

    def plot_2D_calculations(self, model1, x, y):
        tsne_data = model1.fit_transform(x)
        class1 = []
        class2 = []

        for i in range(len(y)):
            if y[i] == 0:
                class1.append(tsne_data[i])
            else:
                class2.append(tsne_data[i])

        class1 = np.array(class1).T
        class2 = np.array(class2).T
        return class1, class2






    #Method to plot the 2d graph of the calculated values after apllying T-sne.......................................


    def plot_2D_visuals(self, plot_train_data =True, plot_test_data = False, save_fig_path="dont"):
        model1 = TSNE(n_components=2)

        if plot_train_data:
             class1, class2 = self.plot_2D_calculations(model1, self.model.x, self.model.y)
        if plot_test_data:
             class3, class4 = self.plot_2D_calculations(model1, self.model.x_calculated, self.model.y_calculated)

        plt.style.use('dark_background')

        fig = plt.figure(figsize=(8, 6))

        plt.title(self.title,
                  fontdict=res.Resources.title_dict)


        if plot_train_data:

            plt.scatter(class1[0], class1[1], color=self.train_class0_color, alpha=self.train_alpha, label = self.train_class0_label)
            plt.scatter(class2[0], class2[1], color=self.train_class1_color, alpha=self.train_alpha, label= self.train_class1_label)
        if plot_test_data:

            plt.scatter(class3[0], class3[1], color=self.test_class0_color, alpha=self.test_alpha, label=self.test_class0_label)
            plt.scatter(class4[0], class4[1], color=self.test_class1_color, alpha=self.test_alpha, label=self.test_class1_label)
        plt.legend()
        plt.xlabel(self.x_label,
                   fontdict=res.Resources.label_dict)
        plt.ylabel(self.y_label,
                   fontdict=res.Resources.label_dict)
        if not(save_fig_path=="dont"):
            plt.savefig(save_fig_path + "/2D" + self.title)

        plt.show()
















class Knn_plot():



    #Init method to set default values..................................................

    def __init__(self, model):
        self.model = model
        self.title = "K-nearest neighbours visulisation"
        self.background_color = (0, 0, 0)
        self.test_alpha = 0.6
        self.train_alpha = 0.9
        self.color_dict = res.Resources.get_color_dict()
        self.train_data_color = [key for key in sorted(self.color_dict)[:15]]
        self.test_data_color = [key for key in sorted(self.color_dict)[15:]]
        random.shuffle(self.train_data_color)
        random.shuffle(self.test_data_color)


    def set_colors(self, train_data_color, test_data_color):
        self.train_data_color = train_data_color
        self.test_data_color = test_data_color



    #Method to set the properties of the graph................................
    #Not necessary, because all the default values are already set......................................

    def set_plot_properties(self, train_alpha = 0.9,test_alpha = 0.6, background_color = (0, 0, 0),title = "K-nearest neighbours visulisation"):
        self.title = title
        self.background_color = background_color
        self.test_alpha = test_alpha
        self.train_alpha = train_alpha





    #method that does all the calculatons before plotting the graph.....................................................

    def plot_3D_calculations(self, model1, x, y):
        tsne_data = model1.fit_transform(x)
        classes_dict = {}

        for i in range(len(tsne_data)):
            if y[i] in classes_dict:
                classes_dict[y[i]].append(tsne_data[i])
            else:
                classes_dict[y[i]] = [tsne_data[i]]
        return classes_dict








    # Method to plot the 2d graph of the calculated values after apllying T-sne.......................................

    def plot_3D_visuals(self, plot_train_data =True, plot_test_data = True, save_fig_path="dont"):
        model1 = TSNE(n_components=3)

        if plot_train_data:
             train_classes_dict = self.plot_3D_calculations(model1, self.model.x_data, self.model.y_data)
        if plot_test_data:
             test_classes_dict = self.plot_3D_calculations(model1, self.model.x_classified, self.model.y_classified)

        plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 12))
        ax = plt.axes(projection='3d')
        ax.text2D(0.30, 0.98, self.title, transform=ax.transAxes)

        ax.set_facecolor(self.background_color)
        if plot_train_data:
            label_dist = 0.02
            ax.text2D(0.86, 0.98, "Train ", transform=ax.transAxes)
            for group in train_classes_dict:
                plotx = []
                ploty = []
                plotz = []
                for coordp in train_classes_dict[group]:
                    plotx.append(coordp[0])
                    ploty.append(coordp[1])
                    plotz.append(coordp[2])

                ax.text2D(0.86, 0.98-label_dist, group, transform=ax.transAxes, color=self.train_data_color[int(label_dist/0.02)%15-1])
                ax.scatter3D(plotx, ploty, plotz, alpha=self.train_alpha, color=self.train_data_color[int(label_dist/0.02)%15-1])
                label_dist+=0.02
        if plot_test_data:
            label_dist = 0.02
            ax.text2D(0.98, 0.98, "Test ", transform=ax.transAxes)
            for group in test_classes_dict:
                plotx = []
                ploty = []
                plotz = []
                for coordp in test_classes_dict[group]:
                    plotx.append(coordp[0])
                    ploty.append(coordp[1])
                    plotz.append(coordp[2])

                ax.text2D(0.98, 0.98-label_dist, group , transform=ax.transAxes, color=self.test_data_color[int(label_dist/0.02)%15-1])
                ax.scatter3D(plotx, ploty, plotz, alpha=self.train_alpha, color=self.test_data_color[int(label_dist/0.02)%15-1])
                label_dist+=0.02
        if not(save_fig_path=="dont"):
            plt.savefig(save_fig_path + "/3D" + self.title)

        plt.show()














class Knn_plot_2D():



    #Init method to set default values..................................................

    def __init__(self, model):
        self.model = model
        self.title = "K-nearest neighbours visulisation"
        self.background_color = (0, 0, 0)
        self.test_alpha = 0.6
        self.train_alpha = 0.9
        self.color_dict = res.Resources.get_color_dict()
        self.train_data_color = [key for key in sorted(self.color_dict)[:15]]
        self.test_data_color = [key for key in sorted(self.color_dict)[15:]]
        random.shuffle(self.train_data_color)
        random.shuffle(self.test_data_color)


    def set_colors(self, train_data_color, test_data_color):
        self.train_data_color = train_data_color
        self.test_data_color = test_data_color



    #Method to set the properties of the graph................................
    #Not necessary, because all the default values are already set......................................

    def set_plot_properties(self, train_alpha = 0.9,test_alpha = 0.6, background_color = (0, 0, 0),title = "K-nearest neighbours visulisation"):
        self.title = title
        self.background_color = background_color
        self.test_alpha = test_alpha
        self.train_alpha = train_alpha





    #method that does all the calculatons before plotting the graph.....................................................

    def plot_3D_calculations(self, model1, x, y):
        tsne_data = model1.fit_transform(x)
        classes_dict = {}

        for i in range(len(tsne_data)):
            if y[i] in classes_dict:
                classes_dict[y[i]].append(tsne_data[i])
            else:
                classes_dict[y[i]] = [tsne_data[i]]
        return classes_dict








    # Method to plot the 2d graph of the calculated values after apllying T-sne.......................................

    def plot_3D_visuals(self, plot_train_data =True, plot_test_data = True, save_fig_path="dont"):
        model1 = TSNE(n_components=2)

        if plot_train_data:
             train_classes_dict = self.plot_3D_calculations(model1, self.model.x_data, self.model.y_data)
        if plot_test_data:
             test_classes_dict = self.plot_3D_calculations(model1, self.model.x_classified, self.model.y_classified)

        plt.style.use('dark_background')

        fig = plt.figure(figsize=(12, 12))
        #ax = plt.axes(projection='3d')
        #ax.text2D(0.30, 0.98, self.title, transform=ax.transAxes)
        plt.title(self.title)

        #ax.set_facecolor(self.background_color)
        if plot_train_data:
            label_dist = 0.02
            #ax.text2D(0.86, 0.98, "Train ", transform=ax.transAxes)
            for group in train_classes_dict:
                plotx = []
                ploty = []
                plotz = []
                for coordp in train_classes_dict[group]:
                    plotx.append(coordp[0])
                    ploty.append(coordp[1])


                #ax.text2D(0.86, 0.98-label_dist, group, transform=ax.transAxes, color=self.train_data_color[int(label_dist/0.02)%15-1])

                plt.scatter(plotx, ploty, alpha=self.train_alpha, label="Train "+str(group), color=self.train_data_color[int(label_dist/0.02)%15-1])
                label_dist+=0.02
        if plot_test_data:
            label_dist = 0.02
            #ax.text2D(0.98, 0.98, "Test ", transform=ax.transAxes)
            for group in test_classes_dict:
                plotx = []
                ploty = []

                for coordp in test_classes_dict[group]:
                    plotx.append(coordp[0])
                    ploty.append(coordp[1])



                #ax.text2D(0.98, 0.98-label_dist, group , transform=ax.transAxes, color=self.test_data_color[int(label_dist/0.02)%15-1])
                plt.scatter(plotx, ploty, alpha=self.train_alpha, label="Test "+str(group),color=self.test_data_color[int(label_dist/0.02)%15-1])
                label_dist+=0.02
        plt.legend()
        if not(save_fig_path=="dont"):
            plt.savefig(save_fig_path + "/3D" + self.title)

        plt.show()













class plot_parallel_plot():

    def __init__(self, x, y, plot_labels=[], class_color=[], save_fig_path='dont'):
        self.y = np.array(y)
        self.x = np.array(x)
        self.x_normalized = []
        self.plot_labels = []
        self.class_color =[]
        self.x_label = "Parameters/Dimensions"
        self.y_label = "Normalized Values"
        self.title = "Parallel Plot of Dataset"
        self.aplha = 0.5
        self.color_dict = res.Resources.get_color_dict()
        self.scatter_color = [key for key in sorted(self.color_dict)]
        self.colorList_len = len(self.scatter_color)
        random.shuffle(self.scatter_color)
        self.plot_labels = plot_labels
        self.class_color = class_color

        self.plot_plot(self.x, self.y, save_fig_path)


    def set_plot_properties(self, title="Parallel Plot of Dataset", x_label="Parameters/Dimensions", y_label="Normalized Values", alpha=0.5):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.aplha = alpha




    def plot_plot(self, x, y, save_fig_path='dont'):
        y_classes = np.unique(y)
        y_class_count = len(y_classes)
        self.x_normalized = self.normalize(x)

        plt.style.use('dark_background')
        x_pos = 1
        index = 0
        temp_feature = {}#[[]]*y_class_count
        #print(temp_feature)
        for feature in self.x_normalized:

            temp_feature = {}# [[]] * y_class_count
            for i in range(len(feature)):
                if y[i] in temp_feature:
                    temp_feature[y[i]].append(feature[i])
                else:
                    temp_feature[y[i]] = [feature[i]]
            index = 0
            for key in temp_feature:

                if len(self.class_color) == 0:
                    plt.scatter([x_pos]*len(temp_feature[key]), temp_feature[key], alpha=self.aplha,color=self.scatter_color[index%self.colorList_len])
                else:
                    plt.scatter([x_pos] * len(temp_feature[key]), temp_feature[key], alpha=self.aplha,
                                color=self.class_color[index % len(self.class_color)])

                index += 1
            x_pos += 1
            #print(temp_feature)

        index=0
        for key in temp_feature:

            if len(self.plot_labels) == 0 and len(self.class_color)==0:
                plt.scatter([], [], label="class "+str(key), color=self.scatter_color[index%self.colorList_len])
            elif len(self.plot_labels) == 0:
                plt.scatter([], [], label="class "+str(key), color=self.class_color[index % len(self.class_color)])

            elif len(self.class_color) == 0 :
                plt.scatter([], [], label=str(self.plot_labels[index % len(self.plot_labels)]), color=self.scatter_color[index % self.colorList_len])

            else:
                plt.scatter([], [], label=str(self.plot_labels[index%len(self.plot_labels)]), color=self.class_color[index % len(self.class_color)])
            index+=1

        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.title(self.title)
        plt.legend()
        if not(save_fig_path=="dont"):
            plt.savefig(save_fig_path + "/3D" + self.title)
        plt.show()





    def normalize(self, x):
        x = x.T
        temp_sum = 0
        x_normalized = []
        for i, feature in enumerate(x):
             x_normalized.append((feature/np.sum(feature)))
             #print(list(x_normalized[i]))

        return x_normalized




















class ninja_technique():

    def __init__(self):

        self.x_coords=[]
        self.y_oords=[]
        self.x_coords1 = []
        self.y_oords1 = []
        self.normalize_x = []
        self.tita = 0

    def calculate_normalisation(self, x):

        self.normalize_x = [0 for x in range(len(x[0]))]

        for i in x:
            for index,j in enumerate(i):
                self.normalize_x[index] += j

        #print(self.normalize_x)




    def calculate(self,model, x_data, y_data):
        self.tita = 90/(len(model.weights)-1)
        self.calculate_normalisation(x_data)
        print(self.tita, len(model.weights))

        for index,i in enumerate(x_data):
            temp_x = 0#model.bias
            temp_y = 0#model.bias
            temp_tita = 0
            for j in range(len(i)):
                temp_x = temp_x + (model.weights[j]*i[j]*math.cos(math.radians(temp_tita)))/self.normalize_x[j]
                temp_y = temp_y + (model.weights[j]*i[j]*math.sin(math.radians(temp_tita)))/self.normalize_x[j]

                #temp_x = temp_x + (i[j] * math.cos(math.radians(temp_tita)))/self.normalize_x[j]
                #temp_y = temp_y + (i[j] * math.sin(math.radians(temp_tita)))/self.normalize_x[j]

                temp_tita+=self.tita


            if y_data[index] == 0:
                self.x_coords.append(temp_x)
                self.y_oords.append(temp_y)
            else:
                self.x_coords1.append(temp_x)
                self.y_oords1.append(temp_y)







    def plot(self):
        plt.style.use('dark_background')
        plt.title("Swiss roll dataset plot")
        plt.scatter(self.x_coords, self.y_oords, label="class 0", color="#800080", alpha=0.3)
        plt.scatter(self.x_coords1, self.y_oords1, label="class 1", color='#FF8C00', alpha=0.3)
        plt.legend()
        plt.show()

    def plot_multi(self ,x1, x2, x3):
        plt.style.use('dark_background')
        plt.title("Wine dataset plot")
        plt.scatter(x1[0], x1[1], label="class 0", color="#800080", alpha=0.3)
        plt.scatter(x2[0], x2[1], label="class 1", color='#FF8C00', alpha=0.3)
        plt.scatter(x3[0], x3[1], label="class 2", color='#00FFFF', alpha=0.3)
        plt.legend()
        plt.show()
















"""
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


lor = lore.LogisticRegression(learning_rate=0.01, n_iters=10000)
lor.best_fit(x_train, y_train)
lor.best_fit(x, y)
y_pred = lor.predict(x)
prediction = lor.predict(x_test)

accuracy = np.sum(prediction == y_test)/ len(prediction)
print(prediction)
print(list(y_test))
print(accuracy)



plotly = ninja_technique()
plotly1 = ninja_technique()
#plotly.calculate(lor, x_train + x_test, y_train + y_test)
plotly.calculate(lor, x, y_pred)
plotly.plot()
#plotly1.calculate(lor, x_test, y_test)
#plotly1.plot()


datav = Logistic_regression_plot2D(lor)
datav.set_plot_properties(train_class0_label="benign", train_class1_label="malignant")
datav.plot_2D_visuals(plot_test_data=False)



"""



"""




data = sklearn.datasets.make_swiss_roll(n_samples=5000, noise=0.2, random_state=None)
ldata_x = data[0]
ldata_y = []
#print(data)
#for i in data:
    #print(i)
data = np.array(data[0]).T
plt.style.use('dark_background')
#print(data)

data1 = [[], [], []]
data2 = [[], [], []]

error = [-4, -3, -5]

for i in range(len(data[0])):
    if data[0][i] <= (7 ) and (data[2][i]>-6+ random.random()*random.choice(error) and data[2][i]<10):#- random.random()*random.choice(error)):
        data1[0].append(data[0][i])
        data1[1].append(data[1][i])
        data1[2].append(data[2][i])
        ldata_y.append(0)
    else:
        data2[0].append(data[0][i])
        data2[1].append(data[1][i])
        data2[2].append(data[2][i])
        ldata_y.append(1)



lor1 = lore.LogisticRegression(learning_rate=0.03,n_iters=100000)
lor1.best_fit(ldata_x, ldata_y)
y_data = lor1.predict(ldata_x)
count = 0

for i in range(len(ldata_y)):
    if ldata_y[i] == y_data[i]:
        count+=1

new_data = data1 + data2
data11= []
data12 = []




print("Accuracy = " + str(count/len(ldata_y) * 100))
#plot = Logistic_regression_plot2D(lor1)
#plot.set_plot_properties(title="Swiss Roll dataset")
#plot.plot_2D_visuals()
nj = ninja_technique()
nj.calculate(lor1, ldata_x, ldata_y)
nj.plot()



print(data1[0])
print(data1[1])
print(data1[2])
print(data2)




fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.scatter3D(data1[0], data1[1], data1[2], color="#FF4500", alpha=0.6)
ax.scatter3D(data2[0], data2[1], data2[2], color="#800080", alpha=0.6)
plt.show()

"""






"""

data = sklearn.datasets.make_swiss_roll(n_samples=5000, noise=0.0, random_state=None)
ldata_x = []
ldata_y = []
#print(data)
#for i in data:
    #print(i)
data = np.array(data[0]).T
plt.style.use('dark_background')
#print(data)

data1 = [[], [], []]
data2 = [[], [], []]

error = [-4, -3, -5]

for i in range(len(data[0])):

    if data[0][i] > 2:
        ldata_x.append([data[0][i], data[1][i], data[2][i]])

        if data[0][i] <= (7 ) and (data[2][i]>-6+ random.random()*random.choice(error) and data[2][i]<10):#- random.random()*random.choice(error)):
            data1[0].append(data[0][i])
            data1[1].append(data[1][i])
            data1[2].append(data[2][i])
            ldata_y.append(0)
        else:
            data2[0].append(data[0][i])
            data2[1].append(data[1][i])
            data2[2].append(data[2][i])
            ldata_y.append(1)




lor1 = lore.LogisticRegression(n_iters=100000)
lor1.best_fit(ldata_x, ldata_y)
y_data = lor1.predict(ldata_x)
count = 0
for i in range(len(ldata_y)):
    if ldata_y[i] == y_data[i]:
        count+=1


print("Accuracy = " + str(count/len(ldata_y) * 100))
plot = Logistic_regression_plot2D(lor1)
plot.set_plot_properties(title="Swiss Roll dataset")
plot.plot_2D_visuals()
nj = ninja_technique()
nj.calculate(lor1, ldata_x, ldata_y)
nj.plot()




fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.scatter3D(data1[0], data1[1], data1[2], color="#800080", alpha=0.6)
ax.scatter3D(data2[0], data2[1], data2[2], color="#FF4500", alpha=0.6)
plt.show()

"""











"""
data = sklearn.datasets.make_s_curve(n_samples=10000, noise=0.0, random_state=None)

data1 = [[], [], []]
data2 = [[], [], []]
y_values = []
x_data = []
#data = np.array(data[0]).T
#print(data[0])
#error = [-0.25, 0.25]
error = [0, 0]
for i in data[0]:
    if (-1.1<i[2] and i[2]<1.1 and i[0]+i[2]<1.5 and i[0]+i[2]>-1.5 ):
        x_data.append(i)
        if i[0] > 0 + random.random()*random.choice(error):
            data1[0].append(i[2]*100)
            data1[1].append(i[1]*100)
            data1[2].append(i[0]*100)
            y_values.append(1)
        else:
            data2[0].append(i[2]*100)
            data2[1].append(i[1]*100)
            data2[2].append(i[0]*100)
            y_values.append(0)
data1 = np.array(data1)
data2 = np.array(data2)

lor1 = lore.LogisticRegression(learning_rate=0.3, n_iters=100000)
lor1.best_fit(x_data, y_values)
y_data = lor1.predict(x_data, decision_boundary=0.1)
count = 0
print(len(y_values))
print(y_data)
for i in range(len(y_values)):
    if y_values[i] == y_data[i]:
        count+=1

print("Accuracy = "+ str(count*100/len(y_values)))



plot = Logistic_regression_plot2D(lor1)
plot.set_plot_properties(title="Swiss Roll dataset")
plot.plot_2D_visuals()

 ninja_technique()
nj.calculate(lor1, x_data, y_data)
nj.plot()


plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.scatter3D(data1[0], data1[1], data1[2], alpha=0.6, color="#800080")
ax.scatter3D(data2[0], data2[1], data2[2],alpha=0.6, color="#FF4500")

plt.show()


"""





"""

bc = datasets.load_iris()
x, y = bc.data, bc.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

#print(x)
print(y)


lr1 = sklearn.linear_model.LogisticRegression(max_iter=1000000)
lr1.fit(x, y)



y1 = lr1.predict(x)
count = 0
for i, j in enumerate(y):
    if j == y1[i]:
        count+=1
    else:
        print("Class " + str(j) + " " + str(y1[i]))
print(y1)
print("Accuracy = " + str(count/len(y) * 100))

x1 = [[], []]
x2 = [[], []]
x3 = [[], []]

weight = lr1.coef_
normalize = [0, 0, 0, 0]
print(weight)

for i in x:
    normalize[0] += i[0]
    normalize[1] += i[1]
    normalize[2] += i[2]
    normalize[3] += i[3]

for i, j in enumerate(x):

    alpha = 90/(len(x[0])-1)
    temp_y =0
    temp_x = 0
    temp_tita = 0
    for k in range(len(j)):
        temp_x = temp_x + (weight[int(y[i])][k] * j[k] * math.cos(math.radians(temp_tita))) / normalize[k]
        temp_y = temp_y + (weight[int(y[i])][k] * j[k] * math.sin(math.radians(temp_tita))) / normalize[k]
        #temp_x = temp_x + (j[k] * math.cos(math.radians(temp_tita))) / normalize[k]
        #temp_y = temp_y + (j[k] * math.sin(math.radians(temp_tita))) / normalize[k]
        temp_tita +=alpha

    if y1[i] == 0:
        x1[0].append(temp_x)
        x1[1].append(temp_y)
    elif y1[i] ==1:
        x2[0].append(temp_x)
        x2[1].append(temp_y)
    else:
        x3[0].append(temp_x)
        x3[1].append(temp_y)

print(x1,x2,x3)
nj = ninja_technique()
nj.plot_multi(x1, x2, x3)


print(lr1.coef_)



model1 = TSNE(n_components=2)

tsne_data = model1.fit_transform(x)
classes_dict = {}

for i in range(len(tsne_data)):
    if y[i] in classes_dict:
        classes_dict[y[i]].append(tsne_data[i])
    else:
        classes_dict[y[i]] = [tsne_data[i]]

print(classes_dict.keys())

x11 = np.array(classes_dict[0]).T
x21 = np.array(classes_dict[1]).T
x31 = np.array(classes_dict[2]).T

nj.plot_multi(x11, x21, x31)
"""










"""
bc = datasets.load_wine()
x, y = bc.data, bc.target

print(x)
print(y)


lr1 = sklearn.linear_model.LogisticRegression(max_iter=1000000)
lr1.fit(x, y)



y1 = lr1.predict(x)
count = 0
for i, j in enumerate(y):
    if j == y1[i]:
        count+=1
    else:
        print("Class " + str(j) + " " + str(y1[i]))
print(y1)
print("Accuracy = " + str(count/len(y) * 100))

x1 = [[], []]
x2 = [[], []]
x3 = [[], []]

weight = lr1.coef_
normalize = [0 for i in range(len(x[0]))]
print(weight)

for i in x:
    for j in range(len(i)):
        normalize[j] += i[j]


for i, j in enumerate(x):

    alpha = 90/(len(x[0])-1)
    temp_y =0
    temp_x = 0
    temp_tita = 0
    for k in range(len(j)):
        temp_x = temp_x + (weight[int(y[i])][k] * j[k] * math.cos(math.radians(temp_tita))) / normalize[k]
        temp_y = temp_y + (weight[int(y[i])][k] * j[k] * math.sin(math.radians(temp_tita))) / normalize[k]
        #temp_x = temp_x + (j[k] * math.cos(math.radians(temp_tita))) / normalize[k]
        #temp_y = temp_y + (j[k] * math.sin(math.radians(temp_tita))) / normalize[k]
        temp_tita +=alpha

    if y[i] == 0:
        x1[0].append(temp_x)
        x1[1].append(temp_y)
    elif y[i] ==1:
        x2[0].append(temp_x)
        x2[1].append(temp_y)
    else:
        x3[0].append(temp_x)
        x3[1].append(temp_y)

print(x1,x2,x3)
nj = ninja_technique()
nj.plot_multi(x1, x2, x3)


print(lr1.coef_)



model1 = TSNE(n_components=2)

tsne_data = model1.fit_transform(x)
classes_dict = {}

for i in range(len(tsne_data)):
    if y[i] in classes_dict:
        classes_dict[y[i]].append(tsne_data[i])
    else:
        classes_dict[y[i]] = [tsne_data[i]]

print(classes_dict.keys())

x11 = np.array(classes_dict[0]).T
x21 = np.array(classes_dict[1]).T
x31 = np.array(classes_dict[2]).T

nj.plot_multi(x11, x21, x31)


"""

"""

bc = datasets.load_iris()
x, y = bc.data, bc.target


plt1 = plot_parallel_plot(x, y, class_color=["Red", "Blue", "Green"])
"""
