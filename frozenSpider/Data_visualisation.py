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
        self.color_dict = {"Red": "#FF0000", "Orange": "#FFA500", "Yellow": "#FFFF00", "DeepPink": "#FF1493",
                           "LightPink": "#FFB6C1", "LightPink": "#FFB6C1"
            , "Pink": "#FFC0CB", "Lavender": "#E6E6FA", "Orchid": "#DA70D6", "Violet": "#EE82EE",
                           "DarkOrchid": "#9932CC", "DarkViolet": "#9400D3"
            , "BlueViolet": "#8A2BE2", "Purple": "#800080", "Purple": "#800080", "Indigo": "#4B0082",
                           "Salmon": "#FA8072", "Crimson": "#DC143C", "DarkRed": "#8B0000",
                           "DarkOrange": "#FF8C00", "Coral": "#FF7F50", "OrangeRed": "#FF4500", "Gold": "#FFD700",
                           "GreenYellow": "#ADFF2F", "Lime": "#00FF00", "PaleGreen": "#98FB98"
            , "SpringGreen": "#00FF7F", "Green": "#008000", "LightSeaGreen": "#20B2AA", "Cyan": "#00FFFF",
                           "Aquamarine": "#7FFFD4", "SkyBlue": "#87CEEB",
                           "DeepSkyBlue": "#00BFFF", "Blue": "#0000FF", "MediumBlue": "#0000CD", "Navy": "#000080",
                           "Black": "#000000", "Gray": "#808080"}






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
        self.color_dict = {"Red": "#FF0000", "Orange": "#FFA500", "Yellow": "#FFFF00", "DeepPink": "#FF1493",
                           "LightPink": "#FFB6C1", "LightPink": "#FFB6C1"
            , "Pink": "#FFC0CB", "Lavender": "#E6E6FA", "Orchid": "#DA70D6", "Violet": "#EE82EE",
                           "DarkOrchid": "#9932CC", "DarkViolet": "#9400D3"
            , "BlueViolet": "#8A2BE2", "Purple": "#800080", "Purple": "#800080", "Indigo": "#4B0082",
                           "Salmon": "#FA8072", "Crimson": "#DC143C", "DarkRed": "#8B0000",
                           "DarkOrange": "#FF8C00", "Coral": "#FF7F50", "OrangeRed": "#FF4500", "Gold": "#FFD700",
                           "GreenYellow": "#ADFF2F", "Lime": "#00FF00", "PaleGreen": "#98FB98"
            , "SpringGreen": "#00FF7F", "Green": "#008000", "LightSeaGreen": "#20B2AA", "Cyan": "#00FFFF",
                           "Aquamarine": "#7FFFD4", "SkyBlue": "#87CEEB",
                           "DeepSkyBlue": "#00BFFF", "Blue": "#0000FF", "MediumBlue": "#0000CD", "Navy": "#000080",
                           "Black": "#000000", "Gray": "#808080"}





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

        fig = plt.figure(figsize=(12, 12))

        plt.title(self.title, fontdict={"fontsize": 20})


        if plot_train_data:

            plt.scatter(class1[0], class1[1], color=self.train_class0_color, alpha=self.train_alpha, label = self.train_class0_label)
            plt.scatter(class2[0], class2[1], color=self.train_class1_color, alpha=self.train_alpha, label= self.train_class1_label)
        if plot_test_data:

            plt.scatter(class3[0], class3[1], color=self.test_class0_color, alpha=self.test_alpha, label=self.test_class0_label)
            plt.scatter(class4[0], class4[1], color=self.test_class1_color, alpha=self.test_alpha, label=self.test_class1_label)
        plt.legend()
        plt.xlabel(self.x_label, fontdict={"fontsize": 15})
        plt.ylabel(self.y_label, fontdict={"fontsize": 15})
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
        self.color_dict = {"Red": "#FF0000", "Orange": "#FFA500", "Yellow": "#FFFF00", "DeepPink": "#FF1493",
                           "LightPink": "#FFB6C1", "LightPink": "#FFB6C1"
            , "Pink": "#FFC0CB", "Lavender": "#E6E6FA", "Orchid": "#DA70D6", "Violet": "#EE82EE",
                           "DarkOrchid": "#9932CC", "DarkViolet": "#9400D3"
            , "BlueViolet": "#8A2BE2", "Purple": "#800080", "Purple": "#800080", "Indigo": "#4B0082",
                           "Salmon": "#FA8072", "Crimson": "#DC143C", "DarkRed": "#8B0000",
                           "DarkOrange": "#FF8C00", "Coral": "#FF7F50", "OrangeRed": "#FF4500", "Gold": "#FFD700",
                           "GreenYellow": "#ADFF2F", "Lime": "#00FF00", "PaleGreen": "#98FB98"
            , "SpringGreen": "#00FF7F", "Green": "#008000", "LightSeaGreen": "#20B2AA", "Cyan": "#00FFFF",
                           "Aquamarine": "#7FFFD4", "SkyBlue": "#87CEEB",
                           "DeepSkyBlue": "#00BFFF", "Blue": "#0000FF", "MediumBlue": "#0000CD", "Navy": "#000080",
                           "Black": "#000000", "Gray": "#808080"}
        self.train_data_color = [key for key in sorted(self.color_dict)[:15]]
        self.test_data_color = [key for key in sorted(self.color_dict)[15:]]
        random.shuffle(self.train_data_color)
        random.shuffle(self.test_data_color)






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
















