import numpy as np
import matplotlib.pyplot as plt





class K_nearest_neighbours():

    def __init__(self, x_data, y_data, k):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.k = k
        self.x_classified = []
        self.y_classified = []




    def nearest_neighbour(self, distances_y, k):

        nearest = {}
        for i in range(k):
            if distances_y[i][1] in nearest:
                nearest[distances_y[i][1]] +=1
            else:
                nearest[distances_y[i][1]] = 1

        print(nearest)
        max_points = 0
        max_key = 0
        for key in nearest:
            if nearest[key]>max_points:
                max_points = nearest[key]
                max_key = key

        return max_points, max_key


    def classify(self,coordinates):

        self.x_classified = coordinates
        near_classes = []
        confidences = []


        for coords in coordinates:

            x_arr = []
            for i in range(len(self.x_data)):
                x_arr.append(coords)

            x_arr = np.array(x_arr)

            print(np.sum(np.square(x_arr-self.x_data), axis=1))

            distances = np.array(np.sqrt(np.sum(np.square(x_arr-self.x_data), axis=1)))
            distances_y = []

            for i in range(len(distances)):
                distances_y.append([distances[i], self.y_data[i]])
            distances_y = np.array(distances_y)
            distances_y.sort(axis=0)
            near_points, near_class = self.nearest_neighbour(distances_y, self.k)

            confidence = near_points/ self.k * 100

            near_classes.append(near_class)
            confidences.append(confidence)

            #print(near_points, near_class)
            #print(distances_y)
            #print(distances)

        self.y_classified = near_classes
        return near_classes, confidences













class plot_model():

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
        self.x_label = ["Dimention " + str(x) for x in range(1, 100)]
        self.y_label = ""
        self.xlabel_size = 15
        self.ylabel_size = 15
        self.title_size = 18
        self.model = model
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
        self.calculated_point_size = 10
        self.calculated_point_class0_label = "Calculated class 0"
        self.calculated_point_class1_label = "Calculated class 1"
        self.calculated_point_class0_color = "Red"
        self.calculated_point_class1_color = "Cyan"
        self.calculated_point_alpha = 0.5








    # Function to get dictionary to access the different options avaliable for the colors.........................

    def get_color_dict(self):
        return self.color_dict








    # function to set label of each dimention of the graph................................................

    def set_dimention_labels(self, dimention_label):
        self.x_label = dimention_label







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


        multi_dimentions = np.array(self.model.x_data)
        multi_dimentions_calculated = np.array(self.model.x_classified)
       # print(len(multi_dimentions[0]))
        print(multi_dimentions)

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
                print(i)
                if self.model.y_classified[i] in y_test_data:
                    y_test_data[self.model.y_classified[i]].append(a[dimention])
                else:
                    y_test_data[self.model.y_classified[i]] = [a[dimention]]
                #x_train_data.append(i[dimention])


            print(y_train_data)

            for group in y_train_data:
                print(group)
                y_plot_axis = []
                for i in range(len(y_train_data[group])):
                    y_plot_axis.append(group)
                plt.scatter(y_train_data[group], y_plot_axis, color=self.color_dict[class1_color],
                            label="class" + str(group) , alpha=alpha, zorder=3)                     #plot the best fit line



            if display_calculated_points:
                for group in y_test_data:
                    y_plot_axis = []
                    for i in range(len(y_test_data[group])):
                        y_plot_axis.append(group)

                    plt.scatter(y_test_data[group], y_plot_axis, color=self.color_dict[class0_color],
                          label="class 0", alpha=alpha, zorder=3)  # plot the best fit line

         


            plt.title(self.title, fontdict={"fontsize": 15}, color=self.title_color)
            plt.xlabel(self.x_label[dimention], fontdict={"fontsize": 15}, color=self.xlabel_color)
            plt.ylabel(self.y_label, fontdict={"fontsize": 15}, color=self.ylabel_color)





            plt.legend(loc = 'upper left')

            if self.showGrid:
                plt.grid(color='#cfd8dc', zorder=0)

            figure = plt.gcf()
            figure.set_size_inches(self.sizeX, self.sizeY)


            if not(save_fig_file=='dont') :

                  plt.savefig("./"+save_fig_file +"/"+self.x_label[dimention])#, bbox_inches='tight')
            if display_graph:
               plt.show()
            plt.close()








x = [[1, 1], [2, 2], [3, 3]]
y = [0, 0, 1]

a = K_nearest_neighbours(x, y, 3)
group, confidence = a.classify([[1.5, 1.5]])


pl = plot_model(a)
pl.plot_model()


print(group, confidence)










