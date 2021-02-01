from frozenSpider import spiderAlgorithmLoR as lor
from frozenSpider import Data_visualisation as dv
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
import numpy as np

dataset = ds.load_breast_cancer()

x, y = dataset.data, dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)


model = lor.LogisticRegression(learning_rate=0.01)
model.best_fit(x_train, y_train)

prediction = model.predict(x_test)

accuracy = np.sum(prediction == y_test)/ len(prediction)

print(accuracy)

plot = lor.plot_model(model)
plot.set_marker_properties()




#All the below properties are set default. So you need to change them, only when you want to change it...................................
plot.set_marker_properties(
                              label_default=True,
                              calculated_point_class1_label = "Calculated class 1",
                              calculated_point_class0_label="Calculated class 0",
                              calculated_point_alpha = 0.5,
                              calculated_point_class1_color='red',
                              calculated_point_class0_color="cyan",
                              title_size=15, xlabel_size=10,
                              ylabel_size=10, title="Output vs dimention",
                              y_label="y coordinates",
                              title_color="#FF0000",
                              xlabel_color="#663399",
                              ylabel_color="#663399",
                              )







#ALl the parameters are already defined with default values so not required to reset if not required.................
plot.plot_model(
                display_graph=False,
                class1_color='purple',
                class0_color='orange',
                alpha=0.6,
                save_fig_file='dont',
                display_calculated_points = True,
                #save_fig_file=""                                       # Pass string of the path
                )



datav = dv.Logistic_regression_plot(model)
datav.plot_3D_visuals(plot_test_data=True)

datav = dv.Logistic_regression_plot2D(model)
datav.plot_2D_visuals(plot_test_data=True)





