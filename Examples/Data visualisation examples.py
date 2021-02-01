from frozenSpider import Data_visualisation as dv
from sklearn import datasets as ds


dataset = ds.load_breast_cancer()
x, y = dataset.data, dataset.target

dv.plot_parallel_plot(x, y,                                                        # Compulsory parameters

                      title="Breat cancer Dataset",                                # Optional parameters
                      x_label="Parameters",
                      y_label="Values",
                      alpha=0.4,
                      class_colors=['Red', 'Cyan'],                          # Class colors can be any color name There are 145 options
                      plot_labels=['0 class', '1 class'])




