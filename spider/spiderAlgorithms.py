

#Example code to use the linear regression of the library

"""

m = 0.43
c = -5.34
x = [i for i in range(1000)]
y = []
for i in x:
    y.append(m*i + c + (- 0.5 + random.random())*(min(int(i/3), 200)))

x1 = [104, 205, 625]

le1 = sp.linear_regression()
a,b =le1.bestFit(x, y)
le1.find_unknowns(x1)


print("Slope = " + str(b))
print("x intercept = " + str(a))
print("Squared error = "+ str(le1.squared_error()))


plot2 = sp.plot_model(le1)
plot2.set_marker_properties(unknown_points_size=80, unknown_points_alpha=0.6, unknown_points_color='Red')
plot2.plot_model(save_fig_path="plot2")


"""