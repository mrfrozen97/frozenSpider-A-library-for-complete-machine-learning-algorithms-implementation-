

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




#LOR TSNE
"""


model = TSNE(n_components=2)
tsne_data = model.fit_transform(x_train)

#tsne_data = np.array(tsne_data).T
class1 = []
class2 = []


for i in range(len(y_train)):
    if y_train[i] == 0:
        class1.append(tsne_data[i])
    else:
        class2.append(tsne_data[i])

class1 = np.array(class1).T
class2 = np.array(class2).T

plt.scatter(class1[0], class1[1], color="#FFA500")
plt.scatter(class2[0], class2[1], color="#800080")
plt.show()





tsne_data = model.fit_transform(x_test)

#tsne_data = np.array(tsne_data).T
class1 = []
class2 = []


for i in range(len(lor.y_calculated)):
    if lor.y_calculated[i] == 0:
        class1.append(x_test[i])
    else:
        class2.append(x_test[i])

class1 = np.array(class1).T
class2 = np.array(class2).T

plt.scatter(class1[0], class1[1], color="#FFA500", alpha=0.6)
plt.scatter(class2[0], class2[1], color="#800080", alpha=0.6)
plt.show()




model1 = TSNE(n_components=3)
tsne_data = model1.fit_transform(x_train)

#tsne_data = np.array(tsne_data).T
class1 = []
class2 = []


for i in range(len(y_train)):
    if y_train[i] == 0:
        class1.append(tsne_data[i])
    else:
        class2.append(tsne_data[i])

class1 = np.array(class1).T
class2 = np.array(class2).T




fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_facecolor((0, 0, 0))
ax.scatter3D(class1[0], class1[1], class1[2], color="#800080", alpha=0.9)
ax.scatter3D(class2[0], class2[1], class2[2], color="#FFA500", alpha=0.9)
plt.show()




"""