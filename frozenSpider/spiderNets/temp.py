# Funtion to check if matrix is symmetric
def check_symmetrc(arr, n):
    for i in range(n):
        for j in range(i, n):
            if arr[i][j] != arr[j][i]:
                return False  # false if elements do not match

    return True  # True if elements match


def add_matrix(a, b):
    sum_mat = []
    if len(a) != len(b) and len(a[0]!=len(b[0])):
        print("Matrix cannot be added")

    else:
        for i in range(len(a)):
            temp_sum = []
            for j in range(len(a[i])):
                temp_sum.append(a[i][j] + b[i][j])
            sum_mat.append(temp_sum)

    return sum_mat


def transpose(l1):
    l2 = []
    # iterate over list l1 to the length of an item
    for i in range(len(l1[0])):
        # print(i)
        row = []
        for item in l1:
            row.append(item[i])
        l2.append(row)
    return l2


def multiply(a, b):
    b = transpose(b)
    if len(a) != len(b[0]):
        print("Matrix cannot be multiplied")

    else:
        mul_ans = []
        for i in range(len(a)):
            temp = []
            for j in range(len(b[0])):
                temp.append(sum([a[i][x]*b[j][x] for x in range(len(b[0]))]))
            mul_ans.append(temp)


    return mul_ans

# Number of rows and columns input
n = int(input("Enter number of rows: "))
m = int(input("Enter numbe of columns: "))

# Matrix elements input
print("Enter matrix elements")
arr = []  # list to store the matrix
for i in range(n):
    print("Enter row " + str(i) + ": ", end=" ")
    temp = [int(i) for i in input().split()]
    if len(temp) != m:  # Check for invalid number of inputs
        print("Invalid Input")
        n = m + 1
        break
    arr.append(temp)

# Condition for matix to be symmetric
if n == m:
    if check_symmetrc(arr, n):  # Check for matching elements
        print("It is a symmetric matrix")
    else:
        print("Matrix is not symmetric")
else:
    print("Matrix is not symmetric")




n = int(input("Enter number of rows: "))
m = int(input("Enter numbe of columns: "))

# Matrix elements input
print("Enter matrix elements")
arr1 = []  # list to store the matrix
for i in range(n):
    print("Enter row " + str(i) + ": ", end=" ")
    temp = [int(i) for i in input().split()]
    if len(temp) != m:  # Check for invalid number of inputs
        print("Invalid Input")
        n = m + 1
        break
    arr1.append(temp)


print("Sum is \n" + str(add_matrix(arr, arr1)))
print("Multiply is \n" + str(multiply(arr, arr1)))