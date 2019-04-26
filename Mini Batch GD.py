import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x, y, w):  # Hàm tính độ lỗi (cost function):  1/N * norm(Y - Y_predict, 2) ** 2
    N = x.shape[0]
    return 1/N * np.linalg.norm(y - x.dot(w), 2) ** 2

def grad_function(x, y, w):  # Hàm tính đạo hàm cost function: 2/N * X.T * (X.W - Y)
    N = x.shape[0]
    return 2/N * x.T.dot(x.dot(w) - y)


def mini_batch_gd_unnormalize_data(w_init, lrate, gamma, full_data):  # Mini-Batch Gradient Descent On Unnormalized Data
    v_old = np.zeros_like(w_init)
    loss = []
    cost = [0]
    w = [w_init]
    for epoch in range(10000):
        np.random.shuffle(full_data)  # Trộn dữ liệu để mang tính khách quan
        it = 0
        check = 0
        while True:
            if (it + 1) * 50 + 1 < full_data.shape[0]:
                X = full_data[it * 50: (it + 1) * 50 + 1].T[0:14].T
                Y = np.array([full_data[it * 50: (it + 1) * 50 + 1].T[-1]]).T
            else:
                X = full_data[it * 50: full_data.shape[0]].T[0:14].T
                Y = np.array([full_data[it * 50: full_data.shape[0]].T[-1]]).T
                check = 1
            v_new = gamma * v_old + lrate * grad_function(X, Y, w[-1] - gamma * v_old)  # Dùng kĩ thuật momentum và NAG
            w_new = w[-1] - v_new                                                       # để tối ưu độ chính xác
            loss.append(cost_function(X, Y, w_new))
            cost.append(cost_function(X, Y, w_new))
            if abs(cost[-1] - cost[-2]) < 0.00001:  # Điều kiện dừng
                return w_new, loss, epoch
            if cost[-2] - cost[-1] > 0.000001:  # Nếu độ lỗi giảm chậm thì tăng
                lrate *= 1.00001                # learning rate
            v_old = v_new
            w.append(w_new)
            if check == 1:
                break
            else:
                it += 1
    return w[-1], loss  # Trả về w


def mini_batch_gd_normalize_data(w_init, lrate, gamma, full_data):  # Mini-Batch Gradient Descent On Normalized Data
    v_old = np.zeros_like(w_init)
    loss = []
    cost = [0]
    w = [w_init]
    for epoch in range(10000):
        np.random.shuffle(full_data)  # Trộn dữ liệu để mang tính khách quan
        it = 0
        check = 0
        while True:
            if (it + 1) * 50 + 1 < full_data.shape[0]:
                X = full_data[it * 50: (it + 1) * 50 + 1].T[0:4].T
                Y = np.array([full_data[it * 50: (it + 1) * 50 + 1].T[-1]]).T
            else:
                X = full_data[it * 50: full_data.shape[0]].T[0:4].T
                Y = np.array([full_data[it * 50: full_data.shape[0]].T[-1]]).T
                check = 1
            v_new = gamma * v_old + lrate * grad_function(X, Y, w[-1] - gamma * v_old)  # Dùng kĩ thuật momentum và NAG
            w_new = w[-1] - v_new  # để tối ưu độ chính xác
            loss.append(cost_function(X, Y, w_new))
            cost.append(cost_function(X, Y, w_new))
            if abs(cost[-1] - cost[-2]) < 0.00001:  # Điều kiện dừng
                return w_new, loss, epoch
            if cost[-2] - cost[-1] > 0.000001:  # Nếu độ lỗi giảm chậm thì tăng
                lrate *= 1.00001                # learning rate
            v_old = v_new
            w.append(w_new)
            if check == 1:
                break
            else:
                it += 1
    return w[-1], loss  # Trả về w


# Input data set
data = pd.read_csv('housing.data.csv')

# UNNORMALIZED DATA

unnormalized_data = np.array(
    data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RED', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']])

#np.random.shuffle(unnormalized_data) # Trộn dữ liệu mỗi lần thử để đảm bảo tính khách quan

training_set_unnormalized_data = unnormalized_data[101:507]  # Tạo training set
testing_set_unnormalized_data = unnormalized_data[0:101]  # Tạo testing set

ones = np.array([np.ones_like(training_set_unnormalized_data.T[0])]).T
ones_and_training_set = np.concatenate((ones, training_set_unnormalized_data), axis=1)

X_unnormalized_data = ones_and_training_set.T[0:14].T  # X của training set
Y_unnormalized_data = np.array([ones_and_training_set.T[-1]]).T  # Y của training set

# Training
w_init = np.array([[0.6, -0.1, 0.05, 0, 0.5, 0.1, 5, 0.01, -1, 0.1, -0.01, -0.1, 0.01, -0.5]]).T
w_unnormalized_data, loss = mini_batch_gd_unnormalize_data(w_init, 0.0000001, 0.9,
                                                                  ones_and_training_set)  # Tính w, chọn learing
                                                                                          # rate ban đầu là 10^-7
                                                                                          # và hệ số gamma là 0.9

print("UNNORMALIZED DATA")
plt.plot(loss)
plt.show()

print("w[]:", end=' ')  # In các giá trị w
for i in range(w_unnormalized_data.shape[0]):
    print("w{0} = {1}".format(i, w_unnormalized_data[i][0]), end=' ')
print()

MSE_UN1 = 0
MSE_UN2 = 0
MAE_UN1 = 0
MAE_UN2 = 0

for i in range(training_set_unnormalized_data.shape[0]):
    y_predict = 0 + w_unnormalized_data[0][0]
    for j in range(0, training_set_unnormalized_data[i].shape[0] - 1):
        y_predict += training_set_unnormalized_data[i][j] * w_unnormalized_data[j + 1][0]
    MSE_UN1 += (training_set_unnormalized_data[i][-1] - y_predict) ** 2  # Tính MSE
    MAE_UN1 += abs(training_set_unnormalized_data[i][-1] - y_predict)  # Tính MAE
    #print("{0:.2f} = {1}".format(x_unnormalized_data,training_set_unnormalized_data[i][-1]))
print("MSE in training unnormalized data: {0:.2f}".format(MSE_UN1 / training_set_unnormalized_data.shape[0]))
print("MAE in training unnormalized data: {0:.2f}".format(MAE_UN1 / training_set_unnormalized_data.shape[0]))

print()
print("Ket qua du doan tren testing set:")
for i in range(testing_set_unnormalized_data.shape[0]):
    y_predict = 0 + w_unnormalized_data[0][0]
    for j in range(0, testing_set_unnormalized_data[i].shape[0] - 1):
        y_predict += testing_set_unnormalized_data[i][j] * w_unnormalized_data[j + 1][0]
    MSE_UN2 += (testing_set_unnormalized_data[i][-1] - y_predict) ** 2  # Tính MSE
    MAE_UN2 += abs(testing_set_unnormalized_data[i][-1] - y_predict)  # Tính MAE
    print("{0:.2f} = {1}".format(y_predict, testing_set_unnormalized_data[i][-1]))  # In các giá trị dự đoán
                                                                                    # trên testing set

print("MSE testing set unnormalized data: {0:.2f}".format(MSE_UN2 / testing_set_unnormalized_data.shape[0]))
print("MAE testing set unnormalized data: {0:.2f}".format(MAE_UN2 / testing_set_unnormalized_data.shape[0]))
print()

# NORMALIZED DATA

normalized_data = np.array(data[['RM', 'PTRATIO', 'LSTAT', 'MEDV']])  # Bộ dữ liệu normalized chỉ lấy 3 thuộc tính để
                                                                      # dự đoán MEDV là RM, PTRATIO, LSTAT
#np.random.shuffle(normalized_data) # Trộn dữ liệu mỗi lần thử để đảm bảo tính khách quan

training_set_normalized_data = normalized_data[101:507]  # Tạo training set
testing_set_normalized_data = normalized_data[0:101]  # Tạo testing set

ones = np.array([np.ones_like(training_set_normalized_data.T[0])]).T
ones_and_training_set = np.concatenate((ones, training_set_normalized_data), axis=1)

X_normalized_data = ones_and_training_set.T[0:4].T  # X của training set
Y_normalized_data = np.array([ones_and_training_set.T[-1]]).T  # Y của training set

# Training
w_init1 = np.array([[0.7, 5.5, -0.3, -0.5]]).T
w_normalized_data, loss = mini_batch_gd_normalize_data(w_init1, 0.0000001, 0.9,
                                                      ones_and_training_set)  # Tính w, chọn learing
                                                                              # rate ban đầu là 10^-7
                                                                              # và hệ số gamma là 0.9
plt.plot(loss)
plt.show()

print("NORMALIZED DATA")

print("w[]:", end=' ')  # In các giá trị w
for i in range(w_normalized_data.shape[0]):
    print("w{0} = {1}".format(i, w_normalized_data[i][0]), end=' ')
print()

MSE_N1 = 0
MSE_N2 = 0
MAE_N1 = 0
MAE_N2 = 0

for i in range(training_set_normalized_data.shape[0]):
    y_predict = 0 + w_normalized_data[0][0]
    for j in range(0, training_set_normalized_data[i].shape[0] - 1):
        y_predict += training_set_normalized_data[i][j] * w_normalized_data[j + 1][0]
    MSE_N1 += (training_set_normalized_data[i][-1] - y_predict) ** 2  # Tính MSE
    MAE_N1 += abs(training_set_normalized_data[i][-1] - y_predict)  # Tính MAE
    #print("{0:.2f} = {1}".format(x_normalized_data,training_set_normalized_data[i][-1]))
print("MSE training set normalized data: {0:.2f}".format(MSE_N1 / training_set_normalized_data.shape[0]))
print("MAE training set normalized data: {0:.2f}".format(MAE_N1 / training_set_normalized_data.shape[0]))

print()
print("Ket qua du doan tren testing set:")
for i in range(testing_set_normalized_data.shape[0]):
    y_predict = 0 + w_normalized_data[0][0]
    for j in range(0, testing_set_normalized_data[i].shape[0] - 1):
        y_predict += testing_set_normalized_data[i][j] * w_normalized_data[j + 1][0]
    MSE_N2 += (testing_set_normalized_data[i][-1] - y_predict) ** 2  # Tính MSE
    MAE_N2 += abs(testing_set_normalized_data[i][-1] - y_predict)  # Tính MAE
    print("{0:.2f} = {1}".format(y_predict, testing_set_normalized_data[i][-1]))  # In các giá trị dự đoán
                                                                                  # trên testing set
print("MSE testing set normalized data: {0:.2f}".format(MSE_N2 / testing_set_normalized_data.shape[0]))
print("MAE testing set normalized data: {0:.2f}".format(MAE_N2 / testing_set_normalized_data.shape[0]))
