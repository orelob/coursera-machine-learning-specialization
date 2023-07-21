import numpy as np

def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y


def load_data_multi():
    data = np.loadtxt("data/ex1data2.txt", delimiter=',')
    X = data[:,:2] # lấy tất cả các dòng, 2 cột đầu tiên -> các feature
    y = data[:,2] # lấy tất cả các dòng, cột cuối cùng -> giá trị đầu ra
    return X, y
