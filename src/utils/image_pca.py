import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
from PIL import Image
import os
import pandas as pd

def loadImage(path):
    img = Image.open(path)
    img = img.convert("L")

    width = img.size[0]
    height = img.size[1]
    data = img.getdata()
    data = np.array(data).reshape(height,width)/100
    new_im = Image.fromarray(data*100)
    # new_im.show()
    return data


def error(data,recdata):
    sum1 = 0
    sum2 = 0
    D_value = data - recdata
    
    for i in range(data.shape[0]):
        sum1 += np.dot(data[i],data[i])
        sum2 += np.dot(D_value[i], D_value[i])
    error = sum2/sum1
    print(sum2, sum1, error)


root_tr = '../data/mnist/train/'
save_tr = '../data/mnist_pca/train/'
root_te = '../data/mnist/test/'
save_te = '../data/mnist_pca/test/'
num_list = sorted(os.listdir(root_tr))

for idx in num_list[0]:
    path_tr = os.path.join(root_tr + idx)
    list_img = sorted(os.listdir(path_tr))
    
    temp  = np.empty([1, 784])
    for i in range(len(list_img)):
        img_path = os.path.join(path_tr + '/'+list_img[i])
        save_path = os.path.join(save_tr + idx + '/' + list_img[i])
        img = Image.open(img_path)
        
        img = img.convert("L")
        data = np.array(img.getdata())
        temp  = np.concatenate((temp, data.reshape(1,-1)), axis=0)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(temp[1:,:])
    new = scaler.inverse_transform(X_std)

    pca = PCA(n_components=2).fit(X_std)
    x_new = pca.transform(X_std)
    recdata = pca.inverse_transform(x_new)
    new_recdata = scaler.inverse_transform(recdata)
    