from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

# print(mnist)
# print(len(mnist['data'])) 

x , y = mnist['data'], mnist['target']
print(x.shape) #70000 row กับ 784 pixel

# print(x[69999])
# print("result is : ", y[69999])
# print(np.where( y == 2 )) #show where nu,bbber 2 is
# show = x[12665] # 2
# print(y[12665]) #2
# image =   show.reshape(28,28) #28*28 = 784 #สูง * กว้าง

# plt.imshow(image)
# plt.show()
num_split = 60000

x_train, x_test, y_train, y_test = x[:num_split],x[num_split:],y[:num_split], y[num_split:]

print(x_train)

