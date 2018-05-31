from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

print(mnist)
print(len(mnist['data'])) 

x , y = mnist['data'], mnist['target']

# print(x[69999])
# print("result is : ", y[69999])



