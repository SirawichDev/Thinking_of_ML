from sklearn.linear_model import  LogisticRegression
import numpy as np

dataset = [[-3.215, 0],
                [-1.234,0],
                [0.974,0],
                [3.231,1],
                [1.2345,1],
                [4.5122,1]]

x = np.array(dataset)[:, 0:1]
y = np.array(dataset)[:,1] #ค่าจริง
#print(x)
LR = LogisticRegression(C=1.0 , penalty='l2',tol=0.0001)
LR.fit(x,y)
y_predict = LR.predict(x)
print("right side is Predict_y \n",np.column_stack((y,y_predict)))
#print(LR.predict(x))
