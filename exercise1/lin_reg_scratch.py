import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import math

#Since we are implementing linear regression we will take theta as 2X1 matrix.
theta = np.zeros((2,1))

#Creating a data frame using the values of x and y coordinates in the file.
df = pd.read_csv('ex1data1.txt', header = None, names = ['x','y'])
#Inserting the points obtained in the data frame into x,y arrays
x = np.array(df.x)
y = np.array(df.y)
#Number of examples in the vector x
m=x.size
#Inserting one more column into the vector x which will be the coefficient of (x^0)->(theta0)
x=np.c_[np.ones((m,1)),x]

for i in range(m):
	print x[i][0]," ",x[i][1]," ",y[i]	

#Since number of features=1
'''
if class is not used use below part
#feature_count=1
#theta=np.zeros((feature_count+1,1))	
iterations=1000
#alpha is the learning rate
alpha=0.01
'''
#plotting the training points in the cartesian coordinates
class LinearRegression():
	def __init__(feature_count):
		#if theta vector has to be initialised to 0
		#self.theta=zeros((feature_count+1,1))
		#if theta vector has to be initialised to normally distributed values
		self.theta=np.random.normal(0,1,feature_count+1)
	def cost(theta,feature_count,alpha,):
		
		
		
def plot_points(x,y):
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in $10,000s')
	plt.scatter(x,y,marker='x')
	plt.show()

plot_points(x,y)
