import numpy as np 
import matplotlib.pyplot as plt 


def coefficients(a,b):
	coeff=a*b-len(a)*a.mean()*b.mean()
	return (coeff.sum())

def plot_reg(a,b,B_0,B_1):
	## plotting the initial points 
	plt.scatter(a,b)
	## plotting the predicted line
	plt.plot(a,B_0+B_1*a)
	plt.show()


def main():

	## given X and Y values for regression 
	n=int(input("Enter the number  of values that are given"))
	X,Y=[],[]
	for i in range(n):
		x=int(input("Enter the x : "))
		X.append(x)
	for i in range(n):
		y=int(input("Enter the x[%s] : "%i))
		Y.append(y)
		
	X=np.array(X)
	Y=np.array(Y)

	##X=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	##Y=np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

	SS_xx=coefficients(X,X)
	SS_xy=coefficients(X,Y)
	
	## coefficients of y=B_0+B_1*X

	B_1=SS_xy/SS_xx
	B_0=Y.mean()-B_1*X.mean()
	print ("B_1=",B_1)
	print ("B_0=",B_0)

	## plotting the graph
	plot_reg(X,Y,B_0,B_1)
	

if __name__=="__main__":
	main()

	
	



