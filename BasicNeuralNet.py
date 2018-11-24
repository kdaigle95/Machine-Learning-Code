import numpy as np 

#fast sigmoid function
def sigmoid(x, deriv = False):
	if(deriv == True):
		return x * (1-x)
	else:
		return 1 / (1 + np.exp(-x))

#Rectified Linear Unit activation
#def relu(x):
#	np.maximum(x, 0, x)

#input
trainingInput = np.array([[12, 26, 2, 83], [14, 24, 3, 78], [13, 26, 2, 83], [14, 23, 1, 60], [12, 24, 2, 78], [18, 25, 0, 72], 
          [ 9, 17, 2, 78], [ 8, 10, 1, 78], [ 7,  8, 1, 78],                  [12, 17, 0, 76], [10, 16, 0, 68]])

#output
trainingOutput = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

np.random.seed(1)

#random weights
syn0 = 2 * np.random.random((4, 3)) - 1
syn1 = 2 * np.random.random((3, 3)) - 1
syn2 = 2 * np.random.random((3, 1)) - 1

#training step
for i in range(60000):

	#forward
	layer0 = trainingInput
	layer1 = sigmoid(np.dot(layer0, syn0))
	layer2 = sigmoid(np.dot(layer1, syn1))
	layer3 = sigmoid(np.dot(layer2, syn2))

    #back propogation step
	#layer 2 error calculation
	layer3_error = trainingOutput - layer3

	#calculate layer2 delta by multiplying error rate 
	#with slope of the sigmoid 
	layer3_delta = layer3_error * sigmoid(layer3, True)

    #same as bove, but for layer 1
	layer2_error = layer3_delta.dot(syn2.T)
	#print("l1 error reached")

	layer2_delta = layer2_error * sigmoid(layer2, True)
	#print("l1 delta reached")

	layer1_error = layer2_delta.dot(syn1.T)

	layer1_delta = layer1_error * sigmoid(layer1, True)



	#update weights
	syn0 += np.dot(layer0.T, layer1_delta)
	#print("syn0 updated")
	syn1 += np.dot(layer1.T, layer2_delta)
	#print("syn1 updated")
	syn2 += np.dot(layer2.T, layer3_delta)


print("Trained output: ")
print(layer3)

testingInput = ([[8, 14, 1, 82],
				 [10, 13, 2, 82],
				 [14, 25, 3, 67]])

layer0 = testingInput
layer1 = sigmoid(np.dot(layer0, syn0))
layer2 = sigmoid(np.dot(layer1, syn1))
layer3 = sigmoid(np.dot(layer2, syn2))

print("\nTesting output: ")
print(layer3)

