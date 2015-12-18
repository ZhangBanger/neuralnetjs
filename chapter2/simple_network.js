// Use ReLU nonlinearity
// input size 2
// 1 hidden layer of size 3
// output scalar
// a, b, c d are weights
// x, y are inputs
// n are neurons

// Load data
var data = []
var labels = []
// hardcoded from dataset file
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);

// initialize weights between -0.5 and 0.5
var a1, b1, c1 = Math.random() - 0.5 
var a2, b2, c2 = Math.random() - 0.5
var a3, b3, c3 = Math.random() - 0.5
var a4, b4, c4, d4 = Math.random() - 0.5

for (var iter = 0; iter < 400; iter++) {
	// sample
	var i = Math.floor(Math.random() * data.length)
	var x = data[i][0]
	var y = data[i][1]
	var label = labels[i]

	// forward pass

	// 3 neurons in 1st hidden layer
	// relu activation
	// c1, c2, c3 are bias
	var n1 = Math.max(0, a1*x + b1*y + c1)
	var n2 = Math.max(0, a2*x + b2*y + c2)
	var n3 = Math.max(0, a3*x + b3*y + c3)

	// a4, b4, c4 are weights from 3 hidden neurons to output scalar
	// d4 is bias
	var score = a4*n1 + b4*n2 + c4*n3 + d4

	// compute top pull
	var pull = 0.0
	if (label === 1 && score < 1) pull = 1
	if (label === -1 && score > -1) pull = -1

	// BEGIN BACKPROP

	//backprop through score layer
	var dscore = pull
	var da4 = n1 * dscore
	var dn1 = a4 * dscore
	var db4 = n2 * dscore
	var dn2 = b4 * dscore
	var dc4 = n3 * dscore
	var dn3 = c4 * dscore
	var dd4 = 1.0 * dscore

	// backprop through relu activation
	// set gradient to 0 if neuron did not fire (pass threshold 0)
	var dn3 = n3 === 0 ? 0 : dn3
	var dn2 = n2 === 0 ? 0 : dn2
	var dn1 = n1 === 0 ? 0 : dn1

	// backprop to parameters of neuron 1 (weights from input to n1)
	var da1 = x * dn1
	var db1 = y * dn1
	var dc1 = 1.0 * dn1

	// backprop n2
	var da2 = x * dn2
	var db2 = y * dn2
	var dc2 = 1.0 * dn2

	backprop n3
	var da3 = x * dn3
	var db3 = y * dn3
	var dc3 = 1.0 * dn3

	// END BACKPROP

	// add the pulls from the regularization, tugging all multiplicative
	// parameters (i.e. not the biases) downward, proportional to their value
	da1 += -a1; da2 += -a2; da3 += -a3;
	db1 += -b1; db2 += -b2; db3 += -b3;
	da4 += -a4; db4 += -b4; dc4 += -c4;

	// finally, do the parameter update
	var step_size = 0.01;
	a1 += step_size * da1; 
	b1 += step_size * db1; 
	c1 += step_size * dc1;
	a2 += step_size * da2; 
	b2 += step_size * db2;
	c2 += step_size * dc2;
	a3 += step_size * da3; 
	b3 += step_size * db3; 
	c3 += step_size * dc3;
	a4 += step_size * da4; 
	b4 += step_size * db4; 
	c4 += step_size * dc4; 
	d4 += step_size * dd4;

}

