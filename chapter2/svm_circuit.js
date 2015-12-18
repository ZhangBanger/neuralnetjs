// A wire in a circuit
function Unit (value, grad) {
	// value computed in forward
	this.value = value
	// derivative of circuit output wrt this unit, computed in backward
	this.grad = grad
}

// Computation gates that combine wires, feed forward values, pull back gradients
var multiplyGate = function() {}
multiplyGate.prototype = {
	forward: function(u0, u1) {
		// store pointers to input Units u0 and u1 and output unit utop
		this.u0 = u0
		this.u1 = u1
		this.utop = new Unit(u0.value * u1.value, 0.0)
		return this.utop
	},
	backward: function() {
		// take the gradient in output unit and chain it with the
    	// local gradients, which we derived for multiply gate before
    	// then write those gradients to those Units.
    	this.u0.grad += this.u1.value * this.utop.grad
    	this.u1.grad += this.u0.value * this.utop.grad
	}
}

var addGate = function() {}
addGate.prototype = {
	forward: function(u0, u1) {
		this.u0 = u0
		this.u1 = u1
		this.utop = new Unit(u0.value + u1.value, 0.0)
		return this.utop
	},
	backward: function() {
		// add gate derivateive wrt both inputs is 1
		this.u0.grad += 1 * this.utop.grad
		this.u1.grad += 1 * this.utop.grad
	}
}

var sigmoidGate = function() {
	this.sig = function(x) { return 1 / (1 + Math.exp(-x))}
}
sigmoidGate.prototype = {
	forward: function(u0) {
		this.u0 = u0
		this.utop = new Unit(this.sig(this.u0.value), 0.0)
		return this.utop
	},
	backward: function() {
		var s = this.sig(this.u0.value)
		this.u0.grad += (s * (1 - s)) * this.utop.grad
	}
}

// Circuit: takes 5 units (a,b,c params and x,y data) and outputs a single unit (-1 - 1)
// Also computes gradients wrt input
function Circuit () {
	//gates
	this.mulg0 = new multiplyGate()
	this.mulg1 = new multiplyGate()
	this.addg0 = new addGate()
	this.addg1 = new addGate()
}

Circuit.prototype = {
	forward: function(x, y, a, b, c) {
		this.ax = this.mulg0.forward(a, x) // a * x
		this.by = this.mulg1.forward(b, y) // b * y
		this.axpby = this.addg0.forward(this.ax, this.by) // a*x + b*y
		this.axpbypc = this.addg1.forward(this.axpby, c)
		return this.axpbypc
	},
	backward: function(gradient_top) { // pull from above - + or - 1
		this.axpbypc.grad = gradient_top
		this.addg1.backward()
		this.addg0.backward()
		this.mulg1.backward()
		this.mulg0.backward()
	}
}

// SVM class
function SVM() {
	//random initial weights
	this.a = new Unit(1.0, 0.0)
	this.b = new Unit(-2.0, 0.0)
	this.c = new Unit(-1.0, 0.0)

	this.circuit = new Circuit()
}
SVM.prototype = {
	forward: function(x, y) { // x and y are units (wires that carry value/grad)
		this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c)
		return this.unit_out
	},
	backward: function(label) { // +1 or -1
		// re initialize gradients to 0 since we're accumulating them rather than setting
		this.a.grad = 0.0
		this.b.grad = 0.0
		this.c.grad = 0.0

		// compute top level pull based on circuit output
		var pull = 0.0
		if (label === 1 && this.unit_out.value < 1) {
			pull = 1 // false negative - pull up
		}
		if (label === -1 && this.unit_out.value > -1) {
			pull = -1 // false positive, pull down
		}
		this.circuit.backward(pull) // write gradient into x, y, a, b, c

		// Regularization - gradient also accumulates pull proportional to weights
		// but in opposite direction - to push to 0
		this.a.grad += -this.a.value
		this.b.grad += -this.b.value
	},
	// full forward + backward pass of online SGD
	learnFrom: function(x, y, label) {
		this.forward(x, y) // update .value in all Units
		this.backward(label) // unit .grad in all Units
		this.parameterUpdate() // take a step in params based on tug of grads
	},
	parameterUpdate: function() {
		var step_size = 0.01
		this.a.value += step_size * this.a.grad
		this.b.value += step_size * this.b.grad
		this.c.value += step_size * this.c.grad
		// x and y are not part of SVM state, but are part of Circuit in circuit.axpbypc
	}
}

// Time to train!
var data = []
var labels = []
// hardcoded from dataset file
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);

var svm = new SVM()

// pure recall accuracy
var evalTrainingAccuracy = function() {
	var num_correct = 0
	for (var i = 0; i < data.length; i++) {
		var x = new Unit(data[i][0], 0.0) // remember that x and y are transient wires (Units)
		var y = new Unit(data[i][1], 0.0) // so their gradients don't matter
		var true_label = labels[i]

		var predicted_label = svm.forward(x, y).value > 0 ? 1 : -1
		if (predicted_label === true_label) {
			num_correct++
		}
	}

	return num_correct / data.length
}

// training loop
for (var iter = 0; iter < 400; iter++) {
	//sample
	var i = Math.floor(Math.random() * data.length)
	var x = new Unit(data[i][0], 0.0)
	var y = new Unit(data[i][1], 0.0)
	var label = labels[i]
	svm.learnFrom(x, y, label)

	if (iter % 25 == 0) {
		console.log('training accuracy at iter ' + iter + ': ' + evalTrainingAccuracy())
	}
}
