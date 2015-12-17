function forwardMultiplyGate (a, b) {
	return a * b
}

function forwardAddGate(a, b) {
	return a + b
}

function forwardCircuit (x, y, z) {
	var q = forwardAddGate(x, y)
	var f = forwardMultiplyGate(q, z)
	return f
}

var x = -2, y = 5, z = -4;

var f = forwardCircuit(x, y, z)

console.log(f)

var x = -2, y = 5, z = -4;
var q = forwardAddGate(x, y)
var f = forwardMultiplyGate(q, z)

// gradient of multiply gate wrt its inputs
var derivative_f_wrt_z = q // 3
var derivative_f_wrt_q = z // -4

// derivate of the ADD gate wrt its inputs
var derivative_q_wrt_x = 1.0
var derivative_q_wrt_y = 1.0

//chain rule
var derivative_f_wrt_x = derivative_f_wrt_q * derivative_q_wrt_x
var derivative_f_wrt_y = derivative_f_wrt_q * derivative_q_wrt_y

// [-4, -4 3]
var gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]
console.log(gradient_f_wrt_xyz)

// let the inputs respond to the force/tug:
var step_size = 0.01;
x += step_size * derivative_f_wrt_x; // -2.04
y += step_size * derivative_f_wrt_y; // 4.96
z += step_size * derivative_f_wrt_z; // -3.97

console.log(x, y, z)

// Our circuit now better give higher output:
var q = forwardAddGate(x, y); // q becomes 2.92
var f = forwardMultiplyGate(q, z); // output is -11.59, up from -12! Nice!

console.log(f)

// Numerical gradient check
var x = -2, y = 5, z = -4;

var h = 0.0001;

var x_derivative = (forwardCircuit(x + h, y, z) - forwardCircuit(x, y, z)) / h; // -4
var y_derivative = (forwardCircuit(x, y + h, z) - forwardCircuit(x, y, z)) / h; // -4
var z_derivative = (forwardCircuit(x, y, z + h) - forwardCircuit(x, y, z)) / h; //3

console.log(x_derivative, y_derivative, z_derivative)
