var x = -2, y = 3;
var forwardMultiplyGate = function(x, y) {
	return x * y
}

var out = forwardMultiplyGate(x, y);

for (i = 0; i < 100; i++) {
	var x_gradient = y;
	var y_gradient = x;
	var step_size = 0.01;

	x += step_size * x_gradient;
	y += step_size * y_gradient;

	out = forwardMultiplyGate(x, y)
	console.log(x, y, out)
}
