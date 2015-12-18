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

// random init weights
var a = 1, b = -2, c = -1
for (var iter = 0; iter < 400; iter++) {
	// sample
	var i = Math.floor(Math.random() * data.length)
	var x = data[i][0]
	var y = data[i][1]
	var label = labels[i]

	// compute top level pull based on wrong prediction
	var score = a*x + b*y +c
	var pull = 0.0
	if (label === 1 && score < 1) pull = 1
	if (label === -1 && score > -1) pull = -1

	// compute gradient and update parameters
	var step_size = 0.01
	a += step_size * (x * pull - a) // df/da * df/df - |a|
	b += step_size * (y * pull - b) // df/db * df/df - |b|
	c += step_size * (1 * pull) // df/dc * df/df but no regularization
}
