// product rule
var x = a * b
var dx = 1 // dx/dx is always 1 (derivative wrt itself)
var da = b * dx
var db = a * dx

// sum rule
var x = a + b
var dx = 1
var da = 1.0 * dx
var db = 1.0 * dx

// 3 nums sum
var q = a + b //gate 1
var x = q + c //gate 2

//backward pass:
//backprop gate2
var dc = 1.0 * dx 
var dq = 1.0 * dx
//backprop gate1
var da = 1.0 * dq
var db = 1.0 * dq

// Faster sum
var x = a + b + c;
var da = 1.0 * dx; var db = 1.0 * dx; var dc = 1.0 * dx;

// sum + product
var x = a * b + c
var da = b * dx
var db = a * dx
var dc = 1.0 * dx

var sig = function(x) { return 1 / (1 + Math.exp(-x)); };

//neuron in 2 steps:
var q = a*x + b*y + c
var f = siq(q)

// backprop
var df = 1
var dq = (f * (1 - f)) * df

//to inputs
var da = x * dq
var dx = a * dq
var dy = b * dq
var db = y * dq
var dc = 1.0 * dq

var x = a * a
var dx = 1
var da = a * dx
var da += a * dx
// 2 * da

// short form for a^2
var da = 2 * a * dx

var x = a*a + b*b + c*c
var da = 2 * a * dx
var db = 2 * b * dx
var dc = 2 * c * dx

var x =  Math.pow(((a * b + c) * d), 2) // (d(ab + c))^2
var x1 = a * b + c
var x2 = x1 * d
var x = x2 * x2

//backprop
var dx = 1
var dx2 = 2 * x2 * dx //backprop into x2
var dd  = x1 * dx2 // backprop into d
var dx1 = d * dx2 //backprop into x1
var da = b * dx1
var db = a * dx1
var dc = 1.0 * dx1

// division
var x = 1.0/a
var da = -1.0/(a*a)

// division decomposed
var x = (a + b)/(c + d)

// decompose
var x1 = (a + b)
var x2 = (c + d)
var x3 = 1.0/x2
var x = x1 * x3 // flip & multiply

//backprop
var dx = 1
var dx1 = x3 * dx
var dx1 = x1 * dx
var dx2 = (-1.0/(x2*x2)) * dx3
var da = 1.0 * dx1
var db = 1.0 * dx1
var dc = 1.0 * dx2
var dd = 1.0 * dx2

// max
var x = Math.max(a, b)
var da = a === x ? 1.0 * dx : 0.0
var db = b === x ? 1.0 * dx : 0.0

// ReLU
var x = Math.max(a, 0)
var da = a > 0 ? 1.0 * dx : 0.0
