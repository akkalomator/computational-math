import numpy

from scipy.integrate import quad
from scipy.interpolate import splrep, splev, lagrange
from numpy import sin, abs

# For comparing floats
DELTA = 0.000001


def f(x):
    return quad(
        lambda t: sin(t*t),
        0, x)


start = 1.5
end = 2.7
h = 0.2

# From 1.5 to 2.7 step 0.2
x = numpy.arange(start, end + DELTA, h)
# Map x to f(x)
y_with_eps = numpy.array([f(xi) for xi in x])

print("             Function             ")
print("+-----+------------+-------- -----+")
print("| xi  |     fi     |      eps     |")
for i in range(x.size):
    (fi, eps_i) = y_with_eps[i]
    print("| {0:2.1f} | {1:2.8f} | {2:e} |".format(x[i], fi, eps_i))
print("+-----+------------+--------------+")
print()
print()

new_start = 1.6
new_end = 2.6

# From 1.6 to 2.6 step 0.2
x_new = numpy.arange(new_start, new_end + DELTA, h)
# Map x_new to f(x_new)
y_quanc = numpy.array([f(xi) for xi in x_new])

# Map x_new to spline(x_new)
y = numpy.array([yi for (yi, _) in y_with_eps])
spline_coefficients = splrep(x, y, s=0)  # spline
y_spline = splev(x_new, spline_coefficients, der=0)  # seval

y_lagrange = lagrange(x, y)

print("                             Comparison                                 ")
print("+-----+------------+------------+------------+------------+------------+")
print("| xi  |   quanc    |   spline   | abs(q - s) |  lagrange  | abs(q - l) |")
for i in range(x_new.size):
    (quanc, _) = y_quanc[i]
    spline = y_spline[i]
    lagrange = y_lagrange(x_new[i])
    dqs = abs(quanc - spline)
    dql = abs(quanc - lagrange)
    print("| {0:2.1f} | {1:2.8f} | {2:2.8f} | {3:2.8f} | {4:2.8f} | {5:2.8f} |"
          .format(x_new[i], quanc, spline, dqs, lagrange, dql))
print("+-----+------------+------------+------------+------------+------------+")
