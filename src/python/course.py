import numpy
from matplotlib import pyplot
from scipy.integrate import quad, RK45
from scipy.optimize import toms748

TOLERANCE = 0.00001

MU = 1 / 82.3
T = 12


def r3(x, y):
    return numpy.sqrt(x ** 2 + y ** 2)


def rl(x, y):
    return numpy.sqrt((x - 1) ** 2 + y ** 2)


def motion_equation(t, x):
    r3_ = r3(x[0], x[2]) ** 3
    rl_ = rl(x[0], x[2]) ** 3
    return numpy.array([
        x[1],
        -(1 - MU) * x[0] / r3_ - MU * (x[0] - 1) / rl_,
        x[3],
        -(1 - MU) * x[2] / r3_ - MU * x[2] / rl_
    ])


def x_star_equation(x):
    def x_star_equation_integral(z):
        return 1 / (numpy.e ** (1.7 * (z ** 2)) + x)

    (integral, _) = quad(x_star_equation_integral, 0, 1)
    return integral - 0.3707355 * x


x_star = toms748(x_star_equation, 0, 3)
a = 1.2 * x_star
b = 0
c = 0
d = 1

initial = numpy.array([
    a, b, c, d
])

lower_bound = 0
upper_bound = T
step = 0.1
current = lower_bound
res = initial

t = numpy.arange(lower_bound, upper_bound + TOLERANCE, step)
x, dx, y, dy = [a], [b], [c], [d]

while numpy.abs(current - upper_bound) > TOLERANCE:
    integrator = RK45(
        fun=motion_equation,
        t0=current,
        y0=res,
        t_bound=current + step,
        rtol=0.00001
    )
    while integrator.status == "running":
        integrator.step()
    res = integrator.y
    x_, dx_, y_, dy_ = res
    x.append(x_)
    dx.append(dx_)
    y.append(y_)
    dy.append(dy_)
    current += step
    print("{0:1.2f} : {1}".format(current, res))

_, ax1 = pyplot.subplots()
ax1.plot(0, 0, 'o', label="Earth", markersize=12)
ax1.plot(1, 0, 'o', label="Moon", markersize=6)
ax1.plot(x, y, label="(x, y)")
ax1.set_title("Space plot")
ax1.legend()

_, ax2 = pyplot.subplots()
ax2.plot(t, dx, label="dx/dt(t)")
ax2.plot(t, dy, label="dy/dt(t)")
ax2.set_title("Velocities")
ax2.legend()

pyplot.show()
