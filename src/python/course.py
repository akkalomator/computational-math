import numpy
from matplotlib import pyplot
from scipy.integrate import quad, RK45
from scipy.optimize import toms748

TOLERANCE = 1e-6

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

_, space_plot = pyplot.subplots()
_, velocities_plot = pyplot.subplots()


def simulate(initial_state, name=""):
    current = lower_bound
    res = initial_state.copy()
    total_tol = [0, 0, 0, 0]
    t = numpy.arange(lower_bound, upper_bound + TOLERANCE, step)
    x, dx, y, dy = [initial_state[0]], [initial_state[1]], [initial_state[2]], [initial_state[3]]
    while numpy.abs(current - upper_bound) > TOLERANCE:
        integrator = RK45(
            fun=motion_equation,
            t0=current,
            y0=res,
            t_bound=current + step,
            rtol=TOLERANCE
        )
        while integrator.status == "running":
            integrator.step()
        previous_res = res
        res = integrator.y
        x_, dx_, y_, dy_ = res
        x.append(x_)
        dx.append(dx_)
        y.append(y_)
        dy.append(dy_)
        current += step
        tolerance = abs(res - previous_res) * TOLERANCE
        total_tol += tolerance
        print("{0:1.2f} : {1}\n       ±{2}".format(current, res, total_tol))

    space_plot.plot(x, y, label="{0} : (x, y)".format(name))
    velocities_plot.plot(t, dx, label="{0}: dx/dt(t)".format(name))
    velocities_plot.plot(t, dy, label="{0}: dy/dt(t)".format(name))


print("Relative tolerance = {0}".format(TOLERANCE))
simulate(initial_state=initial, name="1")
simulate(initial_state=initial * 1.05, name="2")
simulate(initial_state=initial * 1.3, name="3")
space_plot.set_title("Space plot")
space_plot.plot(0, 0, 'o', label="Earth", markersize=12)
space_plot.plot(1, 0, 'o', label="Moon", markersize=6)
space_plot.legend()
velocities_plot.set_title("Velocities")
velocities_plot.legend()
pyplot.show()
