import numpy
from scipy.integrate import RK45

from python.RK3 import print_rk3


def f(t, x):
    return numpy.array([
        -4 * x[0] + 23 * x[1] + numpy.exp(-t),
        4 * x[0] - 48 * x[1] + numpy.sin(t)]
    )


initial = numpy.array([
    1,
    0.]
)

lower_bound = 0.0
upper_bound = 2
step = 0.1
current = lower_bound
res = initial.copy()
print("------------RK45-------------")
while current < upper_bound:
    integrator = RK45(
        fun=f,
        t0=current,
        y0=res.copy(),
        t_bound=current + step,
        rtol=0.00001
    )
    while integrator.status == "running":
        integrator.step()
    res = integrator.y
    current += step
    print("{0:1.2f} : {1}".format(current, res))

print_rk3(
    fun=f,
    lower_bound=lower_bound,
    y0=initial.copy(),
    upper_bound=upper_bound,
    integration_step=0.1,
    print_step=0.1)
print_rk3(
    fun=f,
    lower_bound=lower_bound,
    y0=initial.copy(),
    upper_bound=upper_bound,
    integration_step=0.01,
    print_step=0.1)
