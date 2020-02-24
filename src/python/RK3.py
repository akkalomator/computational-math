import numpy


class RK3:

    def __init__(self, f, t0, y0, t):
        self.f = f
        self.lower_bound = t0
        self.t = t0
        self.y = y0
        self.upper_bound = t
        self.status = "running"

    def step(self, h):
        if self.status == "completed":
            return
        if self.t + h > self.upper_bound:
            h = self.upper_bound - self.t
        k1 = self.k1(h)
        k2 = self.k2(h, k1)
        k3 = self.k3(h, k1, k2)
        self.y = self.z(k1, k2, k3)
        self.t += h
        if self.t >= self.upper_bound:
            self.status = "completed"

    def k1(self, h):
        return h * self.f(self.t, self.y)

    def k2(self, h, k1):
        return h * self.f(self.t + h / 2, self.y + k1 / 2)

    def k3(self, h, k1, k2):
        return h * (self.f(self.t + h, self.y - k1 + 2 * k2))

    def z(self, k1, k2, k3):
        return self.y + (k1 + 4 * k2 + k3) / 6


def print_rk3(fun, lower_bound, y0, upper_bound, integration_step, print_step):
    print("------------RK3--------------")
    print("Integration step: {0}".format(integration_step))
    rk3_integrator = RK3(fun, lower_bound, y0, upper_bound)
    cur_step = 0
    while rk3_integrator.status == "running":
        rk3_integrator.step(integration_step)
        cur_step += integration_step
        if numpy.abs(cur_step - print_step) < 0.000001:
            cur_step = 0
            print("{0:1.2f} : {1}".format(rk3_integrator.t, rk3_integrator.y))
