import numpy
from scipy.linalg import lu_factor, lu_solve

BASE_A = numpy.array([
    [-29., 6., -6., -4., -3., -8., -5., 5.],
    [6., -13., -3., 5., 4., 3., 1., 7.],
    [5., -5., -1., 7., 2., 0., 7., 1.],
    [5., -5., 5., 6., 4., -7., 4., 0.],
    [4., 4., 7., -4., 9., -8., -8., -4.],
    [-4., 5., -4., 1., 0., 12., 0., 6.],
    [-3., -2., -4., 2., -8., -3., 16., 4.],
    [7., 5., 0., 2., 0., -6., 8., -12.]
])

BASE_B = numpy.array([-175, 133, 110, 112, 17, 32, 13, -18])

P = numpy.array([10, 1, 0.1, 0.01, 0.0001, 0.000001, 0.00000001, 0.000000000000001, 0])


def generate_system(param):
    a_new = numpy.copy(BASE_A)
    a_new.itemset((0, 0), param + BASE_A[0, 0])
    b_new = numpy.copy(BASE_B)
    b_new.itemset(0, 4 * param + BASE_B[0])
    return a_new, b_new


numpy.set_printoptions(suppress=True)
for p in P:
    print("------------------------------------------------------------------------------------------------")
    (a, b) = generate_system(p)
    lu = lu_factor(a)
    x = lu_solve(lu, b)
    print("p = {0:e}:"
          .format(p))
    print("Simple:")
    print("Det: {0}"
          .format(numpy.linalg.det(a)))
    print("Cond = {0:e}"
          .format(numpy.linalg.cond(a)))
    print("x = {0}"
          .format(x))

    a_transposed = a.transpose()
    a_modified = a_transposed.dot(a)
    b_modified = a_transposed.dot(b)
    lu_modified = lu_factor(a_modified)
    x_modified = lu_solve(lu_modified, b_modified)
    print("Modified:")
    print("Det: {0}"
          .format(numpy.linalg.det(a_modified)))
    print("Cond = {0:e}"
          .format(numpy.linalg.cond(a_modified)))
    print("x = {0}"
          .format(x_modified))
    norm1 = numpy.linalg.norm(x - x_modified)
    norm2 = numpy.linalg.norm(x)
    print("d=||x1 - x2|| / ||x1|| = {0}".format(norm1 / norm2))
    print("------------------------------------------------------------------------------------------------")
