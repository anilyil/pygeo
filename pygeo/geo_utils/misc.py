# External modules
import numpy as np


class KSfunction(object):
    """
    Helper class for KSComp.

    Helper class that can be used to aggregate constraint vectors with a
    Kreisselmeier-Steinhauser Function.

    This is taken from OpenMDAO here: https://github.com/OpenMDAO/OpenMDAO/blob/master/openmdao/components/ks_comp.py
    """

    @staticmethod
    def _compute_values(g, rho):
        """
        Compute values needed by the KS function for the given array of constraints.

        Parameters
        ----------
        g : ndarray
            Array of constraint values, where negative means satisfied and positive means violated.
        rho : float
            Constraint Aggregation Factor.

        Returns
        -------
        tuple
            g_max, g_diff, exponents and summation as needed by compute and derivates functions.
        """
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        return g_max, g_diff, exponents, summation

    @staticmethod
    def compute(g, rho=50.0):
        """
        Compute the value of the KS function for the given array of constraints.

        Parameters
        ----------
        g : ndarray
            Array of constraint values, where negative means satisfied and positive means violated.
        rho : float
            Constraint Aggregation Factor.

        Returns
        -------
        float
            Value of KS function.
        """
        g_max, g_diff, exponents, summation = KSfunction._compute_values(g, rho)

        KS = g_max + 1.0 / rho * np.log(summation)

        return KS

    @staticmethod
    def derivatives(g, rho=50.0):
        """
        Compute elements of [dKS_gd, dKS_drho] for the given array of constraints.

        Parameters
        ----------
        g : ndarray
            Array of constraint values, where negative means satisfied and positive means violated.
        rho : float
            Constraint Aggregation Factor.

        Returns
        -------
        ndarray
            Derivative of KS function with respect to parameter values.
        """
        g_max, g_diff, exponents, summation = KSfunction._compute_values(g, rho)

        dsum_dg = rho * exponents
        dKS_dsum = 1.0 / (rho * summation)
        dKS_dg = dKS_dsum * dsum_dg

        dsum_drho = np.sum(g_diff * exponents, axis=-1)[:, np.newaxis]
        dKS_drho = dKS_dsum * dsum_drho

        return dKS_dg, dKS_drho


def area2(hedge, point):
    """Determines the area of the triangle formed by a hedge and
    an external point"""

    pa = hedge.twin.origin
    pb = hedge.origin
    pc = point
    return (pb.x - pa.x) * (pc[1] - pa.y) - (pc[0] - pa.x) * (pb.y - pa.y)


def isLeft(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])


def lefton(hedge, point):
    """Determines if a point is to the left of a hedge"""

    return area2(hedge, point) >= 0


def hangle(dx, dy):
    """Determines the angle with respect to the x axis of a segment
    of coordinates dx and dy
    """
    length = np.sqrt(dx * dx + dy * dy)

    if dy > 0:
        return np.arccos(dx / length)
    else:
        return 2 * np.pi - np.arccos(dx / length)


def fillKnots(t, k, level):
    t = t[k - 1 : -k + 1]  # Strip out the np.zeros
    newT = np.zeros(len(t) + (len(t) - 1) * level)
    start = 0
    for i in range(len(t) - 1):
        tmp = np.linspace(t[i], t[i + 1], level + 2)
        for j in range(level + 2):
            newT[start + j] = tmp[j]

        start += level + 1

    return newT


def convertTo1D(value, dim1):
    """
    Generic function to process 'value'. In the end, it must be
    array of size dim1. value is already that shape, excellent,
    otherwise, a scalar will be 'upcast' to that size
    """

    if np.isscalar(value):
        return value * np.ones(dim1)
    else:
        temp = np.atleast_1d(value)
        if temp.shape[0] == dim1:
            return value
        else:
            raise ValueError(
                "The size of the 1D array was the incorrect shape! " + f"Expected {dim1} but got {temp.size}"
            )


def convertTo2D(value, dim1, dim2):
    """
    Generic function to process 'value'. In the end, it must be dim1
    by dim2. value is already that shape, excellent, otherwise, a
    scalar will be 'upcast' to that size
    """

    if np.isscalar(value):
        return value * np.ones((dim1, dim2))
    else:
        temp = np.atleast_2d(value)
        if temp.shape[0] == dim1 and temp.shape[1] == dim2:
            return value
        else:
            raise ValueError("The size of the 2D array was the incorrect shape")
