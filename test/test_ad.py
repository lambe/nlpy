# Tests relative to algorithmic differentiation.
from nlpy.model import AdolcModel
import numpy as np

# Define a few problems.

class AdolcRosenbrock(AdolcModel):
    "The standard Rosenbrock function."

    def obj(self, x, **kwargs):
        return np.sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )


class AdolcHS7(AdolcModel):
    "Problem #7 in the Hock and Schittkowski collection."

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return (1 + x[0]**2)**2 + x[1]**2 - 4


def get_derivatives(nlp):
    g = nlp.grad(nlp.x0)
    H = nlp.dense_hess(nlp.x0, nlp.x0)
    if nlp.m > 0:
        J = nlp.dense_jac(nlp.x0)
        v = -np.ones(nlp.n)
        w = 2*np.ones(nlp.m)
        Jv = nlp.jac_vec(nlp.x0,v)
        JTw = nlp.vec_jac(nlp.x0,w)
        return (g,H,J,Jv,JTw)
    else:
        return (g,H)


def test_rosenbrock():
    rosenbrock = AdolcRosenbrock(n=5, name='Rosenbrock', x0=-np.ones(5))
    (g,H) = get_derivatives(rosenbrock)
    expected_g = np.array([-804., -1204., -1204., -1204., -400.])
    expected_H = np.array([[1602.,  400.,    0.,    0.,   0.],
                           [ 400., 1802.,  400.,    0.,   0.],
                           [   0.,  400., 1802.,  400.,   0.],
                           [   0.,    0.,  400., 1802., 400.],
                           [   0.,    0.,    0.,  400., 200.]])
    assert(np.allclose(g,expected_g))
    assert(np.allclose(H,expected_H))


def test_hs7():
    hs7 = AdolcHS7(n=2, m=1, name='HS7', x0=2*np.ones(2))
    (g,H,J,Jv,JTw) = get_derivatives(hs7)
    expected_g = np.array([0.8, -1.])
    expected_H = np.array([[-0.24, 0.],[0., 0.]])
    expected_J = np.array([[40., 4.]])
    expected_Jv = np.array([-44.])
    expected_JTw = np.array([80., 8.])
    assert(np.allclose(g,expected_g))
    assert(np.allclose(H,expected_H))
    assert(np.allclose(J,expected_J))
    print Jv, expected_Jv
    assert(np.allclose(Jv,expected_Jv))
    assert(np.allclose(JTw,expected_JTw))
