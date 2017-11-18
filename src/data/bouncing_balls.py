"""
ADAPTED FROM @ZHEGAN27:
https://github.com/zhegan27/TSBN_code_NIPS2015/blob/master/bouncing_balls/data/data_handler_bouncing_balls.py

This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""
from math import (sqrt)

from numpy import (shape, array, stack, zeros, dot, arange, meshgrid, exp, save)
from numpy.random import (randn, rand)

from PIL import Image

SIZE = 10  # size of bounding box: SIZE X SIZE.


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x):
    return sqrt((x ** 2).sum())


def sigmoid(x):
    return 1. / (1. + exp(-x))


def bounce_n(steps=128, n=2, r=None, m=None):
    if r is None:
        r = array([1.2] * n).astype('float32')
    if m is None:
        m = array([1] * n).astype('float32')
    # r is to be rather small.
    x = zeros((steps, n, 2), dtype='float32')
    v = randn(n, 2).astype('float32')
    v = v / norm(v) * .5
    good_config = False
    while not good_config:
        _x = 2 + rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if _x[i][z] - r[i] < 0:
                    good_config = False
                if _x[i][z] + r[i] > SIZE:
                    good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(_x[i] - _x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(steps):
        # for how long do we show small simulation
        for i in range(n):
            x[t, i] = _x[i]

        for mu in range(int(1 / eps)):
            for i in range(n):
                _x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if _x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if _x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(_x[i] - _x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = _x[i] - _x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)
    return x.astype('float32')


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float32')


def matricize(x, res, r=None):
    steps, n = shape(x)[0:2]
    if r is None:
        r = array([1.2] * n).astype('float32')

    a = zeros((steps, res, res), dtype='float32')

    [i, j] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(steps):
        for ball in range(n):
            ball_x, ball_y = x[t, ball]
            _delta = exp(-(((i - ball_x) ** 2 + (j - ball_y) ** 2) / (r[ball] ** 2)) ** 4)
            a[t] += _delta

        a[t][a[t] > 1] = 1
    return a


def bounce_mat(res, n=2, steps=128, r=None):
    if r is None:
        r = array([1.2] * n).astype('float32')
    x = bounce_n(steps, n, r)
    a = matricize(x, res, r)
    return a


def bounce_vec(res, n=2, steps=128, r=None, m=None):
    if r is None:
        r = array([1.2] * n).astype('float32')
    x = bounce_n(steps, n, r, m)
    v = matricize(x, res, r)
    v = v.reshape(steps, res ** 2)
    return v


if __name__ == "__main__":
    size = 15  # height and width of frame
    timesteps = 128  # number of timesteps to simulate
    n_balls = 3  # number of balls bouncing around
    # n_train = 4000  # number of train examples
    # n_test = 200  # number of test examples
    n_train = 1
    n_test = 1

    images = [Image.fromarray(step*255) for step in bounce_mat(res=size, n=n_balls, steps=timesteps)]
    im = images[0]
    rest = images[1:]
    with open('../../datasets/test_bounce.gif', 'wb') as f:
        im.save(f, save_all=True, append_images=rest)

    # boulanger-lewandowski's data params 128x15x15
    train_data = stack([bounce_mat(res=size, n=n_balls, steps=timesteps) for _ in range(n_train)])
    test_data = stack([bounce_mat(res=size, n=n_balls, steps=timesteps) for _ in range(n_test)])

    # Sutskever's data params 100x30x30
    size = 30
    timesteps = 100
    train_data = stack([bounce_mat(res=size, n=n_balls, steps=timesteps) for _ in range(n_train)])
    test_data = stack([bounce_mat(res=size, n=n_balls, steps=timesteps) for _ in range(n_test)])
