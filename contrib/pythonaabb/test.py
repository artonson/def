import numpy as np
import plotly.graph_objs as go
import pyaabb
import random


def plot(lines, ps, c):
    plts = []
    for i, l in enumerate(lines):
        plts.append(
            go.Scatter(
                x=[l[0][0], l[1][0]],
                y=[l[0][1], l[1][1]],
                mode='lines+markers',
                line=dict(color =  'red' if i == c else 'blue')
            )
        )

    for p in ps:
        plts.append(
            go.Scatter(
                x=[p[0]],
                y=[p[1]],
                mode='markers',
                marker=dict(color='green')
            )
        )

    fig = go.Figure(data=plts)
    fig.show()

def curve(t):
    a = 1
    b = 1

    kx = 3
    ky = 2

    x = a*np.cos(kx*t)
    y = b*np.sin(ky*t)

    res = []

    for i in range(x.shape[0]-1):
        res.append(
            (np.array([x[i], y[i], 0]),
             np.array([x[i+1], y[i+1], 0]))
        )

    return res

def create_boxes(lines):
    corners = []
    eps = 1e-8
    for l in lines:
        minc = np.array([
            min(l[0][0], l[1][0])-eps,
            min(l[0][1], l[1][1])-eps,
            min(l[0][2], l[1][2])-eps
            ])
        maxc = np.array([
            max(l[0][0], l[1][0])+eps,
            max(l[0][1], l[1][1])+eps,
            max(l[0][2], l[1][2])+eps
        ])
        corners.append([minc, maxc])

    return corners

def dist(p, i, lines):
    p_list = []
    p_list.append(lines[i][0])
    p_list.append(lines[i][1])

    # print(lines[i][0])
    # print(lines[i][1])
    # print(p)

    a = lines[i][0]
    n = lines[i][1]-a
    l = np.linalg.norm(n)
    n /= l
    v = (p-a) - np.dot(p-a, n)*n

    d = np.linalg.norm(v)

    pl = p - v
    t = np.dot(pl-a, n)/l

    assert(np.linalg.norm(((1-t)*a + t*lines[i][1])-pl) < 1e-10)

    if t >= 0 and t <= 1:
        return d**2, pl


    d_list = np.array([np.linalg.norm(p-p_list[0]), np.linalg.norm(p-p_list[1])])
    d = np.min(d_list)
    closest = p_list[np.argmin(d_list)]
    return d**2, closest



if __name__ == "__main__":
    t = np.linspace(0, 2*np.pi, 20)

    lines = curve(t[:-1])
    corners = create_boxes(lines)

    aabb = pyaabb.AABB()
    aabb.build(corners)

    # random.seed(1)

    pt = np.array([random.random(), random.random(), 0])



    dist_func = lambda p, i: dist(p, i, lines)
    # print(dist_func(pt, 11))
    # plot(lines, [pt], 11)

    nn = aabb.nearest_point(pt, dist_func)
    print(nn)
    plot(lines, [pt], nn[0])



