import pywt
import numpy as np
from matplotlib import pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def origin_s(t):
    return 2 + 3 * np.cos (2 * np.pi * 50 * t - np.pi * 30/180) + 1.5 * np.cos(2 * np.pi * 75* t + np.pi * 90/180)
origin_s = np.vectorize(origin_s)

def main():

    # Number of sample points
    N = 256
    # sample spacing
    T = 1.0 / 256

    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = origin_s(x)

    # (cA, cD) : tuple Approximation and detail coefficients.
    (A1,D1) = pywt.dwt(y, 'haar')
    (A2,D2) = pywt.dwt(A1, 'haar')
    (A3,D3) = pywt.dwt(A2, 'haar')

    fig = plt.figure()
    a1 = fig.add_subplot(411)
    a1.plot(x, y)

    a2 = fig.add_subplot(412)
    a2.plot(x, np.concatenate((A1,D1)))

    a3 = fig.add_subplot(413)
    a3.plot(x, np.concatenate((A2,D2,np.zeros(N-2*len(A2)))))

    a4 = fig.add_subplot(414)
    a4.plot(x, np.concatenate((A3,D3,np.zeros(N-2*len(A3)))))


    plt.grid()
    plt.show()

if __name__ == 'main':
    main()