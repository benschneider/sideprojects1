from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import logging

def plot3dHist(data):
    # plt.ion()
    # fig = plt.figure()
    fig = Figure()
    ax = Axes3D(fig)
    ax.clear()
    column_names = ['I1a','Q1a','I2a','Q2a']
    row_names = ['I1b','Q1b','I2b','Q2b']
    xpos = np.arange(0, 4, 1)    # Set up a mesh of positions
    ypos = np.arange(0, 4, 1)
    xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(16)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    # dz = data.flatten()
    dz = data
    colors = []
    for item in dz:
        if item < 0:
            colors.append('b')
        else:
            colors.append('r')
    ticksx = np.arange(0.5, 4.5, 1)
    ticksy = np.arange(0.5, 4.5, 1)
    # ax.grid(True)
    ax.set_xticks(ticksx)
    ax.set_yticks(ticksy)
    ax.w_xaxis.set_ticklabels(column_names)
    ax.w_yaxis.set_ticklabels(row_names)
    ax.mouse_init()
    # Plot a horizontal plane for orientation:
    x = np.arange(0, 4.5, 0.5)
    y = np.arange(0, 4.5, 0.5)
    X, Y = np.meshgrid(x, y)
    # ax.zaxis.set_major_locator(LinearLocator(6))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.autoscale_view(tight=None, scalex=True, scaley=True, scalez=True)
    # ax.plot_wireframe(X, Y, 0, rstride=1, cstride=1)
    ax.plot_surface(X, Y, 0, alpha=0.4, color='g')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', alpha=0.9)
    ax.set_zlim3d(bottom=-max(abs(dz)), top=max(abs(dz)))
    return fig, ax


def plot3dHist2(data):
    # plt.ion()
    # fig = plt.figure()
    fig = Figure()
    ax = Axes3D(fig)
    ax.clear()
    column_names= ['I1a','Q1a','I2a','Q2a','I1b','Q1b','I2b','Q2b']
    row_names = ['I1a','Q1a','I2a','Q2a','I1b','Q1b','I2b','Q2b']
    lx = 8            # Matrix dimensions
    ly = 8
    xpos = np.arange(0,lx,1)    # Set up a mesh of positions
    ypos = np.arange(0,ly,1)
    xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    logging.debug(str(data))
    dz = data.flatten()
    # dz = data.flatten()
    colors = []
    for item in dz:
        if item < 0:
            colors.append('b')
        else:
            colors.append('r')
    ticksx = np.arange(0.5, 8.5, 1)
    ticksy = np.arange(0.5, 8.5, 1)
    # ax.grid(True)
    ax.set_xticks(ticksx)
    ax.set_yticks(ticksy)
    ax.w_xaxis.set_ticklabels(column_names)
    ax.w_yaxis.set_ticklabels(row_names)
    ax.mouse_init()
    # Plot a horizontal plane for orientation:
    x = np.arange(0, 8.5, 0.5)
    y = np.arange(0, 8.5, 0.5)
    X, Y = np.meshgrid(x, y)
    # ax.zaxis.set_major_locator(LinearLocator(6))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.autoscale_view(tight=None, scalex=True, scaley=True, scalez=True)
    # ax.plot_wireframe(X, Y, 0, rstride=1, cstride=1)
    ax.plot_surface(X, Y, 0, alpha=0.4, color='g', cstride=4, rstride=4)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', alpha=0.9)
    ax.set_zlim3d(bottom=-max(abs(dz)), top=max(abs(dz)))
    return fig, ax

if __name__ == '__main__':
    data = np.random.random([4,4])
    plot3dHist(data)
