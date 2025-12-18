from numpy import *
import matplotlib.pyplot as plt

def flux_surface():
    """
    Compute a Miller-like flux surface.

    Returns
    -------
    R_s : ndarray
        Major radius coordinates
    Z_s : ndarray
        Vertical coordinates
    """
    A=2.2
    kappa=1.5
    delta=0.3
    R0=2.5
    theta=linspace(0,2*pi)
    r=R0/A
    R_s=R0+r*cos(theta+(arcsin(delta)*sin(theta)))
    Z_s=kappa*r*sin(theta)
    return R_s, Z_s


def plot_surface(R_s, Z_s, savefig = True):
    """
    plots and saves the surface for me woop woop

    Args:
        R_s,Z_s
    """
    plt.plot(R_s, Z_s)
    plt.axis("equal")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")

    if savefig:
        plt.savefig("./miller.png")
    
def main()-> None:
    R_s, Z_s = flux_surface()
    plot_surface(R_s, Z_s)
    print("All done :) ")