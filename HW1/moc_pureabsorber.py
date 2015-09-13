import numpy as np
from math import *
from scipy.optimize import brenth as rootfind

def main(n_azimuth, polar_ang, tol=1.e-12):

    #loop over particles

    #Start at point of interest and trace upstream

    #Determine where you hit (if it is fuel, etc)

        #If it is fuel need to account for fuel generation

        #Else just need the amount of mfp traveled 

    # Define geometry parameters
    x_left  = -2.
    x_right =  2.
    y_top   =  2.
    y_bot   = -2.
    radius  =  1.

 
    #Pick starting point
    x_prev = 1.5
    y_prev = 1.5
    
    #Pick direction
    theta = polar_ang
    phi   = pi/2.
    xcos =  sin(theta)*cos(phi)
    ycos =  sin(theta)*sin(phi)

    #We are ray tracing upstream, so flip cosines
    xcos *= -1.
    ycos *= -1.

    #s is parametric length of vector
    f_circ = lambda s: (x_prev + xcos*s)**2 + (y_prev + y_cos*s)**2 - radius**2

    #Calculate boundary intersections
    if xcos == 0.:
        s_left = -99
        s_right = -99
    else:
        s_left  = (x_left - x_prev)/xcos
        s_right = (x_right - x_prev)/xcos

    if ycos == 0.:
        s_top = -99
        s_bot = -99
    else:
        s_top   = (y_top  - y_prev)/ycos
        s_bot   = (y_bot - y_prev)/ycos

    #Find the min s that is positive, this is the intersection with boundary
    intersects = [s_left, s_right, s_top, s_bot]
    s_min = min(i for i in intersects if i > 0 )

    #Now check if we hit the circle
    if f_circ(s_min)*f_circ(

    #Eval coordinates
    x_new = x_prev + xcos*s_min
    y_new = y_prev + ycos*s_min

    print "I am now here", x_new, y_new

    #Find intersection
    #rootfind(f_circ, 









if __name__ == "__main__":
    main(10,pi/2)
