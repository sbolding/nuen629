import numpy as np
import re
from math import *
from scipy.optimize import fsolve

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

    #cross sections
    sigma_f = 0.1414
    sigma_m = 0.08
    Q_f_tot = 1.
    q_f     = Q_f_tot/(4.*pi)
    q_mod   = 0.0

    debug_mode = True
    if debug_mode:
        sigma_m = sigma_f
        q_mod = q_f

    #Pick the point of interest and trace upstream from it
    x_prev = 1.5
    y_prev = 1.5
    in_fuel = False
    entered_fuel_last_iter = False
    hit_boundary = False

    #Det if we started in the fuel, this is for roundoff purposes
    if (x_prev**2 + y_prev**2 < radius**2):
        entered_fuel_last_iter = True

    #Num of mfp we've traveled, and angular flux contribution to this point
    psi = 0.0
    n_mfp = 0.0
    max_mfp = -1.*log(tol)

    #Pick direction
    theta = polar_ang
    phi   = pi/4.2
    xcos =  sin(theta)*cos(phi)
    ycos =  sin(theta)*sin(phi)

    #We are ray tracing upstream, so flip cosines
    xcos *= -1.
    ycos *= -1.

    #s is parametric length of vector
    #f_circ can be used to check if circle hit or not
    f_circ = lambda s: (x_prev + xcos*s)**2 + (y_prev + ycos*s)**2 - radius**2

    while (n_mfp < max_mfp):

        print "MFP TOL: ", n_mfp, max_mfp

        #Calculate all boundary intersections
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

        #Roots of parametric equation for circle
        A = xcos**2 + ycos**2
        B = 2.*xcos*x_prev + 2.*ycos*y_prev
        C = x_prev**2 + y_prev**2 - radius**2
        det = B*B - 4.*A*C

        #Determine if we hit the circle, if so this overrides the boundary
        if (det > 0): #We hit the circle

            #Roots of quadratic eq
            s_circ1 = (-1.*B + sqrt(det))/(2.*A)
            s_circ2 = (-1.*B - sqrt(det))/(2.*A)

        else: #We didnt hit the circle, so we must have left

            s_circ1 = -99
            s_circ2 = -99

        #Find the min s that is positive, this is the face we hit
        intersects = [s_left, s_right, s_top, s_bot, s_circ1,s_circ2]
        s_min = min(i for i in intersects if i > 0 )
        face_id = intersects.index(s_min)
        face_map = ["left","right","top","bot","circ1","circ2"]
        face = face_map[face_id]
        print "THIS FACE", face

        #Check if center of path is in fuel, in this case we were in the fuel
        r_cent = (x_prev + xcos*s_min*0.5)**2 + (y_prev + ycos*y_prev*0.5)**2
        if (r_cent < radius*radius): #in the fuel:

            print "THIS PATH WAS IN THE FUEL"
                
            #Contribution to psi from this source is based on flux leaving fuel and how
            #many mfp it traveled to get to this point
            mfp_fuel = s_min*sigma_f
            psi     += q_f/(sigma_f)*(1.-exp(-1.*mfp_fuel))*exp(-1.*n_mfp)
            n_mfp   += mfp_fuel
            in_fuel = False #we can't stay in fuel

        else: #just traveling in moderator

            n_mfp += s_min*sigma_m #We had to have been in moderator

            #For debugging we have a moderator source that we add in. It will contribute
            #however much is at previous point and how far it has had to attenuate
            if debug_mode:
                psi    += q_mod/(sigma_f)*(1.-exp(-1.*s_min*sigma_m))*exp(-1.*(n_mfp - s_min*sigma_m))

        #Move to the new coordinates
        s_min *= 1.+5.E-14                  #Add a small factor so we are for sure on other side of the face
        x_prev = x_prev + xcos*s_min
        y_prev = y_prev + ycos*s_min

        print "I am now here", x_prev, y_prev
        #Determine if we hit a boundary. Either we hit a circle or boundary
        if not re.search("circ",face):
    
            #Move to opposite face
            if face == "left":
                x_prev = x_right*(1-5.E-14)
            elif face == "right":
                x_prev = x_left*(1+5.E-14)
            elif face == "bot":
                y_prev = y_top*(1-5.E-14)
            elif face == "top":
                y_prev = y_bot*(1-5.E-14)
            else:
                raise ValueError("Something wrong in faces")

            hit_boundary = True
            print "I am now over here", x_prev, y_prev

            #Check if we are in a corner, based on symmetry, requires attention
            if x_left == y_bot: #Check for symmetry
                if abs(abs(x_prev) - abs(y_prev)) < 1.E-13*abs(x_left):

                    print "WE HIT A CORNER"
                    
                    #flip the face we haven't flipped yet
                    if face == "left" or face == "right":
                        y_prev *= -1.
                        y_prev = y_prev/abs(y_prev)*y_prev*(1.-5.E-14)
                    else:
                        x_prev *= -1.
                        x_prev = x_prev/abs(x_prev)*x_prev*(1.-5.E-14)


                    print "and now over here", x_prev, y_prev

        print ""


    print "Final solution", psi
    if debug_mode:
        print "Desired solution: ", q_mod/sigma_f, q_mod, q_f
        print "error: ", (q_mod/sigma_f - psi)
        print "tol:   ", psi*tol
    
if __name__ == "__main__":
    main(10,pi/4,tol=2.06115362e-9)
