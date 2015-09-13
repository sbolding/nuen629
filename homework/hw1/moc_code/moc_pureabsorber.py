import numpy as np
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
    sigma_f = 0.1414*10
    sigma_m = 0.08*10
    Q_f_tot = 1.
    Q_f     = Q_f_tot/(4.*pi)
    Q_mod   = 0.0

    debug_mode = False
    if debug_mode:
        Q_mod = Q_f

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
    phi   = pi/4.
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

        #Find the min s that is positive, this is the intersection with boundary
        intersects = [s_left, s_right, s_top, s_bot]
        s_min = min(i for i in intersects if i > 0 )
        face_id = intersects.index(s_min)
        face_map = ["left","right","top","bot"]
        face = face_map[face_id]
        print "THIS FACE", face

        #Roots of parametric equation for circle
        A = xcos**2 + ycos**2
        B = 2.*xcos*x_prev + 2.*ycos*y_prev
        C = x_prev**2 + y_prev**2 - radius**2
        det = B*B - 4.*A*C

        #Determine if we hit the circle, if so this overrides the boundary
        if (det > 0): #We hit the circle

            #Roots of quadratic eq
            s1 = (-1.*B + sqrt(det))/(2.*A)
            s2 = (-1.*B - sqrt(det))/(2.*A)

            #If we were already in the circle, then one is positive and other is negative,
            #Due to roundoff both may be positive
            print "ROOTS OF CIRCLE: ", s1, s2

            #check if one of the roots is zero
            if entered_fuel_last_iter:

                s_circ = max(s1,s2)

                #Verify that circle distance is less than boundary distance
                if s_circ > s_min:
                    raise ValueError("Error in circle logic")

                s_min  = s_circ
                print "I WAS IN THE CIRCLE, LEAVING NOW", s_min
                in_fuel = True
                entered_fuel_last_iter = False

            else: #no sign change
                
                if s1 < 0.0: #No hits, headed the wrong direction

                    print "HEADED AWAY FROM CIRCLE"
                    hit_boundary = True
                    s_min = s_min

                else: #May have just entering the fuel, both roots positive

                    print "JUST HIT THE CIRCLE"
                    s_min = min(s1,s2)
                    print s_min
                    entered_fuel_last_iter = True

        else: #We didnt hit the circle, so we must have left

            print "NO CIRCLE HIT, HEADED TO BOUNDARY"
            s_min = s_min
            hit_boundary = True
            

        #Compute contribution to psi and to total mfp traveled
        if in_fuel:

            #Contribution to psi from this source is based on flux leaving fuel and how
            #many mfp it traveled to get to this point
            mfp_fuel = s_min*sigma_f
            psi     += Q_f*(1.-exp(-1.*mfp_fuel))*exp(-1.*n_mfp)
            n_mfp   += mfp_fuel
            in_fuel = False #we can't stay in fuel

        else:

            n_mfp += s_min*sigma_m #We had to have been in moderator

            #For debugging we have a moderator source that we add in. It will contribute
            #however much is at previous point and how far it has had to attenuate
            if debug_mode:
                psi    += Q_mod*(1.-exp(-1.*smin*sigma_m))*exp(-1.*(n_mfp - smin*sigma_m))


        #Move to the new coordinates
        s_min *= 1.+1.E-14                  #Add a small factor so we are on other side of the face
        x_prev = x_prev + xcos*s_min
        y_prev = y_prev + ycos*s_min

        print "I am now here", x_prev, y_prev
        #If we left the boundary
        if hit_boundary:
    
            #Move to opposite face
            if face == "left":
                x_prev = x_right
            elif face == "right":
                x_prev = x_left
            elif face == "bot":
                y_prev = y_top
            elif face == "top":
                y_prev = y_bot
            else:
                raise ValueError("Something wrong in faces")

            hit_boundary = False
            print "I am now over here", x_prev, y_prev

           #Check if we are in a corner, based on symmetry
            if x_left == y_bot: #Check for symmetry
                if abs(abs(x_prev) - abs(y_prev)) < 1.E-14*abs(x_left):

                    print "WE HIT A CORNER"
                    
                    #flip the face we haven't flipped yet
                    if face == "left" or face == "right":
                        y_prev *= -1.
                    else:
                        x_prev *= -1.

                    print "and now over here", x_prev, y_prev

        print ""



if __name__ == "__main__":
    main(10,pi/2,tol=2.06115362e-9)
