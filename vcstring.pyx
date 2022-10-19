# Source code for implementing the Committor-Consistent Variational String Method (CCVSM)
# Version 1.0
# Roux Group, The University of Chicago
# Author: Ziwei He
# Date: September 1, 2022
#
# For publication about the development and application of this method, please refer to:
# Z. He, C. Chipot, B. Roux, J. Phys. Chem. Lett. 13, 9263–9271 (2022).
# doi: 10.1021/acs.jpclett.2c02529

"""
Implement the committor-consistent variational string method from trajectories.
"""

import sys
import math
import numpy as np
cimport numpy as np
from numpy import exp, zeros, sqrt, square, log, where, pi, mean, average, arange, sin, cos, tan
from copy import copy, deepcopy
import random

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class vcstring:
    cdef str state_assignment
    cdef int num_basis
    cdef double cut_A, cut_B
    cdef double theta_deg, theta, cut
    cdef double ra_A, rb_A, ra_B, rb_B, alpha_deg_A, alpha_deg_B, alpha_A, alpha_B, x0_A, y0_A, x0_B, y0_B

    def __init__(self, str state_assignment, int num_basis, cut_A=None, cut_B=None, theta_deg=None, cut=None, ra_A=None, rb_A=None, ra_B=None, rb_B=None, alpha_deg_A=None, alpha_deg_B=None, x0_A=None, y0_A=None, x0_B=None, y0_B=None):
        """Define basis set

        Parameters
        ----------
        state_assignment : str
            How to assign end states, A and B, and the intermediate state given a set 
            of collective variables. '1d' is for the 1-dimensional space, 'ellipse' bounds the 
            end states in ellipsoids, 'orthogonal_projection' defines end state boundaries based
            on an orthogonal projection of a straight line with respect to the x-axis.

        num_basis: int
            Number of basis functions

        cut_A : float, default=None
            Cut off value in collective variable space for state A in the 1-dimensional space.
            Must be defined when state_assignment == '1d'

        cut_B : float, default=None
            Cut off value in collective variable space for state B in the 1-dimensional space.
            Must be defined when state_assignment == '1d'

        theta_deg : float, default=None
            defines angle of rotation of reaction coordinate with respect to the x-axis in degrees,
            must be defined when state_assignment == 'orthogonal_projection'

        cut : float, default=None
            defines cutoff value along straight path reaction coordinate
            must be defined when state_assignment == 'orthogonal_projection

        ra_A : float, default=None
            major axis of the ellipse that gives the state A boundary,
            must be defined when state_assignment == 'ellipse'

        rb_A : float, default=None
            minor axis of the ellipse that gives the state A boundary,
            must be defined when state_assignment == 'ellipse'

        ra_B : float, default=None
            major axis of the ellipse that gives the state B boundary,
            must be defined when state_assignment == 'ellipse'

        rb_B : float, default=None
            minor axis of the ellipse that gives the state B boundary,
            must be defined when state_assignment == 'ellipse'

        alpha_deg_A : float, default=None
            degree of rotation of state A ellipse,
            must be defined when state_assignment == 'ellipse'

        alpha_deg_B : float, default=None
            degree of rotation of state B ellipse, 
            must be defined when state_assignment == 'ellipse'

        x0_A, y0_A: float, default=None
            Ellipse of state A centered at position (x0_A, y0_A),
            must be defined when state_assignment == 'ellipse'

        x0_B, y0_B : float, default=None
            Ellipse of state B centered at position (x0_B, y0_B),
            must be defined when state_assignment == 'ellipse'
        
        """
        self.num_basis = num_basis  # Number of basis functions
        self.state_assignment = state_assignment
        if state_assignment == '1d':  # for 1-dimensional colvars
            self.cut_A = cut_A
            self.cut_B = cut_B
        if state_assignment == 'ellipse':
            self.ra_A = ra_A
            self.rb_A = rb_A
            self.ra_B = ra_B
            self.rb_B = rb_B
            self.alpha_deg_A = alpha_deg_A
            self.alpha_A = alpha_deg_A * np.pi / 180.0
            self.alpha_deg_B = alpha_deg_B
            self.alpha_B = alpha_deg_B * np.pi / 180.0
            self.x0_A = x0_A
            self.y0_A = y0_A
            self.x0_B = x0_B
            self.y0_B = y0_B
        elif state_assignment == 'orthogonal_projection':
            self.theta_deg = theta_deg  # Degrees
            self.theta = theta_deg * pi / 180.0  # Convert to radians
            self.cut = cut  # Intermediate region cutoff along RC when theta = 0i

    cpdef intermediate_region(self):
        """Define intermediate region bounds for a given theta angle and cut off value.
        
        Returns
        -------
        xa : float, boundary of state A

        xb : float, boundary of state B
        
        """
        cdef double xa, xb
        xa = ((tan(self.theta + pi/2) - 1) * self.cut) / (tan(self.theta) - tan(self.theta + pi/2))
        xb = ((-tan(self.theta + pi/2) + 1) * self.cut) / (tan(self.theta) - tan(self.theta + pi/2))
        return xa, xb

    cpdef centroids(self):
        """Returns the position of images along a straight reaction coordinate
        
        Returns
        -------
        z : ndarray of shape (num_basis, 2)

        """
        cdef double xa, xb
        cdef np.ndarray[DTYPE_t, ndim=1] xcenters, ycenters
        cdef np.ndarray[DTYPE_t, ndim=2] z
        xa, xb = self.intermediate_region()
        xcenters = np.linspace(xa, xb, self.num_basis+1)
        xcenters = (xcenters[:-1] + xcenters[1:]) / 2
        ycenters = tan(self.theta) * xcenters
        z = np.c_[xcenters, ycenters]
        return z

    cpdef assign_states(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z):
        """
        Assignment to Voronoi cells. For a single trajectory.

        Parameters
        ----------
        traj : list of trajectory with shape (T, d),
            where T is the number of time steps and d is the number of dimensions

        tau : int
            lag time step

        z : ndarray(T, 2)
            image positions

        Returns
        -------
        ids : ndarray(T, V)
            where T is the number of time steps corresponding to the trajectory and V is the 
            number of columns depending on the state_assignment type. 
            If state_assignment == '1d' the columns are [h_I, h_B, nc, x_t].
            If state_assignment == 'orthogonal_projection', columns are [h_I, h_B, nc, x_t, y_t].
            If state_assignment == 'ellipse', columns are [h_I, h_B, nc, x_t, y_t, ell_tA, ell_tB].
        """
        cdef double xa, xb
        cdef double ra_A, rb_A, alpha_A, x0_A, y0_A, ell_tA, ell_tauA
        cdef double ra_B, rb_B, alpha_B, x0_B, y0_B, ell_tB, ell_tauB
        cdef np.ndarray[DTYPE_t, ndim=1] xcenters, ycenters
        cdef int N, i, j, t
        cdef list x, y, D_0, D_tau, g_0, g_tau
        cdef np.ndarray[DTYPE_t, ndim=1] dist
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        cdef double x_t, y_t, nc, u_t, h
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau
        cdef np.ndarray[DTYPE_t, ndim=2] D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b

        print('\n=========================')
        print('VORONOI STATE ASSIGNMENT')
        print('=========================')
        print('State_assignment type:', self.state_assignment)
        if self.state_assignment == 'orthogonal_projection':
            xa, xb = self.intermediate_region()

        N = len(traj)       # Length of trajectory
        x = [row[1] for row in traj]
        xcenters = z[:, 0]
        if self.state_assignment != '1d':
            ycenters = z[:, 1]  # y coordinates of centroids
            y = [row[2] for row in traj]
        print('Voronoi centroid positions:\n', z)

        # Check that lag time is smaller than data set size
        if tau > N:
            raise RuntimeError ("\nInvalid lag time. Use a smaller lag time.")
            exit()

        # Assign to Voronoi cell
        ids = []
        print('Performing Voronoi tesselation')
        sys.stdout.flush()
        for t in range(N):
            # Get snapshot at time t and t + tau
            h_I = 0.0
            h_B = 0.0
            x_t = x[t]
            if self.state_assignment == '1d':
                dist = abs(x_t - xcenters) # Find nearest Voronoi center
            else:
                y_t = y[t]
                dist = sqrt((x_t - xcenters)**2 + (y_t - ycenters)**2) # Find nearest Voronoi center
            nc = np.argmin(dist)  # Get index of nearest Voronoi

            if self.state_assignment == '1d':
                if (x_t >= self.cut_A) and (x_t <= self.cut_B):  # State I
                    h_I = 1.0
                elif x_t > self.cut_B:  # State B
                    h_B = 1.0

            # Orthogonal projection onto the RC line, then check for x boundary
            elif self.state_assignment == 'orthogonal_projection':
                # Orthogonal projection of (x,y) -> (u,v) onto reaction coordinate
                u_t = (y_t * tan(self.theta) + x_t) / (tan(self.theta)**2 + 1)

                # Check if (u,v) is inside/outside of the intermediate region
                # indicator function h=0 in state A, 0.5 in intermediate region, 1 in state B
                if u_t >= xa and u_t <= xb:  # Intermediate region
                    h_I = 1.0
                elif u_t > xb:  # Ouside of intermediate region (state B)
                    h_B = 1.0

            # Fits rotated ellipses to the well minima to define states A, B
            elif self.state_assignment == 'ellipse':
                # Value of fitted ellipse at point (x_t, y_t) or (x_tau, y_tau)
                ell_tA = elliptical(self.ra_A, self.rb_A, self.alpha_A, self.x0_A, self.y0_A, x_t, y_t)
                ell_tB = elliptical(self.ra_B, self.rb_B, self.alpha_B, self.x0_B, self.y0_B, x_t, y_t)
                
                # Assign to states
                if ell_tA > 1.0 and ell_tB > 1.0:  # Intermediate state
                    h_I = 1.0
                elif ell_tB <= 1.0:  # Outside intermediate region (state B)
                    h_B = 1.0

            # Append ID assignments at each timestep
            if self.state_assignment == '1d':
                ids.append([h_I, h_B, nc, x_t])
            elif self.state_assignment == 'orthogonal_projection':
                ids.append([h_I, h_B, nc, x_t, y_t])
            elif self.state_assignment == 'ellipse':
                ids.append([h_I, h_B, nc, x_t, y_t, ell_tA, ell_tB]) # state I, state B, Voronoi index, x pos, y pos 
            else:
                raise TypeError('Incorrect state assignment. Chosen state_assignment method is not available.')
        ids = np.asarray(ids)
        print('Voronoi assignment completed')
        return ids 


    cpdef assign_states_ensemble(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, int t0=0, list weights=None, stride=None):
        """
        Assign points to Voronoi cells for an ensemble of aggregate trajectories.

        Parameters
        ----------
        traj : list of trajectory with shape (T, d),
            where T is the number of time steps and d is the number of dimensions

        tau : int
            lag time step

        z : ndarray(T, 2)
            image positions

        t0 : int, default=0
            smeow

        weights : optional, default=None
            weight of each trajectory

        stride : optional, default=None
            use every stride-th frame of a trajectory. Cuts down the volume of data, for speed up.

        Returns
        -------
        ids : ndarray(T, V)
            where T is the number of time steps corresponding to the trajectory and V is the
            number of columns depending on the state_assignment type.
            If state_assignment == '1d' the columns are [h_I0, h_B0, h_Itau, h_Btau, nc_t, nc_tau, x_t, x_tau, weights.
            If state_assignment == 'orthogonal_projection' the columns are [h_I0, h_B0, h_Itau, h_Btau, nc_t, nc_tau, x_t, y_t, x_tau, y_tau, weights]
            If state_assignment == 'ellipse' the columns are [h_I0, h_B0, h_Itau, h_Btau, nc_t, nc_tau, x_t, y_t, x_tau, y_tau, weights, ell_tA, ell_tB] 

        """
        cdef double xa, xb
        cdef double ra_A, rb_A, alpha_A, x0_A, y0_A, ell_tA, ell_tauA
        cdef double ra_B, rb_B, alpha_B, x0_B, y0_B, ell_tB, ell_tauB
        cdef np.ndarray[DTYPE_t, ndim=1] xcenters, ycenters
        cdef int N, i, j, t
        cdef list x, y, D_0, D_tau, g_0, g_tau
        cdef np.ndarray[DTYPE_t, ndim=1] dist
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        cdef double x_t, y_t, nc, u_t, h
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau
        cdef np.ndarray[DTYPE_t, ndim=2] D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b

        print('\n===================================')
        print('VORONOI STATE ASSIGNMENT - ENSEMBLE')
        print('===================================')
        print('Voronoi centroid positions:\n', z)
        print('State_assignment type:', self.state_assignment)
        if self.state_assignment == 'orthogonal_projection':
            xa, xb = self.intermediate_region()

        if stride == None:
            stride = 1
        N = len(traj)       # Number of aggregate trajectories in ensemble
        if weights == None:
            weights = [1.] * N
        else:
            print('Weights shape =', np.shape(weights))
        xcenters = z[:, 0]
        if self.state_assignment != '1d':
            ycenters = z[:, 1]  # y coordinates of centroids
        
        # Check that lag time is smaller than data set size
        if tau > len(traj[0]):
            raise RuntimeError ("\nInvalid lag time. Use a smaller lag time.")
            exit()

        # Assign to Voronoi cell
        ids = []
        # print('Performing Voronoi tesselation')
        sys.stdout.flush()
        count = 0
        for t in range(N):
            # Get snapshot at time t and t + tau
            h_I0, h_Itau = 0., 0.
            h_B0, h_Btau = 0., 0.

            # Get frame coordinates and closest Voronoi at time 0 and time tau
            x_t = [row[1] for row in traj[t]][::stride][t0]
            x_tau = [row[1] for row in traj[t]][::stride][t0+tau]
            if self.state_assignment == '1d':
                dist_t = abs(x_t - xcenters) # Find nearest Voronoi center
                dist_tau = abs(x_tau - xcenters)
            else:
                y_t = [row[2] for row in traj[t]][::stride][t0]
                y_tau = [row[2] for row in traj[t]][::stride][t0+tau]
                dist_t = sqrt((x_t - xcenters)**2 + (y_t - ycenters)**2)  # Find nearest Voronoi center
                dist_tau = sqrt((x_tau - xcenters)**2 + (y_tau - ycenters)**2)
            nc_t = np.argmin(dist_t)  # Get index of nearest Voronoi
            nc_tau = np.argmin(dist_tau)

            # Assign states given boundary definitions
            # If 1-dimensional system
            if self.state_assignment == '1d':
                if (x_t >= self.cut_A) and (x_t <= self.cut_B):  # State I
                    h_I0 = 1.
                elif x_t > self.cut_B:  # State B
                    h_B0 = 1.
                if (x_tau >= self.cut_A) and (x_tau <= self.cut_B):  # State I
                    h_Itau = 1.
                elif x_tau > self.cut_B:  # State B
                    h_Btau = 1.

            # Orthogonal projection onto the RC line, then check for x boundary
            elif self.state_assignment == 'orthogonal_projection':
                # Orthogonal projection of (x,y) -> (u,v) onto reaction coordinate
                u_t = (y_t * tan(self.theta) + x_t) / (tan(self.theta)**2 + 1)
                u_tau = (y_tau * tan(self.theta) + x_tau) / (tan(self.theta)**2 + 1)
                # Check if (u,v) is inside/outside of the intermediate region
                # indicator function h=0 in state A, 0.5 in intermediate region, 1 in state B
                if u_t >= xa and u_t <= xb:  # Intermediate region
                    h_I0 = 1.0
                elif u_t > xb:  # Ouside of intermediate region (state B)
                    h_B0 = 1.0
                if u_tau >= xa and u_tau <= xb:  # Intermediate region
                    h_Itau = 1.0
                elif u_tau > xb:  # Ouside of intermediate region (state B)
                    h_Btau = 1.0

            # Fits rotated ellipses to the well minima to define states A, B
            elif self.state_assignment == 'ellipse':
                # Value of fitted ellipse at point (x_t, y_t) or (x_tau, y_tau)
                ell_tA = elliptical(self.ra_A, self.rb_A, self.alpha_A, self.x0_A, self.y0_A, x_t, y_t)
                ell_tB = elliptical(self.ra_B, self.rb_B, self.alpha_B, self.x0_B, self.y0_B, x_t, y_t)
                ell_tauA = elliptical(self.ra_A, self.rb_A, self.alpha_A, self.x0_A, self.y0_A, x_tau, y_tau)
                ell_tauB = elliptical(self.ra_B, self.rb_B, self.alpha_B, self.x0_B, self.y0_B, x_tau, y_tau)
                if ell_tA > 1.0 and ell_tB > 1.0:  # Intermediate state
                    h_I0 = 1.0
                elif ell_tB <= 1.0:  # State B
                    h_B0 = 1.0
                if ell_tauA > 1.0 and ell_tauB > 1.0:  # Intermediate state
                    h_Itau = 1.0
                elif ell_tauB <= 1.0:  # State B
                    h_Btau = 1.0

            # Append ID assignments for each trajectory
            # h_I0(0), h_B(0), h_I0(tau), h_B(tau), Voronoi index, x(0), y(0), x(tau), y(tau)
            if self.state_assignment == '1d':
                ids.append([h_I0, h_B0, h_Itau, h_Btau, nc_t, nc_tau, x_t, x_tau, weights[count]])
            elif self.state_assignment == 'orthogonal_projection':
                ids.append([h_I0, h_B0, h_Itau, h_Btau, nc_t, nc_tau, x_t, y_t, x_tau, y_tau, weights[count]])
            elif self.state_assignment == 'ellipse':
                ids.append([h_I0, h_B0, h_Itau, h_Btau, nc_t, nc_tau, x_t, y_t, x_tau, y_tau, weights[count], ell_tA, ell_tB]) 
            else:
                raise TypeError('Incorrect state assignment. Chosen state_assignment method is not available.')
            count += 1
        ids = np.asarray(ids)
        print('Voronoi assignment completed')
        return ids

    
    cpdef calc_coeffs(self, int N, int tau, np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=2] ids, symmetric=True):
        """
        Solves the linear equation from minimizing the forward steady-state flux:
        (D(tau) - D(0)) b = g(0) - g(tau)
        Best implemented for a single, well sampled trajectory.

        Parameters
        ----------
        N : int
            number of frames in trajectory

        tau : int
            lag time steps

        z : ndarray(M, 2)
            M image positions on a string

        ids : ndarray(T, V)
            Voronoi index assignment to discrete basis functions, where T is the number of time
            steps and V are columns with assignment details

        symmetric : bool, default=True
            enforces matrix symmetry, 
            for symmetric systems, e.g., a symmetric double well

        Returns
        -------
        b : 1D array of length M
            Coefficients from solving the linear equation. 
            This gives the trial committor probabilities corresponding to the images of z.

        D_0_arr : ndarray(M, M)
            Square D matrix at time 0

        D_tau_arr : ndarray(M, M)
            Square D matrix at time tau

        g_0_arr : 1D array of length M
            vector g at time 0

        g_tau_arr : 1D array of length M
            vector g at time tau
            
        """
        cdef int i, j, t
        # cdef np.ndarray[DTYPE_t, ndim=1] h_I0, h_Itau, h_B0, h_Btau, centroid
        cdef np.ndarray[DTYPE_t, ndim=1] centroid
        cdef np.ndarray[np.int64_t, ndim=1] where_i0, where_itau, where_j
        cdef list x, y, D_0, D_tau, g_0, g_tau, test
        # cdef np.ndarray[DTYPE_t, ndim=1] dist
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        # cdef double x_t, y_t, nc, u_t, h
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau
        cdef np.ndarray[DTYPE_t, ndim=2] D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b
        
        print('\n======================================')
        print('SOLVING LINEAR BASIS SET COEFFICIENTS')
        print('======================================')
        print('lag time =', tau)
        print('Symmetrization of D matrices:', symmetric)
        centroid = ids[:, 2]  # Indices for Voronoi centroid assignment
        D_0 = []
        D_tau = []
        g_0 = []
        g_tau = []
        for i in range(self.num_basis):
            val_g_0, val_g_tau = 0.0, 0.0
            # test = []  # for testing
            for j in range(self.num_basis):
                print("Matrix elements", i, j)
                sys.stdout.flush()
                val_D_0, val_D_tau = 0.0, 0.0
                for t in range(N-tau):
                    f_i0, f_itau, f_j = 0.0, 0.0, 0.0
                    h_I0 = ids[:, 0][t]
                    h_Itau = ids[:, 0][t + tau]
                    h_B0 = ids[:, 1][t]
                    h_Btau = ids[:, 1][t + tau]
                    if centroid[t] == i:
                        f_i0 = 1.0
                    if centroid[t+tau] == i:
                        f_itau = 1.0
                    if centroid[t] == j:
                        f_j = 1.0

                    # Summation of terms
                    val_D_0 += h_I0 * h_I0 * f_i0 * f_j
                    val_D_tau += h_Itau * h_I0 * f_itau * f_j
                    if j == 0:
                        val_g_0 += (h_I0 * f_i0 * h_B0) + (h_B0 * h_I0 * f_i0)
                        val_g_tau += (h_Itau * f_itau * h_B0) + (h_Btau * h_I0 * f_i0)

                # Divide by data size
                D_0.append(val_D_0 / (N - tau))
                D_tau.append(val_D_tau / (N - tau))
            g_0.append(val_g_0 / (N - tau))
            g_tau.append(val_g_tau / (N - tau))

        # Reshape into appropriate matrix size
        D_0_arr = np.asarray(D_0).reshape(self.num_basis, self.num_basis)
        D_tau_arr = np.asarray(D_tau).reshape(self.num_basis, self.num_basis)
        g_0_arr = np.asarray(g_0) # .reshape(self.num_basis)
        g_tau_arr = np.asarray(g_tau)# .reshape(self.num_basis)
        print('D_0 =', D_0_arr)
        print('D_tau =', D_tau_arr)
        print('g_0 =', g_0_arr)
        print('g_tau =', g_tau_arr)

        # Symmetrize for reversible dynamics
        if symmetric == True:
            D_0_arr = 0.5 * (D_0_arr + D_0_arr.T)
            D_tau_arr = 0.5 * (D_tau_arr + D_tau_arr.T)
            print('Symmetric D_0 =', D_0_arr)
            print('Symmetric D_tau =', D_tau_arr)

        # Calculate committor coefficients, b
        D = D_0_arr - D_tau_arr
        g = g_tau_arr - g_0_arr
        b = 0.5 * np.linalg.inv(D).dot(g)  
        print('b =', b)

        return b, D_0_arr, D_tau_arr, g_0_arr, g_tau_arr

    cpdef calc_coeffs_accelerated(self, int image, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z):
        """
        Accelerated method to solve the linear equation from minimizing the forward steady-state flux:
        (D(tau) - D(0)) b = g(0) - g(tau)
        Best implemented for a single, well sampled trajectory. 
        This accelerated method calculates only the rows/column elements of the affected image 
        and its nearest neighboring images.

        WARNING: THIS METHOD IS NOT RECOMMENDED FOR MOST CASES. MOVING THE IMAGES 
        MORE THAN A CERTAIN AMOUNT CAN AFFECT THE VORONOI TESSELATION IN FARTHER OUT IMAGES 
        OTHER THAN THE NEAREST NEIGHBORS.

        Parameters
        ----------
        image : int
            index of image on strong

        traj : list
            trajectory of length T time steps

        tau : int
            lag time steps

        z : ndarray(M, 2)
            M image positions on a string

        Returns
        -------
        b : 1D array of length M
            Coefficients from solving the linear equation.
            This gives the trial committor probabilities corresponding to the images of z.

        D_0_arr : ndarray(M, M)
            Square D matrix at time 0

        D_tau_arr : ndarray(M, M)
            Square D matrix at time tau

        g_0_arr : 1D array of length M
            vector g at time 0

        g_tau_arr : 1D array of length M
            vector g at time tau

        """
        cdef double xa, xb
        cdef np.ndarray[DTYPE_t, ndim=1] xcenters, ycenters
        cdef int N, i, j, t
        cdef list x, y # , D_0, D_tau, g_0, g_tau
        cdef list affected_images
        # cdef np.ndarray dat
        cdef np.ndarray[DTYPE_t, ndim=2] D_0, D_tau
        cdef np.ndarray[DTYPE_t, ndim=1] g_0, g_tau
        cdef np.ndarray[DTYPE_t, ndim=1] dist_t, dist_tau
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        cdef double x_t, y_t, x_tau, y_tau, nc_t, nc_tau, u_t, u_tau
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau
        cdef np.ndarray[DTYPE_t, ndim=2] D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b

        print('\nSOLVING LINEAR BASIS SET COEFFICIENTS, ACCELERATED METHOD')
        print('-----------------------------------------------------------')
        print("WARNING: THIS METHOD IS NOT RECOMMENDED FOR MOST CASES BECAUSE MOVING THE IMAGES MORE THAN A CERTAIN AMOUNT CAN AFFECT THE VORONOI TESSELATION IN FARTHER OUT IMAGES OTHER THAN THE NEAREST NEIGHBORS.")
        xa, xb = self.intermediate_region()
        xcenters = z[:, 0]  # x coordinates of centroids
        ycenters = z[:, 1]  # y coordinates of centroids
        print('Image', image)
        print('Voronoi centroid positions:\n', z)
        print('theta deg', self.theta_deg)
        print('theta rad', self.theta)

        N = len(traj)       # Length of trajectory
        x = [row[1] for row in traj]
        y = [row[2] for row in traj]

        # Check that lag time is smaller than data set size
        if tau > N:
            raise RuntimeError ("\nInvalid lag time. Use a smaller lag time.")
            exit()

        # Compute matrix/vector elements
        dat = np.load('output/coeffs_input.npz')
        D_0 = dat['D_0']
        D_tau = dat['D_tau']
        g_0 = dat['g_0']
        g_tau = dat['g_tau']
        b = dat['b']
        print('\nInput data')
        print('D_0 =', D_0)
        print('D_tau =', D_tau)
        print('g_0 =', g_0)
        print('g_tau =', g_tau)
        print('b =', b)

        # Find the images that cause changes to matrix/vector elements
        # indices should be the specified image and it's nearest neighbors: image-1, image, image+1
        if image == 0:
            affected_images = [image, image+1]
        elif image == self.num_basis-1:
            affected_images = [image-1, image]
        else: 
            affected_images = [image-1, image, image+1]
        print('\nAffected images:', affected_images)

        for i in range(len(xcenters)):
            # val_g_0, val_g_tau = 0.0, 0.0
            for j in range(len(xcenters)):
                if i in affected_images or j in affected_images: # Update specific elements
                    print("Matrix elements", i, j)
                    sys.stdout.flush()
                    D_0[i][j] = 0.0
                    D_tau[i][j] = 0.0
                    if j == 0:
                        g_0[i] = np.sum(g_0[0:i])
                        g_tau[i] = np.sum(g_tau[0:i])

                    for t in range(N - tau):
                        # Get snapshot at time t and t + tau
                        x_t = x[t]
                        y_t = y[t]
                        x_tau = x[t + tau]
                        y_tau = y[t + tau]

                        # Find nearest Voronoi center from (x,y)
                        dist_t = sqrt((x_t - xcenters)**2 + (y_t - ycenters)**2)
                        dist_tau = sqrt((x_tau - xcenters)**2 + (y_tau - ycenters)**2)
                        nc_t = np.argmin(dist_t)
                        nc_tau = np.argmin(dist_tau)

                        # Orthogonal projection of (x,y) -> (u,v) onto reaction coordinate
                        u_t = (y_t * tan(self.theta) + x_t) / (tan(self.theta)**2 + 1)
                        u_tau = (y_tau * tan(self.theta) + x_tau) / (tan(self.theta)**2 + 1)

                        # Check if (u,v) is inside/outside of the intermediate region
                        f_i0, f_itau, f_j = 0.0, 0.0, 0.0  # Voronoi basis functions
                        h_I0, h_B0, h_Itau, h_Btau = 0.0, 0.0, 0.0, 0.0  # Indicator functions of states

                        if u_t >= xa and u_t <= xb:  # Intermediate region
                            h_I0 = 1.0
                            if i == nc_t:
                                f_i0 = 1.0
                            if j == nc_t:
                                f_j = 1.0
                        elif u_t > xb:  # Ouside of intermediate region (state B)
                            h_B0 = 1.0

                        if u_tau >= xa and u_tau <= xb:  # Intermediate region
                            h_Itau = 1.0
                            if i == nc_tau:
                                f_itau = 1.0
                        elif u_tau > xb:  # Outside intermediate region (state B)
                            h_Btau = 1.0

                        # Summation of terms
                        D_0[i][j] += h_I0 * h_I0 * f_i0 * f_j
                        D_tau[i][j] += h_Itau * h_I0 * f_itau * f_j
                        if j == 0:
                            g_0[i] += ((h_I0 * f_i0 * h_B0) + (h_B0 * h_I0 * f_i0)) / (N - tau)
                            g_tau[i] += ((h_Itau * f_itau * h_B0) + (h_Btau * h_I0 * f_i0)) / (N - tau)

                    # Divide by data size
                    D_0[i][j] = D_0[i][j] / (N - tau)
                    D_tau[i][j] = D_tau[i][j] / (N - tau)

        # Reshape into appropriate matrix size
        D_0_arr = np.asarray(D_0).reshape(self.num_basis, self.num_basis)
        D_tau_arr = np.asarray(D_tau).reshape(self.num_basis, self.num_basis)
        g_0_arr = np.asarray(g_0) # .reshape(self.num_basis)
        g_tau_arr = np.asarray(g_tau)# .reshape(self.num_basis)

        # Symmetrize for reversible dynamics
        # D_0_sym = 0.5 * (D_0_arr + D_0_arr.T)
        # D_tau_sym = 0.5 * (D_tau_arr + D_tau_arr.T)

        # Calculate committor coefficients
        # D = D_0_sym - D_tau_sym
        D = D_0_arr - D_tau_arr
        g = g_tau_arr - g_0_arr
        b = 0.5 * np.linalg.inv(D).dot(g)
        print('D_0 =', D_0_arr)
        print('D_tau =', D_tau_arr)
        print('g_0 =', g_0_arr)
        print('g_tau =', g_tau_arr)
        print('b =', b)

        return b, D_0_arr, D_tau_arr, g_0_arr, g_tau_arr


    cpdef calc_coeffs_ensemble(self, int N, int tau, np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=2] ids, symmetric=True):
        """
        Solves the linear equation from minimizing the forward steady-state flux:
        (D(tau) - D(0)) b = g(0) - g(tau)
        Must use this method for an ensemble of trajectories - this is particularly useful 
        for trajectories from enhanced sampling simulations.


        Parameters
        ----------
        N : int
            number of independent trajectories

        tau : int
            lag time steps

        z : ndarray(M, 2)
            M image positions on a string

        ids : ndarray(T, V)
            Voronoi index assignment to discrete basis functions, where T is the number of time
            steps and V are columns with assignment details

        symmetric : bool, default=True
            enforces matrix symmetry,
            for symmetric systems, e.g., a symmetric double well

        Returns
        -------
        b : 1D array of length M
            Coefficients from solving the linear equation.
            This gives the trial committor probabilities corresponding to the images of z.

        D_0_arr : ndarray(M, M)
            Square D matrix at time 0

        D_tau_arr : ndarray(M, M)
            Square D matrix at time tau

        g_0_arr : 1D array of length M
            vector g at time 0

        g_tau_arr : 1D array of length M
            vector g at time tau

        """
        cdef int i, j, t
        # cdef np.ndarray[DTYPE_t, ndim=1] h_I0, h_Itau, h_B0, h_Btau, centroid
        cdef np.ndarray[DTYPE_t, ndim=1] centroid
        cdef np.ndarray[np.int64_t, ndim=1] where_i0, where_itau, where_j
        cdef list x, y, D_0, D_tau, g_0, g_tau, test
        # cdef np.ndarray[DTYPE_t, ndim=1] dist
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        # cdef double x_t, y_t, nc, u_t, h
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau, weight
        cdef np.ndarray[DTYPE_t, ndim=2] D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b

        print('\n======================================')
        print('SOLVING LINEAR BASIS SET COEFFICIENTS')
        print('======================================')
        print('lag time =', tau)
        print('Number of trajs:', N)
        print('Symmetrization of D matrices:', symmetric)

        D_0 = []
        D_tau = []
        g_0 = []
        g_tau = []
        for i in range(self.num_basis):
            val_g_0, val_g_tau = 0.0, 0.0
            for j in range(self.num_basis):
                print("Matrix elements", i, j)
                sys.stdout.flush()
                val_D_0, val_D_tau = 0.0, 0.0
                for t in range(N):
                    f_i0, f_itau, f_j = 0.0, 0.0, 0.0
                    h_I0 = ids[t][0]
                    h_Itau = ids[t][2]
                    h_B0 = ids[t][1]
                    h_Btau = ids[t][3]
                    if ids[t][4] == i:  # Match Voronoi centroids
                        f_i0 = 1.0
                    if ids[t][5] == i:
                        f_itau = 1.0
                    if ids[t][4] == j:
                        f_j = 1.0

                    # Summation of terms
                    if self.state_assignment == '1d':
                        weight = ids[t][8]
                    else:
                        weight = ids[t][10]
                    val_D_0 += h_I0 * h_I0 * f_i0 * f_j * weight
                    val_D_tau += h_Itau * h_I0 * f_itau * f_j * weight
                    if j == 0:
                        val_g_0 += ((h_I0 * f_i0 * h_B0) + (h_B0 * h_I0 * f_i0)) * weight
                        val_g_tau += ((h_Itau * f_itau * h_B0) + (h_Btau * h_I0 * f_i0)) * weight

                # Divide by data size
                D_0.append(val_D_0 / N)
                D_tau.append(val_D_tau / N)
            g_0.append(val_g_0 / N)
            g_tau.append(val_g_tau / N)

        # Reshape into appropriate matrix size
        D_0_arr = np.asarray(D_0).reshape(self.num_basis, self.num_basis)
        D_tau_arr = np.asarray(D_tau).reshape(self.num_basis, self.num_basis)
        g_0_arr = np.asarray(g_0) # .reshape(self.num_basis)
        g_tau_arr = np.asarray(g_tau)# .reshape(self.num_basis)
        print('D_0 =', D_0_arr)
        print('D_tau =', D_tau_arr)
        print('g_0 =', g_0_arr)
        print('g_tau =', g_tau_arr)

        # Symmetrize for reversible dynamics
        if symmetric == True:
            D_0_arr = 0.5 * (D_0_arr + D_0_arr.T)
            D_tau_arr = 0.5 * (D_tau_arr + D_tau_arr.T)
            print('Symmetric D_0 =', D_0_arr)
            print('Symmetric D_tau =', D_tau_arr)

        # Calculate committor coefficients, b
        D = D_0_arr - D_tau_arr
        g = g_tau_arr - g_0_arr
        b = 0.5 * np.linalg.inv(D).dot(g)
        print('b =', b)

        return b, D_0_arr, D_tau_arr, g_0_arr, g_tau_arr


    cpdef calc_coeffs_ensemble_timeavg(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, list weights=None, t_final=None, symmetric=True):
        """
        Solves the linear equation from minimizing the forward steady-state flux
        (D(tau) - D(0)) b = g(0) - g(tau)
        using the time average of an ensemble of trajectories.

        Parameters
        ----------
        traj : list of arrays or ndarrays
            trajectory in the collective variable space

        tau : int
            lag time steps

        z : ndarray(M, 2)
            M image positions on a string

        weight : list, optional, default=None
            assigns a weight for each independent trajectory

        t_final : int, optional, default=None
            final time step in trajectory, use this to truncate the trajectory

        symmetric : bool, default=True
            enforces matrix symmetry,
            for symmetric systems, e.g., a symmetric double well
 
        Returns
        -------
        b : 1D array of length M
            Coefficients from solving the linear equation.
            This gives the trial committor probabilities corresponding to the images of z.

        D_0_arr : ndarray(M, M)
            Square D matrix at time 0

        D_tau_arr : ndarray(M, M)
            Square D matrix at time tau

        g_0_arr : 1D array of length M
            vector g at time 0

        g_tau_arr : 1D array of length M
            vector g at time tau

        """
        cdef int i, j, count, t0, t
        cdef np.ndarray[DTYPE_t, ndim=1] centroid
        cdef np.ndarray[np.int64_t, ndim=1] where_i0, where_itau, where_j
        cdef list x, y, D_0, D_tau, g_0, g_tau, test
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau, weight
        cdef np.ndarray[DTYPE_t, ndim=2] ids, D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b

        print('\n======================================')
        print('SOLVING LINEAR BASIS SET COEFFICIENTS')
        print('======================================')
        print('lag time =', tau)
        # print('Number of trajs:', N)
        print('Symmetrization of D matrices:', symmetric)
        
        N = len(traj)
        D_0 = []
        D_tau = []
        g_0 = []
        g_tau = []
        for i in range(self.num_basis):
            val_g_0, val_g_tau = 0.0, 0.0
            for j in range(self.num_basis):
                print("Matrix elements", i, j)
                sys.stdout.flush()
                val_D_0, val_D_tau = 0.0, 0.0
                count = 0
                for t0 in range(len(traj[0])):
                    print('\nt0:', t0)
                    if t_final == None:
                        if t0 + tau >= len(traj[0]):
                            print('out of index')
                            break
                    else:
                        if t0 + tau >= t_final:
                            print('out of index')
                            break
                    
                    ids = self.assign_states_ensemble(traj, tau, z, t0=t0, weights=weights)
                    for t in range(N):
                        f_i0, f_itau, f_j = 0.0, 0.0, 0.0
                        h_I0 = ids[t][0]
                        h_Itau = ids[t][2]
                        h_B0 = ids[t][1]
                        h_Btau = ids[t][3]
                        if ids[t][4] == i:  # Match Voronoi centroids
                            f_i0 = 1.0
                        if ids[t][5] == i:
                            f_itau = 1.0
                        if ids[t][4] == j:
                            f_j = 1.0

                        # Summation of terms
                        if self.state_assignment == '1d':
                            weight = ids[t][8]
                        else:
                            weight = ids[t][10]
                        val_D_0 += h_I0 * h_I0 * f_i0 * f_j * weight
                        val_D_tau += h_Itau * h_I0 * f_itau * f_j * weight
                        if j == 0:
                            val_g_0 += ((h_I0 * f_i0 * h_B0) + (h_B0 * h_I0 * f_i0)) * weight
                            val_g_tau += ((h_Itau * f_itau * h_B0) + (h_Btau * h_I0 * f_i0)) * weight
                        count += 1

                # Divide by data size
                print('Dividing D by data size\n')
                D_0.append(val_D_0 / count)
                D_tau.append(val_D_tau / count)
            g_0.append(val_g_0 / count)
            g_tau.append(val_g_tau / count)

        # Reshape into appropriate matrix size
        D_0_arr = np.asarray(D_0).reshape(self.num_basis, self.num_basis)
        D_tau_arr = np.asarray(D_tau).reshape(self.num_basis, self.num_basis)
        g_0_arr = np.asarray(g_0) # .reshape(self.num_basis)
        g_tau_arr = np.asarray(g_tau)# .reshape(self.num_basis)
        print('D_0 =', D_0_arr)
        print('D_tau =', D_tau_arr)
        print('g_0 =', g_0_arr)
        print('g_tau =', g_tau_arr)

        # Symmetrize for reversible dynamics
        if symmetric == True:
            D_0_arr = 0.5 * (D_0_arr + D_0_arr.T)
            D_tau_arr = 0.5 * (D_tau_arr + D_tau_arr.T)
            print('Symmetric D_0 =', D_0_arr)
            print('Symmetric D_tau =', D_tau_arr)

        # Calculate committor coefficients, b
        D = D_0_arr - D_tau_arr
        g = g_tau_arr - g_0_arr
        b = 0.5 * np.linalg.inv(D).dot(g)
        print('b =', b)

        return b, D_0_arr, D_tau_arr, g_0_arr, g_tau_arr


    cpdef calc_coeffs_ensemble_timeavg_fast(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, list weights=None, symmetric=True):
        """
        Solves the linear equation from minimizing the forward steady-state flux
        (D(tau) - D(0)) b = g(0) - g(tau)
        using the time average of an ensemble of trajectories.

        Parameters
        ----------
        traj : list of arrays or ndarrays
            trajectory in the collective variable space

        tau : int
            lag time steps

        z : ndarray(M, 2)
            M image positions on a string

        weight : list, optional, default=None
            assigns a weight for each independent trajectory

        t_final : int, optional, default=None
            final time step in trajectory, use this to truncate the trajectory

        symmetric : bool, default=True
            enforces matrix symmetry,
            for symmetric systems, e.g., a symmetric double well

        Returns
        -------
        b : 1D array of length M
            Coefficients from solving the linear equation.
            This gives the trial committor probabilities corresponding to the images of z.

        D_0_arr : ndarray(M, M)
            Square D matrix at time 0

        D_tau_arr : ndarray(M, M)
            Square D matrix at time tau

        g_0_arr : 1D array of length M
            vector g at time 0

        g_tau_arr : 1D array of length M
            vector g at time tau

        """
        cdef int i, j, count, t0, t
        cdef np.ndarray[DTYPE_t, ndim=1] centroid
        cdef np.ndarray[np.int64_t, ndim=1] where_i0, where_itau, where_j
        cdef list x, y, D_0, D_tau, g_0, g_tau, test
        cdef double val_g_0, val_g_tau, val_D_0, val_D_tau
        cdef double f_i0, f_itau, f_j, h_I0, h_B0, h_Itau, h_Btau, weight
        cdef np.ndarray[DTYPE_t, ndim=2] ids, D_0_arr, D_tau_arr, D_0_sym, D_tau_sym, D
        cdef np.ndarray[DTYPE_t, ndim=1] g_0_arr, g_tau_arr, g, b

        print('\n======================================')
        print('SOLVING LINEAR BASIS SET COEFFICIENTS')
        print('======================================')
        print('lag time =', tau)
        # print('Number of trajs:', N)
        print('Symmetrization of D matrices:', symmetric)

        N = len(traj)
        if weights == None:
            print('Using uniform weights')
            weights = [1.] * N

        D_0 = []
        D_tau = []
        g_0 = []
        g_tau = []
        for i in range(self.num_basis):
            val_g_0, val_g_tau = 0.0, 0.0
            for j in range(self.num_basis):
                print("\nMatrix elements", i, j)
                sys.stdout.flush()
                val_D_0, val_D_tau = 0.0, 0.0
                count = 0

                for k in range(N):  # Iterate over number of aggregate trajectories
                    # print('Traj', k)
                    ids = self.assign_states(traj[k], tau, z)
                    weight = weights[k]
                    for t in range(len(traj[k])-tau):  # Iterate over trajectory steps
                        f_i0, f_itau, f_j = 0.0, 0.0, 0.0
                        h_I0 = ids[t][0]
                        h_Itau = ids[t + tau][0]
                        h_B0 = ids[t][1]
                        h_Btau = ids[t + tau][1]
                        if ids[t][2] == i:  # Match Voronoi centroids
                            f_i0 = 1.0
                        if ids[t + tau][2] == i:
                            f_itau = 1.0
                        if ids[t][2] == j:
                            f_j = 1.0

                        # Summation of terms
                        val_D_0 += h_I0 * h_I0 * f_i0 * f_j * weight
                        val_D_tau += h_Itau * h_I0 * f_itau * f_j * weight
                        if j == 0:
                            val_g_0 += ((h_I0 * f_i0 * h_B0) + (h_B0 * h_I0 * f_i0)) * weight
                            val_g_tau += ((h_Itau * f_itau * h_B0) + (h_Btau * h_I0 * f_i0)) * weight
                        count += 1

                # Divide by data size
                print(f'Dividing D by data size: {count}\n')
                D_0.append(val_D_0 / count)
                D_tau.append(val_D_tau / count)
            g_0.append(val_g_0 / count)
            g_tau.append(val_g_tau / count)

        # Reshape into appropriate matrix size
        D_0_arr = np.asarray(D_0).reshape(self.num_basis, self.num_basis)
        D_tau_arr = np.asarray(D_tau).reshape(self.num_basis, self.num_basis)
        g_0_arr = np.asarray(g_0) # .reshape(self.num_basis)
        g_tau_arr = np.asarray(g_tau)# .reshape(self.num_basis)
        print('D_0 =', D_0_arr)
        print('D_tau =', D_tau_arr)
        print('g_0 =', g_0_arr)
        print('g_tau =', g_tau_arr)

        # Symmetrize for reversible dynamics
        if symmetric == True:
            D_0_arr = 0.5 * (D_0_arr + D_0_arr.T)
            D_tau_arr = 0.5 * (D_tau_arr + D_tau_arr.T)
            print('Symmetric D_0 =', D_0_arr)
            print('Symmetric D_tau =', D_tau_arr)

        # Calculate committor coefficients, b
        D = D_0_arr - D_tau_arr
        g = g_tau_arr - g_0_arr
        b = 0.5 * np.linalg.inv(D).dot(g)
        print('b =', b)

        return b, D_0_arr, D_tau_arr, g_0_arr, g_tau_arr


    cpdef calc_corr(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=1] b, np.ndarray[DTYPE_t, ndim=2] D_0, np.ndarray[DTYPE_t, ndim=2] D_tau, np.ndarray[DTYPE_t, ndim=1] g_0, np.ndarray[DTYPE_t, ndim=1] g_tau):
        """
        Compute the committor time-correlation function from the matrix elements:
        <r(0)r(0) - r(0)r(t)> = b.T[D(0)-D(tau)]b + [g(0)-g(tau)]b + <hB(0)hB(0)> - <hB(tau)hB(0)>,
        where D is an NxN matrix, b is a column vector, and b.T and g are row vectors.
        
        Parameters
        ----------
        b : 1D array
            Committor probabilities corresponding to images on string

        D_0 : ndarray(M, M)
            Square D matrix at time 0

        D_tau : ndarray(M, M)
            Square D matrix at time tau

        g_0 : 1D array of length M
            vector g at time 0

        g_tau : 1D array of length M
            vector g at time tau

        Returns
        -------
        C : float
            committor time-correlation function

        Notes
        -----
        This returns a C that is 1/2 the value as from calculating using only the b's
        e.g., calc_corr_q2 method computes C = <[b(t+tau) - b(t)] ** 2> at time t and t+tau
        
        """
        cdef double xa, xb, rx, ry, alpha, x0_A, y0_A, x0_B, y0_B, ell_tB, ell_tauB
        cdef list x, y
        cdef int N, i, j, t
        cdef double C_00, C_tau0, bD_0j, bD_tj, Bval_0, Bval_tau, hB_0, hB_tau, u_t, u_tau, C

        if self.state_assignment == 'orthogonal_projection':
            xa, xb = self.intermediate_region()
        
        x = [row[1] for row in traj]
        if self.state_assignment != '1d':
            y = [row[2] for row in traj]
        N = len(x)
   
        # Calculate correlation function
        print('\n=================================')
        print('CALCULATING CORRELATION FUNCTION')
        print('=================================')
        sys.stdout.flush()
        C_00, C_tau0 = 0.0, 0.0
        for i in range(self.num_basis):
            bD_0j, bD_tj = 0.0, 0.0
            for j in range(self.num_basis):
                bD_0j += b[j] * D_0[i][j]
                bD_tj += b[j] * D_tau[i][j]
            C_00 += b[i] * bD_0j + b[i] * g_0[i]
            C_tau0 += b[i] * bD_tj + b[i] * g_tau[i]
    
        # Calculate <hB(0) hB(0)> - <hB(tau) hB(0)>
        Bval_0, Bval_tau = 0.0, 0.0
        for t in range(N-tau):
            hB_0, hB_tau = 0.0, 0.0
            x_t = x[t]
            x_tau = x[t + tau]
            if self.state_assignment != '1d':  # If system is 2-dimensional or higher
                y_t = y[t]
                y_tau = y[t + tau]

            if self.state_assignment == '1d':
                if x_t > self.cut_B:
                    hB_0 = 1.0
                if x_tau > self.cut_B:
                    hB_tau = 1.0

            elif self.state_assignment == 'orthogonal_projection':
                u_t = (y_t * tan(self.theta) + x_t) / (tan(self.theta)**2 + 1)
                u_tau = (y_tau * tan(self.theta) + x_tau) / (tan(self.theta)**2 + 1)
                if u_t > xb:
                    hB_0 = 1.0
                if u_tau > xb:
                    hB_tau = 1.0
            elif self.state_assignment == 'ellipse':
                ell_tB = elliptical(self.ra_B, self.rb_B, self.alpha_B, self.x0_B, self.y0_B, x_t, y_t)
                ell_tauB = elliptical(self.ra_B, self.rb_B, self.alpha_B, self.x0_B, self.y0_B, x_tau, y_tau)
                if ell_tB <= 1.0:
                    hB_0 = 1.0
                if ell_tauB <= 1.0:
                    hB_tau = 1.0
            Bval_0 += hB_0 * hB_0
            Bval_tau += hB_tau * hB_0
        Bval_0 = Bval_0 / (N - tau)
        Bval_tau = Bval_tau / (N - tau)
        Bval = (Bval_0 - Bval_tau)
        C = C_00 - C_tau0 + Bval

        print('Corr =', C)
        sys.stdout.flush()
    
        return C

    cpdef calc_corr_q2(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=1] b, np.ndarray[DTYPE_t, ndim=2] ids):
        """
        Compute committor time-correlation function using trial committor at time t and t+tau:
        <[b(tau)-b(0)] ** 2>.

        Parameters
        ----------
        b : 1D array
            Committor probabilities corresponding to images on string

        D_0 : ndarray(M, M)
            Square D matrix at time 0

        D_tau : ndarray(M, M)
            Square D matrix at time tau

        g_0 : 1D array of length M
            vector g at time 0

        g_tau : 1D array of length M
            vector g at time tau

        Returns
        -------
        C : float
            committor time-correlation function

        Notes
        -----
        This returns a C that is 2x the value as from calculating using only the b's
        e.g., calc_corr_q2 method computes C = <[b(t+tau) - b(t)] ** 2> at time t and t+tau

        """
        cdef list x, y
        cdef int N, t, id_vor_t, id_vor_tau
        cdef double x_t, y_t, x_tau, y_tau, q_t, q_tau, dq, C

        x = [row[1] for row in traj]
        # if self.state_assignment != '1d':
        #     y = [row[2] for row in traj]
        N = len(x)

        # Calculate correlation function
        print('\n=================================')
        print('CALCULATING CORRELATION FUNCTION')
        print('=================================')
        sys.stdout.flush()
        C = 0.0
        for t in range(N-tau):
            if ids[t][0] == 0.0 and ids[t][1] == 0.0:  # State A at time t
                q_t = 0.0
            elif ids[t][1] == 1.0: # State B at time t
                q_t = 1.0
            else:                                      # State I at time t
                id_vor_t = int(ids[t][2])              # Check voronoi index at time t
                q_t = b[id_vor_t]

            if ids[t+tau][0] == 0.0 and ids[t+tau][1] == 0.0:  # State A at time t + tau
                q_tau = 0.0
            elif ids[t+tau][1] == 1.0:                         # State B at time t + tau
                q_tau = 1.0
            else:                                              # State I at time t + tau
                id_vor_tau = int(ids[t+tau][2])                # Check voronoi index at time t + tau
                q_tau = b[id_vor_tau]
            C += (q_tau - q_t) ** 2
        C = C / (N - tau)
        print('Corr =', C)
        sys.stdout.flush()

        return C


    cpdef calc_corr_q2_ensemble(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=1] b, np.ndarray[DTYPE_t, ndim=2] ids):
        """
        Compute committor time-correlation function using trial committor at time t and t+tau:
        <[b(tau)-b(0)] ** 2>. 
        Use this when working with aggregate trajectories.

        Parameters
        ----------
        traj : list
            list of array of trajectories with shape (T, d) for T time steps and d dimensions

        tau : int
            lag time step

        b : 1D array
            Committor probabilities corresponding to images on string

        ids : ndarray(T, V)
            array of Voronoi assignments for each trajectory time step

        Returns
        -------
        C : float
            committor time-correlation function

        Notes
        -----
        This returns a C that is 2x the value as from calculating using only the b's
        e.g., calc_corr_q2 method computes C = <[b(t+tau) - b(t)] ** 2> at time t and t+tau

        """
        cdef list x, y
        cdef int N, t, id_vor_t, id_vor_tau
        cdef double x_t, y_t, x_tau, y_tau, q_t, q_tau, dq, weight, C
        
        print('\n=================================')
        print('CALCULATING CORRELATION FUNCTION')
        print('=================================')
        sys.stdout.flush()
        
        N = len(traj)  # Number of aggregate trajectories
        dq = 0.0
        for i in range(N):
            q_t, q_tau = 0., 0.

            # Calculate q(0) for i-th Trajectory
            if ids[i][0] == 0.0 and ids[i][1] == 0.0:  # State A 
                q_t = 0.0
            elif ids[i][0] == 0.0 and ids[i][1] == 1.0:  # State B
                q_t = 1.0
            else:                                      # State I
                id_vor_t = int(ids[i][4])              # Check Voronoi index for Traj i
                q_t = b[id_vor_t]

            # Calculate q(tau) for i-th Trajectory
            if ids[i][2] == 0.0 and ids[i][3] == 0.0:  # State A 
                q_tau = 0.0
            elif ids[i][2] == 0.0 and ids[i][3] == 1.0: # State B
                q_tau = 1.0
            else:                                      # State I
                id_vor_tau = int(ids[i][5])            # Check Voronoi index for Traj i
                q_tau = b[id_vor_tau]
            
            if self.state_assignment == '1d':
                weight = ids[i][8]
            else:
                weight = ids[i][10]
                
            dq += weight * (q_tau - q_t) ** 2
            # dq += (q_tau - q_t) ** 2
        C = dq / N
        # C = dq
        print('N =', N)
        print('traj shape:', np.shape(traj))
        print('Corr =', C)
        sys.stdout.flush()

        return C


    cpdef calc_corr_q2_ensemble_timeavg(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=1] b, list weights=None):
        """
        Compute a time-averaged committor time-correlation function using a 
        trial committor at time t and t+tau:
        <[b(tau)-b(0)] ** 2>.
        Use this method when working with aggregate trajectories.

        Parameters
        ----------
        traj : list
            list of array of trajectories with shape (T, d) for T time steps and d dimensions

        tau : int
            lag time step

        z : ndarray(M, 2)
            M image positions on string

        b : 1D array
            Committor probabilities corresponding to images on string

        weights : list, optional, default=None
            assign a weight for each trajectory in traj

        Returns
        -------
        C : float
            committor time-correlation function

        Notes
        -----
        This returns a C that is 2x the value as from calculating using only the b's
        e.g., calc_corr_q2 method computes C = <[b(t+tau) - b(t)] ** 2> at time t and t+tau
        """
        cdef list x, y
        cdef np.ndarray[DTYPE_t, ndim=2] ids
        cdef int N, t, id_vor_t, id_vor_tau, t0
        cdef double x_t, y_t, x_tau, y_tau, q_t, q_tau, dq, weight, C

        print('\n=================================')
        print('CALCULATING CORRELATION FUNCTION')
        print('=================================')
        sys.stdout.flush()

        N = len(traj)  # Number of aggregate trajectories
        dq = 0.0
        count = 0
        for t0 in range(len(traj[0])):
            print('t0:', t0)
            sys.stdout.flush()
            if t0 + tau >= len(traj[0]) - 1:
                print('out of index')
                break
            ids = self.assign_states_ensemble(traj, tau, z, t0=t0, weights=weights)

            for i in range(N):
                q_t, q_tau = 0., 0.

                # Calculate q(0) for i-th Trajectory
                if ids[i][0] == 0.0 and ids[i][1] == 0.0:  # State A
                    q_t = 0.0
                elif ids[i][0] == 0.0 and ids[i][1] == 1.0:  # State B
                    q_t = 1.0
                else:                                      # State I
                    id_vor_t = int(ids[i][4])              # Check Voronoi index for Traj i
                    q_t = b[id_vor_t]

                # Calculate q(tau) for i-th Trajectory
                if ids[i][2] == 0.0 and ids[i][3] == 0.0:  # State A
                    q_tau = 0.0
                elif ids[i][2] == 0.0 and ids[i][3] == 1.0: # State B
                    q_tau = 1.0
                else:                                      # State I
                    id_vor_tau = int(ids[i][5])            # Check Voronoi index for Traj i
                    q_tau = b[id_vor_tau]

                if self.state_assignment == '1d':
                    weight = ids[i][8]
                else:
                    weight = ids[i][10]

                dq += weight * (q_tau - q_t) ** 2
                # dq += (q_tau - q_t) ** 2
                count += 1
        C = dq / count
        print('N =', N)
        print('count =', count)
        print('traj shape:', np.shape(traj))
        print('Corr =', C)
        sys.stdout.flush()

        return C


    cpdef calc_corr_q2_ensemble_timeavg_fast(self, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=1] b, list weights=None):
        """
        Compute a time-averaged committor time-correlation function using a
        trial committor at time t and t+tau:
        <[b(tau)-b(0)] ** 2>.
        Use this method when working with aggregate trajectories.

        Parameters
        ----------
        traj : list
            list of array of trajectories with shape (T, d) for T time steps and d dimensions

        tau : int
            lag time step

        z : ndarray(M, 2)
            M image positions on string

        b : 1D array
            Committor probabilities corresponding to images on string

        weights : list, optional, default=None
            assign a weight for each trajectory in traj

        Returns
        -------
        C : float
            committor time-correlation function

        Notes
        -----
        This returns a C that is 2x the value as from calculating using only the b's
        e.g., calc_corr_q2 method computes C = <[b(t+tau) - b(t)] ** 2> at time t and t+tau
        
        """
        cdef list x, y
        cdef np.ndarray[DTYPE_t, ndim=2] ids
        cdef int N, t, id_vor_t, id_vor_tau, t0
        cdef double x_t, y_t, x_tau, y_tau, q_t, q_tau, dq, weight, C

        print('\n=================================')
        print('CALCULATING CORRELATION FUNCTION')
        print('=================================')
        sys.stdout.flush()

        N = len(traj)  # Number of aggregate trajectories
        dq = 0.0
        count = 0
        if weights == None:
            weights = [1.0] * N
            print('Using uniform weights')

        for k in range(N):  # Iterate over number of aggregate trajectories
            ids = self.assign_states(traj[k], tau, z)
            weight = weights[k]
            for t in range(len(traj[k])-tau):  # Iterate over trajectory steps
                q_t, q_tau = 0., 0.

                # Calculate q(0) for k-th Trajectory at time t
                if ids[t][0] == 0.0 and ids[t][1] == 0.0:  # State A at time t
                    q_t = 0.0
                elif ids[t][1] == 1.0: # State B at time t
                    q_t = 1.0
                else:                                      # State I at time t
                    id_vor_t = int(ids[t][2])              # Check voronoi index at time t
                    q_t = b[id_vor_t]

                if ids[t+tau][0] == 0.0 and ids[t+tau][1] == 0.0:  # State A at time t + tau
                    q_tau = 0.0
                elif ids[t+tau][1] == 1.0:                         # State B at time t + tau
                    q_tau = 1.0
                else:                                              # State I at time t + tau
                    id_vor_tau = int(ids[t+tau][2])                # Check voronoi index at time t + tau
                    q_tau = b[id_vor_tau]

                dq += weight * (q_tau - q_t) ** 2
                count += 1
        C = dq / count
        print('N =', N)
        print('count =', count)
        print('traj shape:', np.shape(traj))
        print('Corr =', C)
        sys.stdout.flush()

        return C


    cpdef finite_diff_corr(self, int image, int dim, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=2] ids, double delta=0.1):
        """
        Finite difference to compute the gradient of the committor correlation function.
        A central finite difference is used here.
    
        Parameters:
        -----------
        image : int
            index of image on string

        dim : int
            number of dimensions in collective variable space

        traj : list
            trajectory with shape (T, d) for T time steps and d dimensions

        tau : int
            lag time steps

        z : ndarray(M, 2)
            M image positions

        ids : ndarray(T, V)
            Voronoi assignment at each trajectory time step. Columns are defined according to the
            state_assignment

        delta : float, default=0.1
            move position by a value of delta

        Returns
        -------
        Cp_list : list
            correlation value of z+ + delta

        Cm_list : list
            correlation value of z+ - delta

        gradC_list : list
            gradient of C 
            
        """
        cdef np.ndarray[DTYPE_t, ndim=2] zm, zp
        cdef list Cp_list, Cm_list, gradC_list
        cdef int i, j, N
        cdef double Cp, Cm, dC
        N = len(traj)
        print('\nFINITE DIFFERENCE PARAMETERS')
        print('----------------------------')
        print('Step change in position =', delta)
        print(z)
    
        Cp_list = []
        Cm_list = []
        gradC_list = []
        for i in [image]:
            for j in [dim]:
                print('\n\n======================================')
                print('CALCULATING FINITE DIFFERENCE GRADIENT')
                print('======================================')
                print('image, dim:', i, j)
                print('Initial position:', z[i][j])
                sys.stdout.flush()

                # Calculate the finite difference gradient
                z[i][j] = z[i][j] + delta
                zp = deepcopy(z)
                b, D_0, D_tau, g_0, g_tau = self.calc_coeffs(N, tau, zp, ids) # Note: will be overwritten if saved 
                Cp = self.calc_corr(traj, tau, b, D_0, D_tau, g_0, g_tau)
                Cp_list.append(Cp)
                print('Corr (+) =', Cp)
                print()
                z[i][j] = z[i][j] - 2 * delta
                zm = deepcopy(z)
                b, D_0, D_tau, g_0, g_tau = self.calc_coeffs(N, tau, zm, ids)
                Cm = self.calc_corr(traj, tau, b, D_0, D_tau, g_0, g_tau)
                Cm_list.append(Cm)
                print('Corr (-) =', Cm)
                dC = (Cp - Cm) / (2 * delta)  # derivative
                z[i][j] = z[i][j] + delta  # move z back to original value for next iteration
                print('\nInitial position:', z[i][j])
                print('Plus/minus delta:', zp[i][j], zm[i][j])
                print('gradC:', dC)
                gradC_list.append(dC)
                # else:
                #     print('\nWARNING: No finite difference computed')
        np.savez('output/gradC.npz',
                 Cp=Cp_list,
                 Cm=Cm_list,
                 gradC=gradC_list,)
        print('\nFINITE DIFFERENCE COMPLETED')
        print('\n\n===============')
        print('    SUMMARY    ')
        print('===============')
        print('\nCp =', Cp_list)
        print('Cm =', Cm_list)
        print('gradC', gradC_list)
       
        return Cp_list, Cm_list, gradC_list

    cpdef finite_diff_committor(self, np.ndarray[DTYPE_t, ndim=2] z, np.ndarray[DTYPE_t, ndim=1] b, list start, double delta=0.1, double epsilon=1.0):
        """
        Finite difference to compute the gradient of the committor. 
        A central finite difference is used here.
        
        Parameters:
        -----------
        z : ndarray(M, 2)
            M image positions

        b : 1d array of shape (M,)
            committor corresponding to z

        start : list
            starting position

        delta : float, default=0.1
            move position by a value of delta

        epsilon : float, default=1.0
            moves z by a multiple of grad q. Specifices how much to change z.

        Returns
        -------
        dq_list : list
            gradient of q

        z[id_new] : list
            image position on string z that is closest to position given by start
        
        b[id_new] : list
            committor for specified image

        Note
        ----
        This is an alternative to the Monte Carlo-based methods below. 
        Not recommended for most cases.
            
        """
        cdef np.ndarray[DTYPE_t, ndim=1] dist, dist_p, dist_m, dist_new
        cdef np.ndarray[DTYPE_t, ndim=2] zp, zm
        cdef int id_start, i, id_p, id_m, id_new
        cdef double qp, qm, dc, znew_x, znew_y
        cdef list dq_list, znew
        
        # print('Starting position:', start)
        dist = sqrt((z[:, 0] - start[0]) ** 2 + (z[:, 1] - start[1]) ** 2) # Find closest point in z
        id_start = np.argmin(dist)
        dq_list = []
        for i in range(2): # Finite difference in 2 dimensions (x and y)
            zp = deepcopy(z)
            zp[id_start][i] += delta
            dist_p = sqrt((z[:, 0] - zp[id_start][0]) ** 2 + (z[:, 1] - zp[id_start][1]) ** 2)
            id_p = np.argmin(dist_p)
            qp = b[id_p]

            zm = deepcopy(z)
            zm[id_start][i] -= delta
            dist_m = sqrt((z[:, 0] - zm[id_start][0]) ** 2 + (z[:, 1] - zm[id_start][1]) ** 2)
            id_m = np.argmin(dist_m)
            qm = b[id_m]

            dq = (qp - qm) / (2 * delta)
            dq_list.append(dq)
        # print('dq_list =', dq_list)
        znew_x = start[0] - epsilon * dq_list[0]
        znew_y = start[1] - epsilon * dq_list[1]
        znew = [znew_x, znew_y]
        dist_new = sqrt((z[:, 0] - znew_x) ** 2 + (z[:, 1] - znew_y) ** 2)
        id_new = np.argmin(dist_new)
        # print('znew =', znew)
        # print('z[id_new] =', z[id_new])
        # print('b[id_new] =', b[id_new])
        return dq_list, z[id_new], b[id_new]
        
    cpdef montecarlo(self, list images, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, double target, double delta=0.01, int n=100, ensemble_timeavg_v1=False, ensemble_timeavg_v2=False, ensemble_avg=False, stride=None, list weights=None, symmetric=True, manual=False, hx_man=None, hy_man=None, reparm=True, mindist_constraint=False, double mindist=0.25, accelerated=False, corr_method='matrix_elements', save=True):
        """
        Monte Carlo (MC) accept-reject to variationally minimize the
        committor time-correlation function.

        Parameters:
        -----------
        images : list
            indices of the images on the string z

        traj : list
            trajectory or list of array of trajectories with shape (T, d)

        tau : int
            lag time step

        z : ndarray(M, 2)
            M image positions on the string

        target : float
            upper value of the committor time-correlation function. We seek to minimize this or 
            the Monte Carlo move is rejected

        delta : float, default=0.01
            move the current image in z by a value of delta

        n : int, default=100
            how many MC moves to perform to find an accepted value

        ensemble_timeavg_v1 : bool, default=False
            computes results using time averages for an ensemble of trajectories,
            best for a handful of long independent trajectories

        ensemble_timeavg_v2 : bool, default=False
            computes results using time averages for an ensemble of trajectories,
            best for a high number of very short independent trajectories
            
        ensemble_avg : bool, default=False
            computes results for an ensemble of trajectories

        stride : int, optional, default=None
            use every stride-th frame of the trajectory 

        weights : list, optional, default=None
            assign weight to each trajectory in an ensemble

        symmetric : bool, default=True
            ensures symmetrized matrices when solving for the committor, only use for
            symmetric systems, e.g., a symmetric double well potential

        manual : bool, optional, default=False
            manual=True to allow specifying hx_man and hy_man. Use to test random sampling.

        hx_man, hy_man : bool, optional, default=None
            move position of image from z by a value of hx_man in the x-direction and 
            by a value of hy_man in the y-direction

        reparm : bool, default=True
            reparametrization such that the images are equidistant from each
            other on the string. This ensures that the images do not pool within
            the well basins over the course of the string optimization.

        mindist_constraint : bool, default=False
            enable this to check the distance between neighboring beads at each MC step

        mindist : float, default=0.25
            constraint for minimimum distance between images, must also set mindist_constraint=True
            to use this feature

        accelerated : bool, default=False
            specifies whether to use the accerlated computation to solve for the trial committor b
            and matrices/vectors D and g. Generally NOT recommended to enable this - can give 
            unstable results depending on the system

        save : bool, default=True
            specifies whether to save the computed results in files in an output/ directory.
            NOTE: must make an /output directory in the current path.

        Returns
        -------
        if save=True, then the following files are written to a directory named output/:
            mc.txt gives [hx, hy, Cnew, z[i][0], z[i][1]].
            z_new.txt gives the new image position if the MC move was accepted
            C_new.txt gives the lower correlation function value if the MC move was accepted
            q_new.txt gives the new set of trial committors if the MC move was accepted
            coeffs_new.npz saves the b as well as the D matrices and g vectors

        """
        cdef list MC_list
        cdef int it, i, N, k
        cdef double hx, hy, Cnew, C_k, d1, d2
        cdef double traj_k_list
        cdef np.ndarray[DTYPE_t, ndim=2] D_0, D_tau, ids, traj_k
        cdef np.ndarray[DTYPE_t, ndim=2] D_0_ens, D_tau_ens
        cdef np.ndarray[DTYPE_t, ndim=1] b, g_0, g_tau, zinit, time
        cdef np.ndarray[DTYPE_t, ndim=1] b_ens, g_0_ens, g_tau_ens

        N = len(traj)
        print('\n=================================')
        print('MONTE CARLO SAMPLING')
        print('=================================')
        print('\nMONTE CARLO PARAMETERS')
        print('-------------------------')
        print('Target for accept/reject:', target)
        print('Step change in position:', delta)
        print('Number of MC steps per image:', n)
        print('lag time:', tau)
        print('Input positions:\n', z)
        print('\nStarting Monte Carlo calculations')
        sys.stdout.flush()

        # for i in range(self.num_basis): # Move each bead on the string
        for i in images:  # Move the i-th bead(s) on the string
            z_init = deepcopy(z[i])
            print('\nIMAGE', i)
            print('--------------------')
            print('Initial position:', z_init)

            MC_list = []
            for it in range(n): # Run n numbers of MC steps
                print('\nCALCULATING MONTE CARLO STEP', it)
                print('-------------------------------')
                sys.stdout.flush()
                if manual == False:
                    hx = random.uniform(-delta, delta) # get random value to move z
                    hy = random.uniform(-delta, delta)
                else:
                    hx = hx_man  # MC with manually selected move
                    hy = hy_man
                z[i][0] = z[i][0] + hx  # Update z
                z[i][1] = z[i][1] + hy
                print('Moving position by {0} , {1}'.format(hx, hy))
                print('New position for image {0} is: {1} , {2}'.format(i, z[i][0], z[i][1]))

                # Reparametrization of string: images are equidistance from one another
                if reparm == True:
                    z = reparm_string(z)

                # Check distance between neighboring beads
                if mindist_constraint == True:
                    if i != 0 or i != len(z)-1:  # If not first or last image on the string
                        d1 = math.sqrt((z[i][0] - z[i-1][0]) ** 2 + (z[i][1] - z[i-1][1]) ** 2)
                        d2 = math.sqrt((z[i][0] - z[i+1][0]) ** 2 + (z[i][1] - z[i+1][1]) ** 2)
                    elif i == 0:  # If first image on string
                        d1 = math.sqrt((z[i][0] - z[i+1][0]) ** 2 + (z[i][1] - z[i+1][1]) ** 2)
                        d2 = mindist
                    elif i == len(z)-1:  # If last image on string
                        d1 = math.sqrt((z[i][0] - z[i-1][0]) ** 2 + (z[i][1] - z[i-1][1]) ** 2)
                        d2 = mindist
                    print('\nEnforcing minimum distance constraint')
                    print('Minimum distance allowed between images:', mindist)
                    print('Distance to neighbhoring images:', d1, d2)
                    if d1 < mindist or d2 < mindist:  # If smaller than the minimum distance allowed
                        z[i][0] = z[i][0] - hx  # Move back to original position
                        z[i][1] = z[i][1] - hy
                        print('Rejected: Moved position by {0} , {1}. Too close to neighboring images.'.format(hx, hy))
                        continue

                # Solve linear coeff equation, calculate correlation function
                if accelerated == True:
                    if (i != 0) and (it != 1): # If not the 1st image of the 1st iteration
                        b, D_0, D_tau, g_0, g_tau = self.calc_coeffs_accelerated(i, traj, tau, z)
                else:
                    # Average of time averages (best for a few aggregate long trajectories)
                    if ensemble_timeavg_v1 == True:
                        print('\n================================')
                        print('ENSEMBLE TIME AVERAGE - method 1')
                        print('================================')
                        print('Number of trajectories:', N)
                        b = np.zeros(self.num_basis)
                        D_0 = np.zeros((self.num_basis, self.num_basis))
                        D_tau = np.zeros((self.num_basis, self.num_basis))
                        g_0 = np.zeros(self.num_basis)
                        g_tau = np.zeros(self.num_basis)
                        C_k = 0.0

                        for k in range(N):
                            print('\nTrajectory', k)
                            # time = np.arange(len(traj[k]))
                            # traj_k = np.c_[time, traj[k]]
                            # traj_k_list = traj_k.tolist()
                            ids = self.assign_states(traj[k], tau, z)
                            b_ens, D_0_ens, D_tau_ens, g_0_ens, g_tau_ens = self.calc_coeffs(len(traj[k]), tau, z, ids, symmetric=symmetric)
                            b += b_ens
                            D_0 += D_0_ens
                            D_tau += D_tau_ens
                            g_0 += g_0_ens
                            g_tau += g_tau_ens
                            # print('b_ens =', b_ens)
                            # print('b     =', b)

                            # Calculate correlation function
                            if corr_method == 'matrix_elements':
                                C_k += self.calc_corr(traj[k], tau, b_ens, D_0_ens, D_tau_ens, g_0_ens, g_tau_ens)
                            elif corr_method == 'q2':
                                C_k += self.calc_corr_q2(traj[k], tau, b_ens, ids)
                            # print('C_k =', C_k)

                        b = b / float(N)
                        D_0 = D_0 / float(N)
                        D_tau = D_tau / float(N)
                        g_0 = g_0 / float(N)
                        g_tau = g_tau / float(N)
                        Cnew = C_k / float(N)
                        print('\nb from ensemble time average =', b)
                        print('Corr from ensemble time average =', Cnew, '\n')

                    # Ensemble time average (best for a lot of SHORT aggregate trajectories)
                    elif ensemble_timeavg_v2 == True:
                        print('\n================================')
                        print('ENSEMBLE TIME AVERAGE - method 2')
                        print('================================')
                        print('Number of trajectories:', N)
                        if weights == None:
                            weights = [1.] * N
                        b, D_0, D_tau, g_0, g_tau = self.calc_coeffs_ensemble_timeavg_fast(traj, tau, z, weights=weights, symmetric=symmetric)
                        Cnew = self.calc_corr_q2_ensemble_timeavg_fast(traj, tau, z, b, weights=weights)
                        # print('\nb from ensemble time average =', b)
                        # print('Corr from ensemble time average =', Cnew, '\n')

                    # Ensemble averaging (best with lots of aggregate trajectories)
                    elif ensemble_avg == True:
                        print('\n==============================')
                        print('ENSEMBLE AVERAGE')
                        print('==============================')
                        print('Number of trajectories:', N)
                        if weights == None:
                            weights = [1.] * N
                        ids = self.assign_states_ensemble(traj, tau, z, weights=weights)
                        b, D_0, D_tau, g_0, g_tau = self.calc_coeffs_ensemble(N, tau, z, ids, symmetric=symmetric)
                        # WARNING: calc_corr with matrixelms method is not currently supported
                        Cnew = self.calc_corr_q2_ensemble(traj, tau, b, ids)
                        # print('\nq from ensemble average =', b)
                        # print('Corr from ensemble average =', Cnew, '\n')

                    # Simple time average for a single trajectory
                    else:
                        ids = self.assign_states(traj, tau, z)  # Reassign states since z is changed
                        b, D_0, D_tau, g_0, g_tau = self.calc_coeffs(N, tau, z, ids, symmetric=symmetric)
                        # Calculate correlation function
                        if corr_method == 'matrix_elements':
                            Cnew = self.calc_corr(traj, tau, b, D_0, D_tau, g_0, g_tau)
                        elif corr_method == 'q2':
                            Cnew = self.calc_corr_q2(traj, tau, b, ids)
                
                # Accept or reject the move
                if Cnew < target:
                    MC_list.extend([hx, hy, Cnew, z[i][0], z[i][1]])
                    if save == True:
                        np.savetxt('output/mc.txt', MC_list)
                        np.savetxt('output/z_new.txt', z)
                        np.savetxt('output/C_new.txt', [Cnew])
                        np.savetxt('output/q_new.txt', b)
                        np.savez('output/coeffs_new.npz',
                                 b=b,
                                 D_0=D_0,
                                 D_tau=D_tau,
                                 g_0=g_0,
                                 g_tau=g_tau)
                    print('Accepted: Moved position by {0} , {1}. Cnew {2} is less than Cmax {3}'.format(hx, hy, Cnew, target))
                    print('Old position:', z_init)
                    print('New position:', z[i])
                    return z
                else:
                    z[i][0] = z_init[0]       # Move back to original position
                    z[i][1] = z_init[1]
                    # z[i][0] = z[i][0] - hx  # Move back to original position
                    # z[i][1] = z[i][1] - hy
                    print('Rejected: Moved position by {0} , {1}. Cnew {2} is greater than Cmax {3}'.format(hx, hy, Cnew, target))
                    continue

            # In the case when all values were rejected
            print('\nWARNING: No values were accepted. Try increasing the number of MC steps.')
            print('Old position:', z_init)

        return

    cpdef montecarlo_symmetric(self, list images, list traj, int tau, np.ndarray[DTYPE_t, ndim=2] z, double target, double delta=0.01, int n=100, manual=False, hx_man=None, hy_man=None, reparm=True, mindist_constraint=False, mindist=0.25, accelerated=False, corr_method='q2', save=True):
        """
        Speed up for systems that are symmetric about an axis of rotation. 
        Monte Carlo (MC) accept-reject to variationally minimize the
        committor time-correlation function.

        Parameters
        ----------
        images : list
            indices of the images on the string z

        traj : list
            trajectory or list of array of trajectories with shape (T, d)

        tau : int
            lag time step

        z : ndarray(M, 2)
            M image positions on the string

        target : float
            upper value of the committor time-correlation function. We seek to minimize this or
            the Monte Carlo move is rejected

        delta : float, default=0.01
            move the current image in z by a value of delta

        n : int, default=100
            how many MC moves to perform to find an accepted value

        manual : bool, optional, default=False
            manual=True to allow specifying hx_man and hy_man. Use to test random sampling.

        hx_man, hy_man : bool, optional, default=None
            move position of image from z by a value of hx_man in the x-direction and
            by a value of hy_man in the y-direction

        reparm : bool, default=True
            reparametrization such that the images are equidistant from each
            other on the string. This ensures that the images do not pool within
            the well basins over the course of the string optimization.

        mindist_constraint : bool, default=False
            enable this to check the distance between neighboring beads at each MC step

        mindist : float, default=0.25
            constraint for minimimum distance between images, must also set mindist_constraint=True
            to use this feature

        accelerated : bool, default=False
            specifies whether to use the accerlated computation to solve for the trial committor b
            and matrices/vectors D and g. Generally NOT recommended to enable this - can give
            unstable results depending on the system

        corr_method : str, 'matrix_elements' or 'q2', default='q2'
            specifies how to compute the committor-correlation function, C.
            'matrix_elements' computes C using the matrix elements from the linear equation.
            'q2' computes C by <[q(t+tau) - q(t)] ** 2> for a trial committor q at time t and t+tau.

        save : bool, default=True
            specifies whether to save the computed results in files in an output/ directory.
            NOTE: must make an /output directory in the current path.

        Returns
        -------
        if save=True, then the following files are written to a directory named output/:
            mc.txt gives [hx, hy, Cnew, z[i][0], z[i][1]].
            z_new.txt gives the new image position if the MC move was accepted
            C_new.txt gives the lower correlation function value if the MC move was accepted
            q_new.txt gives the new set of trial committors if the MC move was accepted
            coeffs_new.npz saves the b as well as the D matrices and g vectors

        """
        cdef list MC_list
        cdef int it, i, ii
        cdef double hx, hy, Cnew, d1, d2
        cdef np.ndarray[DTYPE_t, ndim=2] D_0, D_tau, ids
        cdef np.ndarray[DTYPE_t, ndim=1] b, g_0, g_tau, zinit

        N = len(traj)
        print('\n=================================')
        print('MONTE CARLO SAMPLING - SYMMETRIC')
        print('=================================')
        print('\nMONTE CARLO PARAMETERS')
        print('-------------------------')
        print('Target for accept/reject:', target)
        print('Step change in position:', delta)
        print('Number of MC steps per image:', n)
        print('lag time:', tau)
        print('Input positions:\n', z)
        print('\nStarting Monte Carlo calculations')
        sys.stdout.flush()

        for i in images:  # Move the i-th bead(s) on the string
            z_init = deepcopy(z[i])
            ii = int(self.num_basis - 1 - i)  # Index of bead on the opposite end of string
            z_init_opp = deepcopy(z[ii])
            print('\nIMAGE {0} and {1}'.format(i, ii))
            print('--------------------')
            print('Initial position:', z_init, z_init_opp)

            MC_list = []
            for it in range(n): # Run n numbers of MC steps
                print('\nCALCULATING MONTE CARLO STEP', it)
                print('-------------------------------')
                sys.stdout.flush()
                if manual == False:
                    hx = random.uniform(-delta, delta) # get random value to move z
                    hy = random.uniform(-delta, delta)
                else:
                    hx = hx_man  # MC with manually selected move
                    hy = hy_man
                    # hx = -0.026412443087807838  # testing
                    # hy = -0.004352538461786369  # testing
                z[i][0] = z[i][0] + hx  # Update z
                z[i][1] = z[i][1] + hy
                z[ii][0] = z[ii][0] - hx
                z[ii][1] = z[ii][1] - hy
                print('Image {0}: Moving position by {1} , {2}'.format(i, hx, hy))
                print('New position for image {0}: {1} , {2}'.format(i, z[i][0], z[i][1]))
                print('Image {0}: Moving position by {1} , {2}'.format(ii, -hx, -hy))
                print('New position for image {0}: {1} , {2}'.format(ii, z[ii][0], z[ii][1]))
                print('New string =\n', z)

                # Reparametrization of string: images are equidistance from one another
                if reparm == True:
                    z = reparm_string(z)

                # Check distance between neighboring beads
                if mindist_constraint == True:
                    if i != 0 or i != len(z)-1:  # If not first or last image on the string
                        d1 = math.sqrt((z[i][0] - z[i-1][0]) ** 2 + (z[i][1] - z[i-1][1]) ** 2)
                        d2 = math.sqrt((z[i][0] - z[i+1][0]) ** 2 + (z[i][1] - z[i+1][1]) ** 2)
                    elif i == 0:  # If first image on string
                        d1 = math.sqrt((z[i][0] - z[i+1][0]) ** 2 + (z[i][1] - z[i+1][1]) ** 2)
                        d2 = mindist
                    elif i == len(z)-1:  # If last image on string
                        d1 = math.sqrt((z[i][0] - z[i-1][0]) ** 2 + (z[i][1] - z[i-1][1]) ** 2)
                        d2 = mindist
                    print('\nEnforcing minimum distance constraint')
                    print('Minimum distance allowed between images:', mindist)
                    print('Distance to neighbhoring images:', d1, d2)
                    if d1 < mindist or d2 < mindist:  # If smaller than the minimum distance allowed
                        z[i][0] = z[i][0] - hx  # Move back to original position
                        z[i][1] = z[i][1] - hy
                        z[ii][0] = z[ii][0] + hx
                        z[ii][1] = z[ii][1] + hy
                        print('Rejected: Moved position by {0} , {1}. Too close to neighboring images.'.format(hx, hy))
                        continue

                # Solve linear equation and correlation function
                if accelerated == True:
                    if (i != 0) and (it != 1): # If not the 1st image of the 1st iteration
                        b, D_0, D_tau, g_0, g_tau = self.calc_coeffs_accelerated(i, traj, tau, z)
                else:
                    # print('State assignment:', self.state_assignment)
                    ids = self.assign_states(traj, tau, z)  # Reassign states since z is changed
                    b, D_0, D_tau, g_0, g_tau = self.calc_coeffs(N, tau, z, ids, symmetric=True)

                # Calculate correlation function
                if corr_method == 'matrix_elements':
                    Cnew = self.calc_corr(traj, tau, b, D_0, D_tau, g_0, g_tau)
                elif corr_method == 'q2':
                    Cnew = self.calc_corr_q2(traj, tau, b, ids)
                # Accept or reject the move
                if Cnew < target:
                    MC_list.extend([hx, hy, Cnew, z[i][0], z[i][1], z[ii][0], z[ii][1]])
                    if save == True:
                        np.savetxt('output/mc.txt', MC_list)
                        np.savetxt('output/z_new.txt', z)
                        np.savetxt('output/C_new.txt', [Cnew])
                        np.savez('output/coeffs_new.npz',
                                 b=b,
                                 D_0=D_0,
                                 D_tau=D_tau,
                                 g_0=g_0,
                                 g_tau=g_tau)
                    print('Accepted: Moved position by {0} , {1}. Cnew {2} is less than Cmax {3}'.format(hx, hy, Cnew, target))
                    print('Old position:', z_init, z_init_opp)
                    print('New position:', z[i], z[ii])
                    return z
                else:
                    z[i][0] = z[i][0] - hx  # Move back to original position
                    z[i][1] = z[i][1] - hy
                    z[ii][0] = z[ii][0] + hx
                    z[ii][1] = z[ii][1] + hy
                    print('Rejected: Moved position by {0} , {1}. Cnew {2} is greater than Cmax {3}'.format(hx, hy, Cnew, target))
                    continue

            # In the case when all values were rejected
            print('\nWARNING: No values were accepted. Try increasing the number of MC steps.')
            print('Old position:', z_init, z_init_opp)

        return

cpdef reparm_string(np.ndarray[DTYPE_t, ndim=2] z):
    """
    Reparametrization procedure for the string method to ensure equal distances
    between the images. This prevents the images from clumping at the well basins
    over the course of the string optimization.

    Parameter
    ---------
    z : ndarray(M, 2)
        M image positions of a string

    Returns
    -------
    z_new_arr : ndarray(M, 2)
        M image positions of the reparametrized string

    """
    cdef np.ndarray[DTYPE_t, ndim=1] x, y, dist_im, pos
    cdef np.ndarray[long, ndim=1] where
    cdef np.ndarray[DTYPE_t, ndim=2] grad, M, arr
    cdef int i, j
    cdef double dist_tot, equidist, d, diff
    cdef list z_new

    print('\n=========================')
    print('REPARAMETRIZE THE STRING')
    print('=========================')
    x = z[:, 0]
    y = z[:, 1]
    dist_im = sqrt((x[1:] - x[0:-1]) ** 2 + (y[1:] - y[0:-1]) ** 2)  # distance between images
    dist_tot = sum(dist_im)  # Total distance of string
    equidist = dist_tot / len(dist_im)  # Equidistance length between images
    print('Distance between images:', dist_im)
    print('Total length of string:', dist_tot)
    print('Equidistance:', equidist)
    print()

    grad = z[1:] - z[0:-1]  # unit gradient in x, y direction
    M = sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2).reshape(-1, 1)  # Magnitude of gradient
    grad = grad / M  # Unit vector pointing from one image to the next on string

    z_new = []
    pos = deepcopy(z[0])
    z_new.append([pos[0], pos[1]])
    for i in range(len(z)):
        d = 0.0  # Length of string between current image and previous image
        # print(i)
        # print('equidist:', equidist)
        # print('initial pos:', pos)
        for j in range(len(dist_im)):
            # print(i, j)
            if d < equidist:
                diff = equidist - d
                # print('d, diff:', d, diff)
                # print('dist_im:', dist_im)
                if dist_im[j] <= diff:
                    d += dist_im[j]
                    pos[0] += grad[j][0] * dist_im[j]
                    pos[1] += grad[j][1] * dist_im[j]
                    dist_im[j] = 0.0
                elif dist_im[j] >= diff:
                    d += diff
                    pos[0] += grad[j][0] * diff
                    pos[1] += grad[j][1] * diff
                    dist_im[j] -= diff
                # print('new dist_im:', dist_im)
                # print('new d:', d)
                # print('pos:', pos)
                # print(d == equidist)
    
                if round(d, 5) == round(equidist, 5):
                    # print(d == equidist)
                    z_new.append([pos[0], pos[1]])
                    break
        where = np.where(dist_im == 0)[0]
        dist_im = np.delete(dist_im, where)
        grad = np.delete(grad, where, axis=0)
        # print(z_new)
        # print()
    z_new_arr = np.asarray(z_new)
    print('\nReparametrized string =\n', z_new_arr)

    # Check reparametrized string
    x = z_new_arr[:, 0]
    y = z_new_arr[:, 1]
    dist_im = sqrt((x[1:] - x[0:-1]) ** 2 + (y[1:] - y[0:-1]) ** 2)  # distance between images
    dist_tot = sum(dist_im)  # Total distance of string
    equidist = dist_tot / len(dist_im)  # Equidistance length between images
    print('\nLength input, output:', len(z), len(z_new_arr))
    print('Length input string = output string:', len(z) == len(z_new_arr))
    print('Distance between images:', dist_im)
    print('Total length of string:', dist_tot)
    print('Equidistance:', equidist)
    if len(z) != len(z_new_arr):
        print('ERROR, length of input string != output string')
        return

    return z_new_arr

cpdef elliptical_bounds_params(double ra=1.0, double rb=0.4, double alpha=58.0, double x0_A=-2.2, double y0_A=-2.2, double x0_B=2.2, double y0_B=2.2):
    """
    **DEPRECATED**
    Parameters for defining the elliptical boundaries at the well minima where the
    committor q = 0 or q = 1, respectively.

    Parameters
    ----------
    ra : float, default=1.0
        major radius along x-axis, ra > rb
    
    rb : float, default=0.4 
        minor radius along y-axis

    alpha : float, default=58
        rotation of ellipse in degrees

    x0_A : float, default=-2.2
        x coordinate for center of ellipse for state A

    y0_A : float, default=-2.2
        y coordinate for center of ellipse for state A

    x0_B : float, default=2.2
        x coordinate for center of ellipse for state B

    y0_B : float, default=2.2
        y coordinate for center of ellipse for state B

    Returns
    -------
    ra, rb, alpha*pi/180.0, x0_A, y0_A, x0_B, y0_B
    """
    print('This method elliptical_bounds_params is deprecated')
    return ra, rb, alpha*pi/180.0, x0_A, y0_A, x0_B, y0_B

def elliptical(double a, double b, double alpha, double x0, double y0, xp, yp):
    """
    Given a position (xp, yp), return the value with respect to the ellipse.
    Value greater or equal to 1: position is on or inside ellipse.
    Value less than 1: position is outside of the ellipse.

    Parameters
    ----------
    a : float
        major radius along x-axis

    b : float
        minor radius along y-axis

    alpha : float
        rotation of ellipse in degrees

    x0, y0 : floats
        position of the center of ellipse

    xp, yp : floats
        given (x, y) coordinate inputs

    Returns
    -------
    ell : float
        value used to test position of (xp, yp) with respect to the ellipse

    """
    ell_x = ((math.cos(alpha) * (xp - x0) + math.sin(alpha) * (yp - y0)) ** 2) / (a ** 2)
    ell_y = ((math.sin(alpha) * (xp - x0) - math.cos(alpha) * (yp - y0)) ** 2) / (b ** 2)
    ell = ell_x + ell_y
    return ell

def smoothing(C_list, q_list, lags, first=10, last=10, cut=0):
    """
    Smoothing algorithm to reduce the noise of a dataset over a range of lag times

    Parameters
    ----------
    C_list : list
        list of committor time-correlation functions over lag times

    q_list : list
        list of committor probabilities over lag times

    first : int, default=10
        smoothing using 10 previous elements

    last : int, default=10
        smoothing using 10 later elements

    cut : int, default=0
        index at which to begin the smoothing process

    Returns
    -------
    C_avg : 1d array of shape (t,)
        for t lag times

    q_avg : ndarray(t, M)
        for t lag times and M images on string

    """
    C_avg = []
    q_avg = []
    for i in range(len(lags)):
        if i < cut:
            C = C_list[i]
            C_avg.append(C)
            q = q_list[i]
            q_avg.append(q)
        elif i >= cut:
            C = np.sum(C_list[i-first:i+last]) / len(C_list[i-first:i+last])
            C_avg.append(C)
            q = np.sum(q_list[i-first:i+last], axis=0) / len(q_list[i-first:i+last])
            q_avg.append(q)
    C_avg = np.asarray(C_avg)
    q_avg = np.asarray(q_avg)
    
    return C_avg, q_avg

