import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    Finds the critical buckling load P using the Secant Formula.
    
    Parameters:
        L: Length of column [mm]
        E: Modulus of elasticity [MPa]
        A: Cross-sectional area [mm²]
        r: Radius of gyration [mm]
        c: Distance to extreme fiber [mm]
        e: Eccentricity [mm]
        sigma_allow: Allowable stress [MPa]
    
    Returns:
        Critical load P in Newton (float)
    """
    
    def sigma_max(P):
        """Calculate maximum stress for given load P using Secant formula"""
        if P <= 0:
            return 0.0
        
        term1 = P / A
        eccentricity_factor = (e * c) / (r ** 2)
        argument = (L / (2 * r)) * np.sqrt(P / (E * A))
        sec_term = 1 / np.cos(argument)
        
        return term1 * (1 + eccentricity_factor * sec_term)
    
    def objective(P):
        """Objective function: sigma_max(P) - sigma_allow"""
        return sigma_max(P) - sigma_allow
    
    # Upper bound guess (slightly below Euler load)
    P_upper = (np.pi**2 * E * A) / ((L / r)**2) * 0.9
    
    # Solve using bisection method
    P_critical = bisect(objective, 1e-6, P_upper, xtol=1e-6)
    
    return P_critical
