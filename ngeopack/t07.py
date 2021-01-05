import numpy as np
from numba import njit
from t07scoef import tse, tso, tss

# =========
# = WORKS =
# =========

@njit
def extern(iopgen, a, ps, pdyn, x, y, z):
    '''
    July 2017, G.K.Stephens, This routine was updated to be a double precision subroutine.
    To indicate the update, the subroutine was renamed from EXTMODEL to TS07D_JULY_2017.
    Additionally, this July 2017 update incorporates an upgraded version of the Bessel function evaluator,
    provided by Jay Albert, which significantly speeds up the model. We thank Jay Albert for these
    contributions.

    This subroutine computes and returns the compoents of the external magnetic field vector in the
    GSM coordinate system due to extraterrestrial currents, as described in the following references.
    To compute the total magnetic field, the magnetic field vector from the Earth's internal sources
    must be added. To compute this see Dr. Tsyganenko's most recent implementation of the IGRF
    model in the GEOPACK package (Geopack-2008_dp.for).

    References: (1) Tsyganenko, N. A., and M. I. Sitnov (2007), Magnetospheric configurations from a
                    high-resolution data-based magneticfield model, J. Geophys. Res., 112,A06225,
                    doi:10.1029/2007JA012260.

                (2) Sitnov, M. I., N. A. Tsyganenko, A. Y. Ukhorskiy, B. J. Anderson, H. Korth,
                    A. T. Y. Lui, and P. C. Brandt (2010),Empirical modeling of a CIR‐driven magnetic
                    storm, J. Geophys. Res., 115, A07231, doi:10.1029/2009JA015169.

    Inputs:
       IOPT - An option flag that allows to break out the magnetic field vector contributions from the
          individual modules.
          IOPT=0 - The total external magnetic field (most common)
          IOPT=1 - The field due to only the shielding of the dipole field on the magnetopause
          IOPT=2 - The field due to only the equatorial currents and its shielding field
          IOPT=3 - The field due to only the Birkeland currents and its shielding field
       PS - The Geodipole tilt angle in radians.
       PARMOD - A 10-element array, in this model this input is not used and will have no impact on the
          model evaluation. It is kept here because the TRACE_08 routine in the Geopack package requires
          a consistent signature with other empirical magnetic field models that do use this parameter.
       X,Y,Z - The Cartesian Geocentric position where the model will be evaluated in the GSM coordinate
          system in units of Earth Radii (1 RE=6371.2 KM).

    Common Block Inputs:
       /INPUT/ PDYN - The solar wind dynamic pressure in nanoPascals (nPa)
       /PARAM/ A - An 101-element array containing the variable (time-dependent) coefficients and
          parameters used in evaluating the model
       /TSS/ TSS - An 80x5-element array containing the static (time-independent) coefficients that are
          used to shield the symmetric equatorial expansions
       /TSO/ TSO - An 80x5x4-element array containing the static (time-independent) coefficients that are
          used to shield the ODD axis-symmetric equatorial expansions
       /TSE/ TSE - An 80x5x4-element array containing the static (time-independent) coefficients that are
          used to shield the EVEN axis-symmetric equatorial expansions

    Outputs:
       BX,BY,BZ - the evaluated magnetic field vector in the GSM coordinate system in units of nanoTesla (nT)
    '''

    a0_a = 34.586
    a0_s0 = 1.196
    a0_x0 = 3.4397

    # THE COMMON BLOCK FORWARDS TAIL SHEET THICKNESS
    # SCALING FACTORS FOR BIRKELAN

    # SHUE ET A
    xappa = np.power(pdyn / 2.0, 0.155)

    # 0.155 is the value obtained in TS05
    xappa3 = xappa**3
    d0 = a[95]
    rh0 = a[96]
    g = a[97]
    xkappa1 = a[98]
    xkappa2 = a[99]
    tw = a[100]

    xx = x * xappa
    yy = y * xappa
    zz = z * xappa

    sps = np.sin(ps)

    x0 = a0_x0 / xappa
    am = a0_a / xappa
    s0 = a0_s0

    bxcf = 0.0
    bycf = 0.0
    bzcf = 0.0

    if (iopgen <= 1):
        cfx, cfy, cfz = shlcar3x3(xx, yy, zz, ps)

        # DIPOLE SHIELD
        bxcf = cfx * xappa3
        bycf = cfy * xappa3
        bzcf = cfz * xappa3

    bxts = np.zeros(5)
    byts = np.zeros(5)
    bzts = np.zeros(5)
    bxto = np.zeros((5, 4))
    byto = np.zeros((5, 4))
    bzto = np.zeros((5, 4))
    bxte = np.zeros((5, 4))
    byte = np.zeros((5, 4))
    bzte = np.zeros((5, 4))

    # Index is offset from C / Fortran version
    if (iopgen == 0 or iopgen == 2):
        bxts, byts, bzts, bxto, byto, bzto, bxte, byte, bzte = deformed(ps, xx, yy, zz, g, tw, d0, rh0)

    # WORKS UP TO HERE
    if (iopgen == 0 or iopgen == 3):
        bxr11, byr11, bzr11, bxr12, byr12, bzr12, bxr21a, byr21a, bzr21a, bxr22a, byr22a, bzr22a = birk_tot(ps, xx, yy, zz, xkappa1, xkappa2)
        bxr11s, byr11s, bzr11s, bxr12s, byr12s, bzr12s, bxr21s, byr21s, bzr21s, bxr22s, byr22s, bzr22s = birtotsy(ps, xx, yy, zz, xkappa1, xkappa2)
    else:
        bxr11 = 0.0
        byr11 = 0.0
        bzr11 = 0.0
        bxr12 = 0.0
        byr12 = 0.0
        bzr12 = 0.0
        bxr21a = 0.0
        byr21a = 0.0
        bzr21a = 0.0
        bxr21s = 0.0
        byr21s = 0.0
        bzr21s = 0.0

    a_r11 = a[91]
    a_r12 = a[92]
    a_r21a = a[93]
    a_r21s = a[94]

    tx = 0.0
    ty = 0.0
    tz = 0.0

    pdyn_0 = 2.0
    p_factor = np.sqrt(pdyn / pdyn_0) - 1.0
    ind = 1

    for k in np.arange(1, 5 + 1):
        ind += 1
        tx += (a[ind - 1] + a[ind + 45 - 1] * p_factor) * bxts[k - 1]
        ty += (a[ind - 1] + a[ind + 45 - 1] * p_factor) * byts[k - 1]
        tz += (a[ind - 1] + a[ind + 45 - 1] * p_factor) * bzts[k - 1]

    for k in np.arange(1, 5 + 1):
        for l in np.arange(1, 4 + 1):
            ind += 1
            tx += (a[ind - 1] + a[ind + 45 - 1] * p_factor) * bxto[k - 1, l - 1]
            ty += (a[ind - 1] + a[ind + 45 - 1] * p_factor) * byto[k - 1, l - 1]
            tz += (a[ind - 1] + a[ind + 45 - 1] * p_factor) * bzto[k - 1, l - 1]
            tx += (a[ind + 20 - 1] + a[ind + 65 - 1] * p_factor) * bxte[k - 1, l - 1]
            ty += (a[ind + 20 - 1] + a[ind + 65 - 1] * p_factor) * byte[k - 1, l - 1]
            tz += (a[ind + 20 - 1] + a[ind + 65 - 1] * p_factor) * bzte[k - 1, l - 1]

    bbx = a[1 - 1] * bxcf + tx + a_r11 * bxr11 + a_r12 * bxr12 + a_r21a * bxr21a + a_r21s * bxr21s
    bby = a[1 - 1] * bycf + ty + a_r11 * byr11 + a_r12 * byr12 + a_r21a * byr21a + a_r21s * byr21s
    bbz = a[1 - 1] * bzcf + tz + a_r11 * bzr11 + a_r12 * bzr12 + a_r21a * bzr21a + a_r21s * bzr21s

    bx = bbx
    by = bby
    bz = bbz

    return bx, by, bz


# =========
# = WORKS =
# =========

@njit
def shlcar3x3(x, y, z, ps):
    '''
    THIS S/R RETURNS THE SHIELDING FIELD FOR THE EARTH'S DIPOLE,
    REPRESENTED BY  2x3x3=18 "CARTESIAN" HARMONICS, tilted with respect
    to the z=0 plane (see NB#4, p.74-74)

    The 36 coefficients enter in pairs in the amplitudes of the "cartesian" harmonics (A(1)-A(36).
    The 14 nonlinear parameters (A(37)-A(50) are the scales Pi, Ri, Qi,and Si entering the arguments of exponents, sines, and cosines in each of the 
    18 "Cartesian" harmonics  PLUS TWO TILT ANGLES FOR THE CARTESIAN HARMONICS (ONE FOR THE PSI=0 MODE AND ANOTHER FOR THE PSI=90 MODE)
    '''

    a = np.array([-901.2327248,895.8011176,817.6208321,
	    -845.5880889,-83.73539535,86.58542841,336.8781402,-329.3619944,
	    -311.294712,308.6011161,31.94469304,-31.30824526,125.8739681,
	    -372.3384278,-235.4720434,286.7594095,21.86305585,-27.42344605,
	    -150.4874688,2.669338538,1.395023949,-.5540427503,-56.85224007,
	    3.681827033,-43.48705106,5.103131905,1.073551279,-.6673083508,
	    12.21404266,4.177465543,5.799964188,-.3977802319,-1.044652977,
	    .570356001,3.536082962,-3.222069852,9.620648151,6.082014949,
	    27.75216226,12.44199571,5.122226936,6.982039615,20.12149582,
	    6.150973118,4.663639687,15.73319647,2.303504968,5.840511214,
	    .08385953499,.3477844929])

    p1 = a[36]
    p2 = a[37]
    p3 = a[38]
    r1 = a[39]
    r2 = a[40]
    r3 = a[41]
    q1 = a[42]
    q2 = a[43]
    q3 = a[44]
    s1 = a[45]
    s2 = a[46]
    s3 = a[47]
    t1 = a[48]
    t2 = a[49]

    cps = np.cos(ps)
    sps = np.sin(ps)
    s2ps = cps * 2.

    # MODIFIED HERE (INSTEAD OF SIN(3*PS) I TR
    st1 = np.sin(ps * t1)
    ct1 = np.cos(ps * t1)
    st2 = np.sin(ps * t2)
    ct2 = np.cos(ps * t2)
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2

    # MAKE THE TERMS IN THE 1ST SUM ("PERPENDICULAR" SYMMETRY):

    # I = 1
    sqpr = np.sqrt(1. / (p1 * p1) + 1. / (r1 * r1))
    cyp = np.cos(y / p1)
    syp = np.sin(y / p1)
    czr = np.cos(z1 / r1)
    szr = np.sin(z1 / r1)
    
    expr = np.exp(sqpr * x1)
    fx1 = -sqpr * expr * cyp * szr
    hy1 = expr / p1 * syp * szr
    fz1 = -expr * cyp / r1 * czr
    hx1 = fx1 * ct1 + fz1 * st1
    hz1 = -fx1 * st1 + fz1 * ct1

    sqpr = np.sqrt(1. / (p1 * p1) + 1. / (r2 * r2))
    cyp = np.cos(y / p1)
    syp = np.sin(y / p1)
    czr = np.cos(z1 / r2)
    szr = np.sin(z1 / r2)
    expr = np.exp(sqpr * x1)
    fx2 = -sqpr * expr * cyp * szr
    hy2 = expr / p1 * syp * szr
    fz2 = -expr * cyp / r2 * czr
    hx2 = fx2 * ct1 + fz2 * st1
    hz2 = -fx2 * st1 + fz2 * ct1

    sqpr = np.sqrt(1. / (p1 * p1) + 1. / (r3 * r3))
    cyp = np.cos(y / p1)
    syp = np.sin(y / p1)
    czr = np.cos(z1 / r3)
    szr = np.sin(z1 / r3)
    expr = np.exp(sqpr * x1)
    fx3 = -expr * cyp * (sqpr * z1 * czr + szr / r3 * (x1 + 1. / sqpr))
    hy3 = expr / p1 * syp * (z1 * czr + x1 / r3 * szr / sqpr)

    d__1 = r3
    fz3 = -expr * cyp * (czr * (x1 / (d__1 * d__1) / sqpr + 1.) - z1 / r3 * szr)
    hx3 = fx3 * ct1 + fz3 * st1
    hz3 = -fx3 * st1 + fz3 * ct1

    # I = 2
    sqpr = np.sqrt(1. / (p2 * p2) + 1. / (r1 * r1))
    cyp = np.cos(y / p2)
    syp = np.sin(y / p2)
    czr = np.cos(z1 / r1)
    szr = np.sin(z1 / r1)
    expr = np.exp(sqpr * x1)
    fx4 = -sqpr * expr * cyp * szr
    hy4 = expr / p2 * syp * szr
    fz4 = -expr * cyp / r1 * czr
    hx4 = fx4 * ct1 + fz4 * st1
    hz4 = -fx4 * st1 + fz4 * ct1

    sqpr = np.sqrt(1. / (p2 * p2) + 1. / (r2 * r2))
    cyp = np.cos(y / p2)
    syp = np.sin(y / p2)
    czr = np.cos(z1 / r2)
    szr = np.sin(z1 / r2)
    expr = np.exp(sqpr * x1)
    fx5 = -sqpr * expr * cyp * szr
    hy5 = expr / p2 * syp * szr
    fz5 = -expr * cyp / r2 * czr
    hx5 = fx5 * ct1 + fz5 * st1
    hz5 = -fx5 * st1 + fz5 * ct1

    sqpr = np.sqrt(1. / (p2 * p2) + 1. / (r3 * r3))
    cyp = np.cos(y / p2)
    syp = np.sin(y / p2)
    czr = np.cos(z1 / r3)
    szr = np.sin(z1 / r3)
    expr = np.exp(sqpr * x1)
    fx6 = -expr * cyp * (sqpr * z1 * czr + szr / r3 * (x1 + 1. / sqpr))
    hy6 = expr / p2 * syp * (z1 * czr + x1 / r3 * szr / sqpr)

    fz6 = -expr * cyp * (czr * (x1 / (r3 * r3) / sqpr + 1.) - z1 / r3 * szr)
    hx6 = fx6 * ct1 + fz6 * st1
    hz6 = -fx6 * st1 + fz6 * ct1

    # I = 3
    sqpr = np.sqrt(1. / (p3 * p3) + 1. / (r1 * r1))
    cyp = np.cos(y / p3)
    syp = np.sin(y / p3)
    czr = np.cos(z1 / r1)
    szr = np.sin(z1 / r1)
    expr = np.exp(sqpr * x1)
    fx7 = -sqpr * expr * cyp * szr
    hy7 = expr / p3 * syp * szr
    fz7 = -expr * cyp / r1 * czr
    hx7 = fx7 * ct1 + fz7 * st1
    hz7 = -fx7 * st1 + fz7 * ct1

    sqpr = np.sqrt(1. / (p3 * p3) + 1. / (r2 * r2))
    cyp = np.cos(y / p3)
    syp = np.sin(y / p3)
    czr = np.cos(z1 / r2)
    szr = np.sin(z1 / r2)
    expr = np.exp(sqpr * x1)
    fx8 = -sqpr * expr * cyp * szr
    hy8 = expr / p3 * syp * szr
    fz8 = -expr * cyp / r2 * czr
    hx8 = fx8 * ct1 + fz8 * st1
    hz8 = -fx8 * st1 + fz8 * ct1

    sqpr = np.sqrt(1. / (p3 * p3) + 1. / (r3 * r3))
    cyp = np.cos(y / p3)
    syp = np.sin(y / p3)
    czr = np.cos(z1 / r3)
    szr = np.sin(z1 / r3)
    expr = np.exp(sqpr * x1)
    fx9 = -expr * cyp * (sqpr * z1 * czr + szr / r3 * (x1 + 1. / sqpr))
    hy9 = expr / p3 * syp * (z1 * czr + x1 / r3 * szr / sqpr)

    fz9 = -expr * cyp * (czr * (x1 / (r3 * r3) / sqpr + 1.) - z1 / r3 * szr)
    hx9 = fx9 * ct1 + fz9 * st1
    hz9 = -fx9 * st1 + fz9 * ct1

    a1 = a[0] + a[1] * cps
    a2 = a[2] + a[3] * cps
    a3 = a[4] + a[5] * cps
    a4 = a[6] + a[7] * cps
    a5 = a[8] + a[9] * cps
    a6 = a[10] + a[11] * cps
    a7 = a[12] + a[13] * cps
    a8 = a[14] + a[15] * cps
    a9 = a[16] + a[17] * cps

    bx = a1 * hx1 + a2 * hx2 + a3 * hx3 + a4 * hx4 + a5 * hx5 + a6 * hx6 + a7 * hx7 + a8 * hx8 + a9 * hx9
    by = a1 * hy1 + a2 * hy2 + a3 * hy3 + a4 * hy4 + a5 * hy5 + a6 * hy6 + a7 * hy7 + a8 * hy8 + a9 * hy9
    bz = a1 * hz1 + a2 * hz2 + a3 * hz3 + a4 * hz4 + a5 * hz5 + a6 * hz6 + a7 * hz7 + a8 * hz8 + a9 * hz9

    # MAKE THE TERMS IN THE 2ND SUM ("PARALLEL" SYMMETRY): */

    # I = 1
    sqqs = np.sqrt(1. / (q1 * q1) + 1. / (s1 * s1))
    cyq = np.cos(y / q1)
    syq = np.sin(y / q1)
    czs = np.cos(z2 / s1)
    szs = np.sin(z2 / s1)
    exqs = np.exp(sqqs * x2)
    fx1 = -sqqs * exqs * cyq * czs * sps
    hy1 = exqs / q1 * syq * czs * sps
    fz1 = exqs * cyq / s1 * szs * sps
    hx1 = fx1 * ct2 + fz1 * st2
    hz1 = -fx1 * st2 + fz1 * ct2

    sqqs = np.sqrt(1. / (q1 * q1) + 1. / (s2 * s2))
    cyq = np.cos(y / q1)
    syq = np.sin(y / q1)
    czs = np.cos(z2 / s2)
    szs = np.sin(z2 / s2)
    exqs = np.exp(sqqs * x2)
    fx2 = -sqqs * exqs * cyq * czs * sps
    hy2 = exqs / q1 * syq * czs * sps
    fz2 = exqs * cyq / s2 * szs * sps
    hx2 = fx2 * ct2 + fz2 * st2
    hz2 = -fx2 * st2 + fz2 * ct2

    sqqs = np.sqrt(1. / (q1 * q1) + 1. / (s3 * s3))
    cyq = np.cos(y / q1)
    syq = np.sin(y / q1)
    czs = np.cos(z2 / s3)
    szs = np.sin(z2 / s3)
    exqs = np.exp(sqqs * x2)
    fx3 = -sqqs * exqs * cyq * czs * sps
    hy3 = exqs / q1 * syq * czs * sps
    fz3 = exqs * cyq / s3 * szs * sps
    hx3 = fx3 * ct2 + fz3 * st2
    hz3 = -fx3 * st2 + fz3 * ct2

    # I = 2
    sqqs = np.sqrt(1. / (q2 * q2) + 1. / (s1 * s1))
    cyq = np.cos(y / q2)
    syq = np.sin(y / q2)
    czs = np.cos(z2 / s1)
    szs = np.sin(z2 / s1)
    exqs = np.exp(sqqs * x2)
    fx4 = -sqqs * exqs * cyq * czs * sps
    hy4 = exqs / q2 * syq * czs * sps
    fz4 = exqs * cyq / s1 * szs * sps
    hx4 = fx4 * ct2 + fz4 * st2
    hz4 = -fx4 * st2 + fz4 * ct2

    sqqs = np.sqrt(1. / (q2 * q2) + 1. / (s2 * s2))
    cyq = np.cos(y / q2)
    syq = np.sin(y / q2)
    czs = np.cos(z2 / s2)
    szs = np.sin(z2 / s2)
    exqs = np.exp(sqqs * x2)
    fx5 = -sqqs * exqs * cyq * czs * sps
    hy5 = exqs / q2 * syq * czs * sps
    fz5 = exqs * cyq / s2 * szs * sps
    hx5 = fx5 * ct2 + fz5 * st2
    hz5 = -fx5 * st2 + fz5 * ct2

    sqqs = np.sqrt(1. / (q2 * q2) + 1. / (s3 * s3))
    cyq = np.cos(y / q2)
    syq = np.sin(y / q2)
    czs = np.cos(z2 / s3)
    szs = np.sin(z2 / s3)
    exqs = np.exp(sqqs * x2)
    fx6 = -sqqs * exqs * cyq * czs * sps
    hy6 = exqs / q2 * syq * czs * sps
    fz6 = exqs * cyq / s3 * szs * sps
    hx6 = fx6 * ct2 + fz6 * st2
    hz6 = -fx6 * st2 + fz6 * ct2

    # I = 3
    sqqs = np.sqrt(1. / (q3 * q3) + 1. / (s1 * s1))
    cyq = np.cos(y / q3)
    syq = np.sin(y / q3)
    czs = np.cos(z2 / s1)
    szs = np.sin(z2 / s1)
    exqs = np.exp(sqqs * x2)
    fx7 = -sqqs * exqs * cyq * czs * sps
    hy7 = exqs / q3 * syq * czs * sps
    fz7 = exqs * cyq / s1 * szs * sps
    hx7 = fx7 * ct2 + fz7 * st2
    hz7 = -fx7 * st2 + fz7 * ct2

    sqqs = np.sqrt(1. / (q3 * q3) + 1. / (s2 * s2))
    cyq = np.cos(y / q3)
    syq = np.sin(y / q3)
    czs = np.cos(z2 / s2)
    szs = np.sin(z2 / s2)
    exqs = np.exp(sqqs * x2)
    fx8 = -sqqs * exqs * cyq * czs * sps
    hy8 = exqs / q3 * syq * czs * sps
    fz8 = exqs * cyq / s2 * szs * sps
    hx8 = fx8 * ct2 + fz8 * st2
    hz8 = -fx8 * st2 + fz8 * ct2

    sqqs = np.sqrt(1. / (q3 * q3) + 1. / (s3 * s3))
    cyq = np.cos(y / q3)
    syq = np.sin(y / q3)
    czs = np.cos(z2 / s3)
    szs = np.sin(z2 / s3)
    exqs = np.exp(sqqs * x2)
    fx9 = -sqqs * exqs * cyq * czs * sps
    hy9 = exqs / q3 * syq * czs * sps
    fz9 = exqs * cyq / s3 * szs * sps
    hx9 = fx9 * ct2 + fz9 * st2
    hz9 = -fx9 * st2 + fz9 * ct2
    a1 = a[18] + a[19] * s2ps
    a2 = a[20] + a[21] * s2ps
    a3 = a[22] + a[23] * s2ps
    a4 = a[24] + a[25] * s2ps
    a5 = a[26] + a[27] * s2ps
    a6 = a[28] + a[29] * s2ps
    a7 = a[30] + a[31] * s2ps
    a8 = a[32] + a[33] * s2ps
    a9 = a[34] + a[35] * s2ps

    bx = bx + a1 * hx1 + a2 * hx2 + a3 * hx3 + a4 * hx4 + a5 * hx5 + a6 * hx6 + a7 * hx7 + a8 * hx8 + a9 * hx9
    by = by + a1 * hy1 + a2 * hy2 + a3 * hy3 + a4 * hy4 + a5 * hy5 + a6 * hy6 + a7 * hy7 + a8 * hy8 + a9 * hy9
    bz = bz + a1 * hz1 + a2 * hz2 + a3 * hz3 + a4 * hz4 + a5 * hz5 + a6 * hz6 + a7 * hz7 + a8 * hz8 + a9 * hz9

    return bx, by, bz


# =========
# = WORKS =
# =========

@njit
def deformed(ps, x, y, z, g, tw, d0, rh0):
    '''
    CALCULATES GSM COMPONENTS OF 104 UNIT-AMPLITUDE TAIL FIELD MODES,
    TAKING INTO ACCOUNT BOTH EFFECTS OF DIPOLE TILT:
    WARPING IN Y-Z (DONE BY THE S/R WARPED) AND BENDING IN X-Z (DONE BY THIS SUBROUTINE)
    '''

    rh2 = -5.2
    ieps = 3

    bxe = np.empty((5, 4))
    bye = np.empty((5, 4))
    bze = np.empty((5, 4))
    bxo = np.empty((5, 4))
    byo = np.empty((5, 4))
    bzo = np.empty((5, 4))
    bxs = np.empty(5)
    bys = np.empty(5)
    bzs = np.empty(5)

    # RH0,RH1,RH2, AND IEPS CONTROL THE TILT-RELATED DEFORMATION OF THE TAIL FIELD
    sps = np.sin(ps)
    cps = np.sqrt(1. - sps**2)
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    zr = z / r

    rh = rh0 + rh2 * (zr**2)
    drhdr = -zr / r * 2.0 * rh2 * zr
    drhdz = rh2 * 2.0 * zr / r

    rrh = r / rh
    f = 1.0 / (rrh**ieps + 1.0)**(1.0 / ieps)
    i__1 = ieps - 1
    i__2 = ieps + 1
    dfdr = -rrh**(ieps - 1) * f**(ieps + 1) / rh
    dfdrh = -rrh * dfdr

    spsas = sps * f
    cpsas = np.sqrt(1.0 - spsas**2)

    xas = x * cpsas - z * spsas
    zas = x * spsas + z * cpsas

    facps = sps / cpsas * (dfdr + dfdrh * drhdr) / r
    psasx = facps * x
    psasy = facps * y
    psasz = facps * z + sps / cpsas * dfdrh * drhdz

    dxasdx = cpsas - zas * psasx
    dxasdy = -zas * psasy
    dxasdz = -spsas - zas * psasz
    dzasdx = spsas + xas * psasx
    dzasdy = xas * psasy
    dzasdz = cpsas + xas * psasz
    fac1 = dxasdz * dzasdy - dxasdy * dzasdz
    fac2 = dxasdx * dzasdz - dxasdz * dzasdx
    fac3 = dzasdx * dxasdy - dxasdx * dzasdy

    # DEFORM:
    bxass, byass, bzass, bxaso, byaso, bzaso, bxase, byase, bzase = warped(ps, xas, y, zas, g, tw, d0)

    # New tail structure
    for k in np.arange(1, 5 + 1):
        bxs[k - 1] = bxass[k - 1] * dzasdz - bzass[k - 1] * dxasdz + byass[k - 1] * fac1
        bys[k - 1] = byass[k - 1] * fac2
        bzs[k - 1] = bzass[k - 1] * dxasdx - bxass[k - 1] * dzasdx + byass[k - 1] * fac3

    for k in np.arange(1, 5 + 1):
        for l in np.arange(1, 4 + 1):
            bxo[k - 1, l - 1] = bxaso[k - 1, l - 1] * dzasdz - bzaso[k - 1, l - 1] * dxasdz + byaso[k - 1, l - 1] * fac1
            byo[k - 1, l - 1] = byaso[k - 1, l - 1] * fac2
            bzo[k - 1, l - 1] = bzaso[k - 1, l - 1] * dxasdx - bxaso[k - 1, l - 1] * dzasdx + byaso[k - 1, l - 1] * fac3
            bxe[k - 1, l - 1] = bxase[k - 1, l - 1] * dzasdz - bzase[k - 1, l - 1] * dxasdz + byase[k - 1, l - 1] * fac1
            bye[k - 1, l - 1] = byase[k - 1, l - 1] * fac2
            bze[k - 1, l - 1] = bzase[k - 1, l - 1] * dxasdx - bxase[k - 1, l - 1] * dzasdx + byase[k - 1, l - 1] * fac3

    return bxs, bys, bzs, bxo, byo, bzo, bxe, bye, bze


# =========
# = WORKS =
# =========

@njit
def warped(ps, x, y, z, g, tw, d0):
    '''
    CALCULATES GSM COMPONENTS OF THE WARPED FIELD FOR TWO TAIL UNIT MODES.
    THE WARPING DEFORMATION IS IMPOSED ON THE UNWARPED FIELD, COMPUTED
    BY THE S/R "UNWARPED".  THE WARPING PARAMETERS WERE TAKEN FROM THE
    RESULTS OF GEOTAIL OBSERVATIONS (TSYGANENKO ET AL. [1998]).
    NB # 6, P.106, OCT 12, 2000.
    '''

    bxo = np.empty((5, 4))
    byo = np.empty((5, 4))
    bzo = np.empty((5, 4))
    bxe = np.empty((5, 4))
    bye = np.empty((5, 4))
    bze = np.empty((5, 4))
    bxs = np.empty(5)
    bys = np.empty(5)
    bzs = np.empty(5)

    dgdx = 0.0
    xl = 20.0
    dxldx = 0.0
    sps = np.sin(ps)

    rho2 = y**2 + z**2
    rho = np.sqrt(rho2)

    if (y == 0.0 and z == 0.0):
        phi = 0.0
        cphi = 1.0
        sphi = 0.0
    else:
        phi = np.arctan2(z, y)
        cphi = y / rho
        sphi = z / rho

    rr4l4 = rho / (rho2**2 + xl**4)
    f = phi + g * rho2 * rr4l4 * cphi * sps + tw * (x / 10.0)
    dfdphi = 1.0 - g * rho2 * rr4l4 * sphi * sps

    dfdrho = g * (rr4l4**2) * (xl**4 * 3.0 - rho2**2) * cphi * sps

    dfdx = rr4l4 * cphi * sps * (dgdx * rho2 - g * rho * rr4l4 * 4.0 * (xl**3) * dxldx) + tw / 10.

    # THE LAST TERM DESCRIBES THE IMF-INDUCED TWIS
    cf = np.cos(f)
    sf = np.sin(f)
    yas = rho * cf
    zas = rho * sf
    bx_ass, by_ass, bz_ass, bx_aso, by_aso, bz_aso, bx_ase, by_ase, bz_ase = unwarped(x, yas, zas, d0)

    for k in np.arange(1, 5 + 1):
        # Deforming symmetric modules
        brho_as = by_ass[k - 1] * cf + bz_ass[k - 1] * sf
        bphi_as = -by_ass[k - 1] * sf + bz_ass[k - 1] * cf
        brho_s = brho_as * dfdphi
        bphi_s = bphi_as - rho * (bx_ass[k - 1] * dfdx + brho_as * dfdrho)
        bxs[k - 1] = bx_ass[k - 1] * dfdphi
        bys[k - 1] = brho_s * cphi - bphi_s * sphi
        bzs[k - 1] = brho_s * sphi + bphi_s * cphi

    for k in np.arange(1, 5 + 1):
        for l in np.arange(1, 4 + 1):
            # Deforming odd modules
            brho_as = by_aso[k - 1, l - 1] * cf + bz_aso[k - 1, l - 1] * sf
            bphi_as = -by_aso[k - 1, l - 1] * sf + bz_aso[k - 1, l - 1] * cf
            brho_s = brho_as * dfdphi
            bphi_s = bphi_as - rho * (bx_aso[k - 1, l - 1] * dfdx + brho_as * dfdrho)

            bxo[k - 1, l - 1] = bx_aso[k - 1, l - 1] * dfdphi
            byo[k - 1, l - 1] = brho_s * cphi - bphi_s * sphi
            bzo[k - 1, l - 1] = brho_s * sphi + bphi_s * cphi

            # Deforming even modules
            brho_as = by_ase[k - 1, l - 1] * cf + bz_ase[k - 1, l - 1] * sf
            bphi_as = -by_ase[k - 1, l - 1] * sf + bz_ase[k - 1, l - 1] * cf
            brho_s = brho_as * dfdphi
            bphi_s = bphi_as - rho * (bx_ase[k - 1, l - 1] * dfdx + brho_as * dfdrho)

            bxe[k - 1, l - 1] = bx_ase[k - 1, l - 1] * dfdphi
            bye[k - 1, l - 1] = brho_s * cphi - bphi_s * sphi
            bze[k - 1, l - 1] = brho_s * sphi + bphi_s * cphi

    return bxs, bys, bzs, bxo, byo, bzo, bxe, bye, bze


# =========
# = WORKS =
# =========

@njit
def unwarped(x, y, z, d0):
    '''
    CALCULATES GSM COMPONENTS OF THE SHIELDED FIELD OF 45 TAIL MODES WITH UNIT
    AMPLITUDES,  WITHOUT ANY WARPING OR BENDING.  NONLINEAR PARAMETERS OF THE MODES
    ARE FORWARDED HERE VIA A COMMON BLOCK /TAIL/.

    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's
    improvements to the Bessel function evaluation.
    '''

    bxo = np.empty((5, 4))
    byo = np.empty((5, 4))
    bzo = np.empty((5, 4))
    bxe = np.empty((5, 4))
    bye = np.empty((5, 4))
    bze = np.empty((5, 4))
    bxs = np.empty(5)
    bys = np.empty(5)
    bzs = np.empty(5)

    ajm = np.empty(6)
    ajmd = np.empty(6)

    rnot = 20.0

    # New tail structure
    # Rho_0 - scale parameter along the tail axis
    for k in np.arange(1, 5 + 1):
        rho = np.sqrt(x**2 + y**2)
        rkmr = float(k * rho / rnot)

        # July 2017, G.K.Stephens, all the Bessel functions are now evaluated first,
        # and passed into the subroutines
        ajm = bessjj(5, rkmr)

        # get all n in one call
        for m in np.arange(1, 5 + 1):
            ajmd[m] = ajm[m - 1] - m * ajm[m] / rkmr

        ajmd[0] = -ajm[1]

        bxsk, bysk, bzsk = tailsht_s(k, x, y, z, ajm, d0)
        hxsk, hysk, hzsk = shtbnorm_s(k, x, y, z)

        bxs[k - 1] = bxsk + hxsk
        bys[k - 1] = bysk + hysk
        bzs[k - 1] = bzsk + hzsk
        
        for l in np.arange(1, 4 + 1):
            bxokl, byokl, bzokl = tailsht_oe(1, k, l, x, y, z, ajm, ajmd, d0)
            hxokl, hyokl, hzokl = shtbnorm_o(k, l, x, y, z)
            bxo[k - 1, l - 1] = bxokl + hxokl
            byo[k - 1, l - 1] = byokl + hyokl
            bzo[k - 1, l - 1] = bzokl + hzokl
            bxekl, byekl, bzekl = tailsht_oe(0, k, l, x, y, z, ajm, ajmd, d0)
            hxekl, hyekl, hzekl = shtbnorm_e(k, l, x, y, z)
            bxe[k - 1, l - 1] = bxekl + hxekl
            bye[k - 1, l - 1] = byekl + hyekl
            bze[k - 1, l - 1] = bzekl + hzekl

    return bxs, bys, bzs, bxo, byo, bzo, bxe, bye, bze


# =========
# = WORKS =
# =========

@njit
def tailsht_s(m, x, y, z, ajm, d0):
    '''
    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's
    improvements to the Bessel function evaluation. The Bessel function
    values are now precomputed and  passed into this rather than computed
    inside this routine.
    '''

    # THE COMMON BLOCKS FORWARDS TAIL SHEET THICKNESS
    # This can be replaced by introducing them */
    # through the above common block */
    rnot = 20.0
    dltk = 1.0

    rho = np.sqrt(x**2 + y**2)
    csphi = x / rho
    snphi = y / rho

    dkm = (m - 1) * dltk + 1.0
    rkm = dkm / rnot

    rkmz = rkm * z
    rkmr = rkm * rho
    zd = np.sqrt(z**2 + d0**2)

    rj0 = ajm[0]
    rj1 = ajm[1]

    # July 2017, G.K.Stephens, Bessel functions are now passed in.
    rex = np.exp(rkm * zd)

    bx = rkmz * rj1 * csphi / zd / rex
    by = rkmz * rj1 * snphi / zd / rex
    bz = rkm * rj0 / rex

    # CALCULATION OF THE MAGNETOTAIL CURRENT CONTRIBUTION IS FINISHED
    return bx, by, bz


# =========
# = WORKS =
# =========

@njit
def shtbnorm_s(k, x, y, z):
    '''
    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's
    improvements to the Bessel function evaluation.
    '''

    ak = np.empty(5)
    ajm = np.empty(15)
    ajmd = np.empty(15)

    # modified SHTBNORM_S
    ak[0] = tss[76 - 1, k - 1]
    ak[1] = tss[77 - 1, k - 1]
    ak[2] = tss[78 - 1, k - 1]
    ak[3] = tss[79 - 1, k - 1]
    ak[4] = tss[80 - 1, k - 1]

    phi = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)

    rhoi = 0.0
    if (rho < 1e-8):
        rhoi = 1e8
    else:
        rhoi = 1.0 / rho

    dpdx = -y * rhoi**2
    dpdy = x * rhoi**2

    fx = 0.0
    fy = 0.0
    fz = 0.0

    for n in np.arange(1, 5 + 1):
        akn = np.abs(ak[n - 1])
        aknr = akn * rho

        aknri = 0.0
        if (aknr < 1e-8):
            aknri = 1e8
        else:
            aknri = 1.0 / aknr

        chz = np.cosh(z * akn)
        shz = np.sinh(z * akn)
        ajm = bessjj(14, aknr)

        # get all n in one call
        for m in np.arange(1, 14 + 1):
            ajmd[m] = ajm[m - 1] - m * ajm[m] * aknri

        ajmd[0] = -ajm[1]

        for m in np.arange(0, 14 + 1):
            cmp_ = np.cos(m * phi)
            smp = np.sin(m * phi)

            hx1 = m * dpdx * smp * shz * ajm[m]
            hx2 = -akn * x * rhoi * cmp_ * shz * ajmd[m]
            hx = hx1 + hx2

            hy1 = m * dpdy * smp * shz * ajm[m]
            hy2 = -akn * y * rhoi * cmp_ * shz * ajmd[m]
            hy = hy1 + hy2

            hz = -akn * cmp_ * chz * ajm[m]

            l = n + m * 5

            fx += hx * tss[l - 1, k - 1]
            fy += hy * tss[l - 1, k - 1]
            fz += hz * tss[l - 1, k - 1]

    return fx, fy, fz


# =========
# = WORKS =
# =========

@njit
def tailsht_oe(ievo, mk, m, x, y, z, ajm, ajmd, d0):
    '''
    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's
    improvements to the Bessel function evaluation. The Bessel function
    values are now precomputed and  passed into this rather than computed
    inside this routine.
    '''

    # THE COMMON BLOCKS FORWARDS TAIL SHEET THICKNES
    rnot = 20.0

    # Rho_0 - scale parameter along the tail axis
    dltk = 1.0

    # step in Km
    rho = np.sqrt(x**2 + y**2)

    csphi = x / rho
    snphi = y / rho

    phi = np.arctan2(y, x)
    csmphi = np.cos(m * phi)
    snmphi = np.sin(m * phi)

    dkm = (mk - 1) * dltk + 1.0
    rkm = dkm / rnot

    rkmz = rkm * z
    rkmr = rkm * rho

    zd = np.sqrt(z**2 + d0**2)

    rex = np.exp(rkm * zd)

    # July 2017, G.K.Stephens, Jm is now passed in, not computed internally
    # calculating Jm and its derivatives 

    if (ievo == 0):
        # calculating symmetric modes
        bro = -m * snmphi * z * ajmd[m] / zd / rex
        bphi = -m**2 * csmphi * z * ajm[m] / rkmr / zd / rex
        bz = m * snmphi * ajm[m] / rex
    else:
        # calculating asymmetric modes
        bro = m * csmphi * z * ajmd[m] / zd / rex
        bphi = -m**2 * snmphi * z * ajm[m] / rkmr / zd / rex
        bz = -m * csmphi * ajm[m] / rex

    # transformation from cylindrical ccordinates to GSM
    bx = bro * csphi - bphi * snphi
    by = bro * snphi + bphi * csphi

    # CALCULATION OF THE MAGNETOTAIL CURRENT CONTRIBUTION IS FINISHED */
    return bx, by, bz


# =========
# = WORKS =
# =========

@njit
def shtbnorm_o(k, l, x, y, z):
    '''
    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's */
    improvements to the Bessel function evaluation. */
    '''

    ak = np.empty(5)
    ajm = np.empty(15)
    ajmd = np.empty(15)

    ak[0] = tso[76 - 1, k - 1, l - 1]
    ak[1] = tso[77 - 1, k - 1, l - 1]
    ak[2] = tso[78 - 1, k - 1, l - 1]
    ak[3] = tso[79 - 1, k - 1, l - 1]
    ak[4] = tso[80 - 1, k - 1, l - 1]

    phi = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)

    if (rho < 1e-8):
        rhoi = 1e8
    else:
        rhoi = 1.0 / rho

    dpdx = -y * rhoi**2
    dpdy = x * rhoi**2

    fx = 0.0
    fy = 0.0
    fz = 0.0

    for n in np.arange(1, 5 + 1):
        akn = np.abs(ak[n - 1])
        aknr = akn * rho

        if (aknr < 1e-8):
            aknri = 1e8
        else:
            aknri = 1.0 / aknr

        chz = np.cosh(z * akn)
        shz = np.sinh(z * akn)

        ajm = bessjj(14, aknr)

        # get all n in one call
        for m in np.arange(1, 14 + 1):
            ajmd[m] = ajm[m - 1] - m * ajm[m] * aknri

        ajmd[0] = -ajm[1]

        for m in np.arange(0, 14 + 1):
            cmp = np.cos(m * phi)
            smp = np.sin(m * phi)
            hx1 = m * dpdx * smp * shz * ajm[m]
            hx2 = -akn * x * rhoi * cmp * shz * ajmd[m]
            hx = hx1 + hx2
            hy1 = m * dpdy * smp * shz * ajm[m]
            hy2 = -akn * y * rhoi * cmp * shz * ajmd[m]
            hy = hy1 + hy2
            hz = -akn * cmp * chz * ajm[m]
            l1 = n + m * 5
            fx += hx * tso[l1 - 1, k - 1, l - 1]
            fy += hy * tso[l1 - 1, k - 1, l - 1]
            fz += hz * tso[l1 - 1, k - 1, l - 1]

    return fx, fy, fz


# =========
# = WORKS =
# =========

@njit
def shtbnorm_e(k, l, x, y, z):
    '''
    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's */
    improvements to the Bessel function evaluation. */
    '''
    ak = np.empty(5)

    ak[0] = tse[76 - 1, k - 1, l - 1]
    ak[1] = tse[77 - 1, k - 1, l - 1]
    ak[2] = tse[78 - 1, k - 1, l - 1]
    ak[3] = tse[79 - 1, k - 1, l - 1]
    ak[4] = tse[80 - 1, k - 1, l - 1]

    phi = np.arctan2(y, x)
    rho = np.sqrt(x * x + y * y)

    rhoi = 0.0
    if (rho < 1e-8):
        rhoi = 1e8
    else:
        rhoi = 1.0 / rho

    dpdx = -y * rhoi * rhoi
    dpdy = x * rhoi * rhoi

    fx = 0.0
    fy = 0.0
    fz = 0.0

    for n in np.arange(1, 5 + 1):
        akn = np.abs(ak[n - 1])
        aknr = akn * rho

        aknri = 0.0
        if (aknr < 1e-8):
            aknri = 1e8
        else:
            aknri = 1.0 / aknr

        chz = np.cosh(z * akn)
        shz = np.sinh(z * akn)

        ajm = bessjj(14, aknr)
        ajmd = np.empty(15)

        for m in np.arange(1, 14 + 1):
            ajmd[m] = ajm[m - 1] - m * ajm[m] * aknri

        ajmd[0] = -ajm[1]

        for m in np.arange(0, 14 + 1):
            cmp = np.cos(m * phi)
            smp = np.sin(m * phi)

            hx1 = -m * dpdx * cmp * shz * ajm[m]
            hx2 = -akn * x * rhoi * smp * shz * ajmd[m]
            hx = hx1 + hx2

            hy1 = -m * dpdy * cmp * shz * ajm[m]
            hy2 = -akn * y * rhoi * smp * shz * ajmd[m]
            hy = hy1 + hy2

            hz = -akn * smp * chz * ajm[m]

            l1 = n + m * 5

            fx += hx * tse[l1 - 1, k - 1, l - 1]
            fy += hy * tse[l1 - 1, k - 1, l - 1]
            fz += hz * tse[l1 - 1, k - 1, l - 1]

    return fx, fy, fz 


# =========
# = WORKS =
# =========

@njit
def bessjj(n, x):
    '''
    July 2017, G.K.Stephens, this routine updated to incorporate Jay Albert's
    improvements to the Bessel function evaluation.
    bessJ holds J0 to Jn
    '''

    bessj = np.empty(n + 1)

    ax = np.abs(x)
    tox = 2.0 / ax

    # start at some large m, larger than the desired n, multiply by 2 to ensure m starts at an even number
    m = int((n + int(np.sqrt(n * 40.0))) * 0.5) * 2
    evnsum = 0.0

    # keeps track of the sum of the even Js (J0+J2+J4+...)
    iseven = False

    # we set the value of Jm to some arbitrary value, here Jm=1, after the loop
    # is done, the values will be normalized using the sum
    bjp = 0.0
    bj = 1.0

    # initialize to zero
    i = n
    for i in np.arange(0, n + 1):
        bessj[i] = 0.0

    for j in np.arange(m, 0, -1):
        # the next value int the recursion relation J_n-1 = (2*n/x)*Jn - J_n+1
        bjm = j * tox * bj - bjp
        bjp = bj

        # decrement so shift J_n+1 ot Jn
        bj = bjm

        # if the value gets too large, shift the decimal of everything by 10 places
        # decrement so shift J_n ot J_n-1

        if (np.abs(bj) > 1e10):
            bj *= 1e-10
            bjp *= 1e-10
            evnsum *= 1e-10
            for i in np.arange(j + 1, n + 1):
                bessj[i] *= 1e-10

        if (iseven):
            evnsum += bj

        # only sum over the even Jns
        iseven = not iseven
        if (j <= n):
            bessj[j] = bjp

    # sum is currently the sum of all the evens
    # use Miller's algorithm for Bessel functions which uses the identity:
    # 1.0 = 2.0*sum(J_evens) - J0, thus the quantity (2.0*sum(J_evens) - J0)
    # is used as a normalization factor
    bnorm = evnsum * 2.0 - bj

    # normalize the Bessel functions 
    for i in np.arange(1, n + 1):
        bessj[i] /= bnorm

    bessj[0] = bj / bnorm

    # Apply Jn(-x) = (-1)^n * Jn(x)
    # J0(x)

    if (x < 0.0):
        for i in np.arange(1, n + 1, 2):
            bessj[i] = -bessj[i]

    return bessj


# =========
# = WORKS =
# =========

# Needs ps, x, y, z, xkappa1, xkappa2
# Returns bx11, by11, bz11, bx12, by12, bz12, bx21, by21, bz21, bx22, by22, bz22
@njit
def birk_tot(ps, x, y, z, xkappa1, xkappa2):
    '''
    INPUT PARAMETERS, SPECIFIED
    PARAMETERS, CONTROL DAY
    ====   LEAST SQUARES FITTING ONLY:
          BX11=0.D0
          BY11=0.D0
          BZ11=0.D0
          BX12=0.D0
          BY12=0.D0
          BZ12=0.D0
          BX21=0.D0
          BY21=0.D0
          BZ21=0.D0
          BX22=0.D0
          BY22=0.D0
          BZ22=0.D0
    ===================================
    '''

    sh11 = np.array([46488.84663,  -15541.95244, -23210.09824, -32625.03856,
                     -109894.4551, -71415.32808, 58168.94612,  55564.87578,
	             -22890.60626, -6056.763968, 5091.3681,    239.7001538,
                     -13899.49253, 4648.016991,  6971.310672,  9699.351891,
                     32633.34599,  21028.48811,  -17395.9619,  -16461.11037,
                     7447.621471,  2528.844345,  -1934.094784, -588.3108359,
                     -32588.88216, 10894.11453,  16238.25044,  22925.60557,
	             77251.11274,  50375.97787,  -40763.78048, -39088.6066,
                     15546.53559,  3559.617561,  -3187.730438, 309.1487975,
                     88.22153914,  -243.0721938, -63.63543051, 191.1109142,
                     69.94451996,  -187.9539415, -49.89923833, 104.0902848,
                     -120.2459738, 253.5572433,  89.25456949,  -205.6516252,
	             -44.93654156, 124.7026309,  32.53005523,  -98.85321751,
                     -36.51904756, 98.8824169,   24.88493459,  -55.04058524,
                     61.14493565,  -128.4224895, -45.3502346,  105.0548704,
                     -43.66748755, 119.3284161,  31.38442798,  -92.87946767,
                     -33.52716686, 89.98992001,  25.87341323,  -48.86305045,
	             59.69362881,  -126.5353789, -44.39474251, 101.5196856,
                     59.41537992,  41.18892281,  80.861012,    3.066809418,
                     7.893523804,  30.56212082,  10.36861082,  8.222335945,
                     19.97575641,  2.050148531,  4.992657093,  2.300564232,
                     0.2256245602, -0.05841594319])

    sh12 = np.array([210260.4816,  -1443587.401, -1468919.281,  281939.2993,
                     -1131124.839, 729331.7943,  2573541.307,   304616.7457,
	             468887.5847,  181554.7517,  -1300722.65,   -257012.8601,
                     645888.8041,  -2048126.412, -2529093.041,  571093.7972,
                     -2115508.353, 1122035.951,  4489168.802,   75234.22743,
                     823905.6909,  147926.6121,  -2276322.876,  -155528.5992,
                     -858076.2979, 3474422.388,  3986279.931,   -834613.9747,
	             3250625.781,  -1818680.377, -7040468.986,  -414359.6073,
                     -1295117.666, -346320.6487, 3565527.409,   430091.9496,
                     -.1565573462, 7.377619826,  .4115646037,   -6.14607888,
                     3.808028815,  -.5232034932, 1.454841807,   -12.32274869,
                     -4.466974237, -2.941184626, -0.6172620658, 12.6461349,
	             1.494922012,  -21.35489898, -1.65225696,   16.81799898,
                     -1.404079922, -24.09369677, -10.99900839,  45.9423782,
                     2.248579894,  31.91234041,  7.575026816,   -45.80833339,
                     -1.507664976, 14.60016998,  1.348516288,   -11.05980247,
                     -5.402866968, 31.69094514,  12.28261196,   -37.55354174,
	             4.155626879,  -33.70159657, -8.437907434,  36.22672602,
                     145.0262164,  70.73187036,  85.51110098,   21.47490989,
                     24.34554406,  31.34405345,  4.655207476,   5.747889264,
                     7.802304187,  1.844169801,  4.86725455,    2.941393119,
                     0.1379899178, 0.06607020029])

    sh21 = np.array([162294.6224,  503885.1125,  -27057.67122, -531450.1339,
                     84747.05678,  -237142.1712, 84133.6149,   259530.0402,
	             69196.0516,   -189093.5264, -19278.55134, 195724.5034,
                     -263082.6367, -818899.6923, 43061.10073,  863506.6932,
                     -139707.9428, 389984.885,   -135167.5555, -426286.9206,
                     -109504.0387, 295258.3531,  30415.07087,  -305502.9405,
                     100785.34,    315010.9567,  -15999.50673, -332052.2548,
	             54964.34639,  -152808.375,  51024.67566,  166720.0603,
                     40389.67945,  -106257.7272, -11126.14442, 109876.2047,
                     2.978695024,  558.6019011,  2.685592939,  -338.000473,
                     -81.9972409,  -444.1102659, 89.44617716,  212.0849592,
                     -32.58562625, -982.7336105, -35.10860935, 567.8931751,
	             -1.917212423, -260.2023543, -1.023821735, 157.5533477,
                     23.00200055,  232.0603673,  -36.79100036, -111.9110936,
                     18.05429984,  447.0481,     15.10187415,  -258.7297813,
                     -1.032340149, -298.6402478, -1.676201415, 180.5856487,
                     64.52313024,  209.0160857,  -53.8557401,  -98.5216429,
	             14.35891214,  536.7666279,  20.09318806,  -309.734953,
                     58.54144539,  67.4522685,   97.92374406,  4.75244976,
                     10.46824379,  32.9185611,   12.05124381,  9.962933904,
                     15.91258637,  1.804233877,  6.578149088,  2.515223491,
                     0.1930034238, -0.02261109942])

    sh22 = np.array([-131287.8986, -631927.6885, -318797.4173, 616785.8782,
                     -50027.36189, 863099.9833,  47680.2024,   -1053367.944,
	             -501120.3811, -174400.9476, 222328.6873,  333551.7374,
                     -389338.7841, -1995527.467, -982971.3024, 1960434.268,
                     297239.7137,  2676525.168,  -147113.4775, -3358059.979,
                     -2106979.191, -462827.1322, 1017607.96,   1039018.475,
                     520266.9296,  2627427.473,  1301981.763,  -2577171.706,
	             -238071.9956, -3539781.111, 94628.1642,   4411304.724,
                     2598205.733,  637504.9351,  -1234794.298, -1372562.403,
                     -2.646186796, -31.10055575, 2.295799273,  19.20203279,
                     30.01931202,  -302.102855,  -14.78310655, 162.1561899,
                     0.4943938056, 176.8089129,  -0.244492168, -100.6148929,
	             9.172262228,  137.430344,   -8.451613443, -84.20684224,
                     -167.3354083, 1321.830393,  76.89928813,  -705.7586223,
                     18.28186732,  -770.1665162, -9.084224422, 436.3368157,
                     -6.374255638, -107.2730177, 6.080451222,  65.53843753,
                     143.2872994,  -1028.009017, -64.2273933,  547.8536586,
	             -20.58928632, 597.3893669,  10.17964133,  -337.7800252,
                     159.3532209,  76.34445954,  84.74398828,  12.76722651,
                     27.63870691,  32.69873634,  5.145153451,  6.310949163,
                     6.996159733,  1.971629939,  4.436299219,  2.904964304,
                     0.1486276863, 0.06859991529])

    xkappa = xkappa1

    # FORWARDED IN BIRK_1N2
    x_sc = xkappa1 - 1.1

    # FORWARDED IN BIRK_SHL
    fx11, fy11, fz11 = birk_1n2(1, 1, ps, x, y, z, xkappa)

    # REGION 1
    hx11, hy11, hz11 = birk_shl(sh11, ps, x_sc, x, y, z)

    bx11 = fx11 + hx11
    by11 = fy11 + hy11
    bz11 = fz11 + hz11

    fx12, fy12, fz12 = birk_1n2(1, 2, ps, x, y, z, xkappa)

    # REGION 1
    hx12, hy12, hz12 = birk_shl(sh12, ps, x_sc, x, y, z)

    bx12 = fx12 + hx12
    by12 = fy12 + hy12
    bz12 = fz12 + hz12

    xkappa = xkappa2

    # FORWARDED IN BIRK_1N2
    x_sc = xkappa2 - 1.0

    # FORWARDED IN BIRK_SHL
    fx21, fy21, fz21 = birk_1n2(2, 1, ps, x, y, z, xkappa)

    # REGION 2
    hx21, hy21, hz21 = birk_shl(sh21, ps, x_sc, x, y, z)

    bx21 = fx21 + hx21
    by21 = fy21 + hy21
    bz21 = fz21 + hz21

    fx22, fy22, fz22 = birk_1n2(2, 2, ps, x, y, z, xkappa)

    # REGION 2
    hx22, hy22, hz22 = birk_shl(sh22, ps, x_sc, x, y, z)

    bx22 = fx22 + hx22
    by22 = fy22 + hy22
    bz22 = fz22 + hz22

    return bx11, by11, bz11, bx12, by12, bz12, bx21, by21, bz21, bx22, by22, bz22


# =========
# = WORKS =
# =========

# Needs numb, mode, ps, x, y, z, xkappa
# Returns bx, by, bz
@njit
def birk_1n2(numb, mode, ps, x, y, z, xkappa):
    '''
    calculates components  of region 1/2 field in spherical coords.  derived from the s/r dipdef2c (which
    does the same job, but input/output there was in spherical coords, while here we use cartesian ones)

    input:  numb=1 (2) for region 1 (2) currents
            mode=1 yields simple sinusoidal mlt variation, with maximum current at dawn/dusk meridian
    while mode=2 yields the second harmonic.

    see n
    (1) dphi:   half-difference (in radians) between day and night latitude of fac oval at ionospheric altitude;
                typical value: 0.06
    (2) b:      an asymmetry factor at high-altitudes;  for b=0, the only asymmetry is that from dphi
                typical values: 0.35-0.70
    (3) rho_0:  a fixed parameter, defining the distance rho, at which the latitude shift gradually saturates and
                stops increasing
                its value was assumed fixed, equal to 7.0.
    (4) xkappa: an overall scaling factor, which can be used for changing the size of the f.a.c. oval

    these parameters control
    parameters of the tilt-depend
    '''

    beta = 0.9
    rh = 10.0
    eps = 3.0

    a11 = np.array([.161806835,   -.1797957553, 2.999642482,    -.9322708978,
                    -.681105976,  .2099057262,  -8.358815746,   -14.8603355,
	            .3838362986,  -16.30945494, 4.537022847,    2.685836007,
                    27.97833029,  6.330871059,  1.876532361,    18.95619213,
                    .96515281,    .4217195118,  -.0895777002,   -1.823555887,
                    .7457045438,  -.5785916524, -1.010200918,   .01112389357,
                    .09572927448, -.3599292276, 8.713700514,    .9763932955,
	            3.834602998,  2.492118385,  .7113544659])

    a12 = np.array([.705802694,   -.2845938535, 5.715471266,    -2.47282088,
                    -.7738802408, .347829393,   -11.37653694,   -38.64768867,
	            .6932927651,  -212.4017288, 4.944204937,    3.071270411,
                    33.05882281,  7.387533799,  2.366769108,    79.22572682,
                    .6154290178,  .5592050551,  -.1796585105,   -1.65493221,
                    .7309108776,  -.4926292779, -1.130266095,   -.009613974555,
                    .1484586169,  -.2215347198, 7.883592948,    .02768251655,
	            2.950280953,  1.212634762,  .5567714182])

    a21 = np.array([.1278764024,  -.2320034273, 1.805623266,    -32.3724144,
                    -.9931490648, .317508563,   -2.492465814,   -16.21600096,
	            .2695393416,  -6.752691265, 3.971794901,    14.54477563,
                    41.10158386,  7.91288973,   1.258297372,    9.583547721,
                    1.014141963,  .5104134759,  -.1790430468,   -1.756358428,
                    .7561986717,  -.6775248254, -.0401401642,   .01446794851,
                    .1200521731,  -.2203584559, 4.50896385,     .8221623576,
	            1.77993373,   1.102649543,  .886788002])

    a22 = np.array([.4036015198,  -.3302974212, 2.82773093,     -45.4440583,
                    -1.611103927, .4927112073,  -.003258457559, -49.59014949,
	            .3796217108,  -233.7884098, 4.31266698,     18.05051709,
                    28.95320323,  11.09948019,  .7471649558,    67.10246193,
                    .5667096597,  .6468519751,  -.1560665317,   -1.460805289,
                    .7719653528,  -.6658988668, 2.515179349e-6, .02426021891,
                    .1195003324,  -.2625739255, 4.377172556,    .2421190547,
	            2.503482679,  1.071587299,  .724799743])

    # THESE PARAMETERS CONTROL
    # parameters of the tilt-depend
    b = 0.5
    rho_0 = 7.0
    modenum = mode

    dphi = 0.0
    dtheta = 0.0

    if (numb == 1):
        dphi = .055
        dtheta = .06

    if (numb == 2):
        dphi = .03
        dtheta = .09

    xsc = x * xkappa
    ysc = y * xkappa
    zsc = z * xkappa

    rho = np.sqrt(xsc * xsc + zsc * zsc)
    rsc = np.sqrt(xsc * xsc + ysc * ysc + zsc * zsc)

    rho2 = rho_0 * rho_0
    phi = 0.0

    if (xsc == 0.0 and zsc == 0.0):
        phi = 0.0
    else:
        phi = np.arctan2(-zsc, xsc)

    # FROM CARTESIAN TO CYLINDRICAL (RHO,PHI
    sphic = np.sin(phi)
    cphic = np.cos(phi)

    # "C" means "CYLINDRICAL", TO DISTINGUISH FROM S
    brack = dphi + b * rho2 / (rho2 + 1.0) * (rho * rho - 1.0) / (rho2 + rho * rho)
    r1rh = (rsc - 1.0) / rh

    psias = beta * ps / np.power(np.power(r1rh, eps) + 1.0, 1.0 / eps)
    phis = phi - brack * np.sin(phi) - psias
    dphisphi = 1.0 - brack * np.cos(phi)

    dphisrho = b * -2.0 * rho2 * rho / (rho2 + rho * rho)**2 * np.sin(phi) + beta * ps * np.power(r1rh, eps - 1.0) * rho / (rh * rsc * np.power(np.power(r1rh, eps) + 1.0, 1.0 / eps + 1.0))

    dphisdy = beta * ps * np.power(r1rh, eps - 1.0) * ysc / (rh * rsc * np.power(np.power(r1rh, eps) + 1.0, 1.0 / eps + 1.0))
    sphics = np.sin(phis)
    cphics = np.cos(phis)
    xs = rho * cphics
    zs = -rho * sphics

    if (numb == 1):
        if (mode == 1):
            bxs, byas, bzs = twocones(a11, xs, ysc, zs, dtheta, modenum)
        if (mode == 2):
            bxs, byas, bzs = twocones(a12, xs, ysc, zs, dtheta, modenum)
    else:
        if (mode == 1):
            bxs, byas, bzs = twocones(a21, xs, ysc, zs, dtheta, modenum)
        if (mode == 2):
            bxs, byas, bzs = twocones(a22, xs, ysc, zs, dtheta, modenum)

    brhoas = bxs * cphics - bzs * sphics
    bphias = -bxs * sphics - bzs * cphics
    brho_s = brhoas * dphisphi * xkappa
    bphi_s = (bphias - rho * (byas * dphisdy + brhoas * dphisrho)) * xkappa
    by_s = byas * dphisphi * xkappa

    bx = brho_s * cphic - bphi_s * sphic
    by = by_s
    bz = -brho_s * sphic - bphi_s * cphic

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs a, x, y, z, dtheta, modenum
# Returns bx, by, bz
@njit
def twocones(a, x, y, z, dtheta, modenum):
    '''
    ADDS FIELDS FROM TWO CONES (NORTHERN AND SOUTHERN), WITH A PROPER SYMMETRY OF THE CURRENT AND FIELD,
    CORRESPONDING TO THE REGION 1 BIRKELAND CURRENTS. (SEE NB #6, P.58).
    '''

    bxn, byn, bzn = one_cone(a, x, y, z, dtheta, modenum)
    bxs, bys, bzs = one_cone(a, x, -y, -z, dtheta, modenum)

    bx = bxn - bxs
    by = byn + bys
    bz = bzn + bzs

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs a, ps, x_sx, x, y, z
# Returns bx, by, bz
@njit
def birk_shl(a, ps, x_sc, x, y, z):
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = cps * 2.0

    pst1 = ps * a[85 - 1]
    pst2 = ps * a[86 - 1]
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2

    l = 0
    gx = 0.0
    gy = 0.0
    gz = 0.0

    for m in np.arange(1, 2 + 1):
        # AND M=2 IS FOR THE SECOND SUM ("PARALL." SYMMETRY)
        # M=1 IS FOR THE 1ST SUM ("PERP." SYMMETRY)
        for i in np.arange(1, 3 + 1):
            p = a[i + 72 - 1]
            q = a[i + 78 - 1]
            cypi = np.cos(y / p)
            cyqi = np.cos(y / q)
            sypi = np.sin(y / p)
            syqi = np.sin(y / q)

            for k in  np.arange(1, 3 + 1):
                r = a[k + 75 - 1]
                s = a[k + 81 - 1]
                szrk = np.sin(z1 / r)
                czsk = np.cos(z2 / s)
                czrk = np.cos(z1 / r)
                szsk = np.sin(z2 / s)
                sqpr = np.sqrt(1.0 / (p * p) + 1.0 / (r * r))
                sqqs = np.sqrt(1.0 / (q * q) + 1.0 / (s * s))
                epr = np.exp(x1 * sqpr)
                eqs = np.exp(x2 * sqqs)

                for n in np.arange(1, 2 + 1):
                    # AND N=2 IS FOR THE SECOND ONE
                    # N=1 IS FOR THE FIRST PART OF EACH COEFFI
                    for nn in np.arange(1, 2 + 1):
                        # TO TAKE INTO ACCOUNT THE SCALE FACTOR DEPENDENCE
                        # NN = 1,2 FURTHER SPLITS THE COEFFICI
                        if (m == 1):
                            fx = -sqpr * epr * cypi * szrk
                            fy = epr * sypi * szrk / p
                            fz = -epr * cypi * czrk / r
                            if (n == 1):
                                if (nn == 1):
                                    hx = fx
                                    hy = fy
                                    hz = fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if (nn == 1):
                                    hx = fx * cps
                                    hy = fy * cps
                                    hz = fz * cps
                                else:
                                    hx = fx * cps * x_sc
                                    hy = fy * cps * x_sc
                                    hz = fz * cps * x_sc
                        else:
                            # M.EQ.2
                            fx = -sps * sqqs * eqs * cyqi * czsk
                            fy = sps / q * eqs * syqi * czsk
                            fz = sps / s * eqs * cyqi * szsk
                            if (n == 1):
                                if (nn == 1):
                                    hx = fx
                                    hy = fy
                                    hz = fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if (nn == 1):
                                    hx = fx * s3ps
                                    hy = fy * s3ps
                                    hz = fz * s3ps
                                else:
                                    hx = fx * s3ps * x_sc
                                    hy = fy * s3ps * x_sc
                                    hz = fz * s3ps * x_sc
                        l += 1
                        if (m == 1):
                            hxr = hx * ct1 + hz * st1
                            hzr = -hx * st1 + hz * ct1
                        else:
                            hxr = hx * ct2 + hz * st2
                            hzr = -hx * st2 + hz * ct2

                        gx += hxr * a[l - 1]
                        gy += hy * a[l - 1]
                        gz += hzr * a[l - 1]
    bx = gx
    by = gy
    bz = gz

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs xkappa1, xkappa2
# Sets xkappa
# Returns bx11, by11, bz11, bx12, by12, bz12, bx21, by21, bz21, bx22, by22, bz22
@njit
def birtotsy(ps, x, y, z, xkappa1, xkappa2):
    '''
    this s/r is almost identical to birk_tot, but it is for the symmetric mode, in which 
    j_parallel is an even function of ygsm.

    iopbs -  birkeland field mode flag:
    iopbs=0 - all components
    iopbs=1 - region 1, modes 1 & 2 (symmetric !)
    iopbs=2 - region 2, modes 1 & 2 (symmetric !)

    (joint with  birk_tot  for the antisymmetrical mode)

    input parameters, specified

    parameters, control day
    '''

    sh11 = np.array([4956703.683,   -26922641.21, -11383659.85, 29604361.65,
                     -38919785.97,  70230899.72,  34993479.24,  -90409215.02,
	             30448713.69,   -48360257.19, -35556751.23, 57136283.6,
                     -8013815.613,  30784907.86,  13501620.5,   -35121638.52,
                     50297295.45,   -84200377.18, -46946852.58, 107526898.8,
                     -39003263.47,  59465850.17,  47264335.1,   -68892388.73,
                     3375901.533,   -9181255.754, -4494667.217, 10812618.51,
	             -17351920.97,  27016083.0,   18150032.11,  -33186882.96,
                     13340198.63,   -19779685.3,  -17891788.15, 21625767.23,
                     16135.32442,   133094.0241,  -13845.61859, -79159.98442,
                     432.1215298,   -85438.10368, 1735.386707,  41891.71284,
                     18158.14923,   -105465.8135, -11685.73823, 62297.34252,
	             -10811.08476,  -87631.38186, 9217.499261,  52079.94529,
                     -68.29127454,  56023.02269,  -1246.029857, -27436.42793,
                     -11972.61726,  69607.08725,  7702.743803,  -41114.3681,
                     12.08269108,   -21.30967022, -9.100782462, 18.26855933,
                     -7.000685929,  26.22390883,  6.392164144,  -21.99351743,
	             2.294204157,   -16.10023369, -1.34431475,  9.34212123,
                     148.5493329,   99.79912328,  70.78093196,  35.23177574,
                     47.45346891,   58.44877918,  139.8135237,  91.96485261,
                     6.983488815,   9.055554871,  19.80484284,  2.860045019,
                     0.08213262337, -7.962186676e-6])

    sh12 = np.array([-1210748.72,   -52324903.95, -14158413.33, 19426123.6,
                     6808641.947,   -5138390.983, -1118600.499, -4675055.459,
	             2059671.506,   -1373488.052, -114704.4353, -1435920.472,
                     1438451.655,   61199067.17,  16549301.39,  -22802423.47,
                     -7814550.995,  5986478.728,  1299443.19,   5352371.724,
                     -2994351.52,   1898553.337,  203158.3658,  2270182.134,
                     -618083.3112,  -25950806.16, -7013783.326, 9698792.575,
	             3253693.134,   -2528478.464, -546323.4095, -2217735.237,
                     1495336.589,   -914647.4222, -114374.1054, -1200441.634,
                     -507068.47,    1163189.975,  998411.8381,  -861919.3631,
                     5252210.872,   -11668550.16, -4113899.385, 6972900.95,
                     -2546104.076,  7704014.31,   2273077.192,  -5134603.198,
	             256205.7901,   -589970.8086, -503821.017,  437612.8956,
                     -2648640.128,  5887640.735,  2074286.234,  -3519291.144,
                     1283847.104,   -3885817.147, -1145936.942, 2589753.651,
                     -408.7788403,  1234.054185,  739.8541716,  -965.8068853,
                     3691.383679,   -8628.635819, -2855.844091, 5268.500178,
	             -1774.372703,  5515.010707,  1556.089289,  -3665.43466,
                     204.8672197,   110.7748799,  87.36036207,  5.52249133,
                     31.0636427,    73.57632579,  281.533136,   140.3461448,
                     17.07537768,   6.729732641,  4.100970449,  2.780422877,
                     0.08742978101, -1.028562327e-5])

    sh21 = np.array([-67763516.61,  -49565522.84, 10123356.08,  51805446.1,
                     -51607711.68,  164360662.1,  -4662006.024, -191297217.6,
	             -7204547.103,  30372354.93,  -750371.9365, -36564457.17,
                     61114395.65,   45702536.5,   -9228894.939, -47893708.68,
                     47290934.33,   -149155112.,  4226520.638,  173588334.5,
                     7998505.443,   -33150962.72, 832493.2094,  39892545.84,
                     -11303915.16,  -8901327.398, 1751557.11,   9382865.82,
	             -9054707.868,  27918664.5,   -788741.7146, -32481294.42,
                     -2264443.753,  9022346.503,  -233526.0185, -10856269.53,
                     -244450.885,   1908295.272,  185445.1967,  -1074202.863,
                     41827.75224,   -241553.7626, -20199.1258,  123235.6084,
                     199501.4614,   -1936498.464, -178857.4074, 1044724.507,
	             121044.9917,   -946479.9247, -91808.28803, 532742.7569,
                     -20742.28628,  120633.2193,  10018.49534,  -61599.11035,
                     -98709.58977,  959095.177,   88500.43489,  -517471.5287,
                     -81.56122911,  816.2472344,  55.3071171,   -454.5368824,
                     25.7469381,    -202.500735,  -7.369350794, 104.9429812,
	             58.14049362,   -685.5919355, -51.71345683, 374.0125033,
                     247.9296982,   159.2471769,  102.3151816,  15.81062488,
                     34.99767599,   133.0832773,  219.6475201,  107.9582783,
                     10.00264684,   7.718306072,  25.22866153,  5.013583103,
                     0.08407754233, -9.613356793e-6])

    sh22 = np.array([-43404887.31,  8896854.538,  -8077731.036, -10247813.65,
                     6346729.086,   -9416801.212, -1921670.268, 7805483.928,
	             2299301.127,   4856980.17,   -1253936.462, -4695042.69,
                     54305735.91,   -11158768.1,  10051771.85,  12837129.47,
                     -6380785.836,  12387093.5,   1687850.192,  -10492039.47,
                     -5777044.862,  -6916507.424, 2855974.911,  7027302.49,
                     -26176628.93,  5387959.61,   -4827069.106, -6193036.589,
	             2511954.143,   -6205105.083, -553187.2984, 5341386.847,
                     3823736.361,   3669209.068,  -1841641.7,   -3842906.796,
                     281561.722,    -5013124.63,  379824.5943,  2436137.901,
                     -76337.55394,  548518.2676,  42134.28632,  -281711.3841,
                     -365514.8666,  -2583093.138, -232355.8377, 1104026.712,
	             -131536.3445,  2320169.882,  -174967.6603, -1127251.881,
                     35539.82827,   -256132.9284, -19620.06116, 131598.7965,
                     169033.6708,   1194443.5,    107320.3699,  -510672.0036,
                     1211.177843,   -17278.19863, 1140.037733,  8347.612951,
                     -303.8408243,  2405.771304,  174.0634046,  -1248.72295,
	             -1231.229565,  -8666.932647, -754.0488385, 3736.878824,
                     227.2102611,   115.9154291,  94.3436483,   3.625357304,
                     64.03192907,   109.0743468,  241.4844439,  107.7583478,
                     22.36222385,   6.282634037,  27.79399216,  2.270602235,
                     0.08708605901, -1.256706895e-5])

    xkappa = xkappa1

    # FORWARDED IN BIR1N2SY
    x_sc = xkappa1 - 1.1

    # FORWARDED IN BIRSH_SY
    fx11, fy11, fz11 = bir1n2sy(1, 1, ps, x, y, z, xkappa)

    # REGION 1
    hx11, hy11, hz11 = birsh_sy(sh11, ps, x_sc, x, y, z)

    bx11 = fx11 + hx11
    by11 = fy11 + hy11
    bz11 = fz11 + hz11

    fx12, fy12, fz12 = bir1n2sy(1, 2, ps, x, y, z, xkappa)

    # REGION 1
    hx12, hy12, hz12 = birsh_sy(sh12, ps, x_sc, x, y, z)

    bx12 = fx12 + hx12
    by12 = fy12 + hy12
    bz12 = fz12 + hz12

    xkappa = xkappa2

    # FORWARDED IN BIR1N2SY
    x_sc = xkappa2 - 1.0

    # FORWARDED IN BIRSH_SY
    fx21, fy21, fz21 = bir1n2sy(2, 1, ps, x, y, z, xkappa)

    # REGION 2
    hx21, hy21, hz21 = birsh_sy(sh21, ps, x_sc, x, y, z)
    bx21 = fx21 + hx21
    by21 = fy21 + hy21
    bz21 = fz21 + hz21
    fx22, fy22, fz22 = bir1n2sy(2, 2, ps, x, y, z, xkappa)

    # REGION 2,
    hx22, hy22, hz22 = birsh_sy(sh22, ps, x_sc, x, y, z)
    bx22 = fx22 + hx22
    by22 = fy22 + hy22
    bz22 = fz22 + hz22

    return bx11, by11, bz11, bx12, by12, bz12, bx21, by21, bz21, bx22, by22, bz22


# =========
# = WORKS =
# =========

# Needs numb, mode, ps, x, y, z, xkappa
# Sets b, rho_0, modenum, dphi, dtheta
# Returns bx, by, bz
@njit
def bir1n2sy(numb, mode, ps, x, y, z, xkappa):
    '''
    THIS CODE IS VERY SIMILAR TO BIRK_1N2, BUT IT IS FOR THE "SYMMETRICAL" MODE, IN WHICH J_parallel
    IS A SYMMETRIC (EVEN) FUNCTION OF Ygsm

    CALCULATES COMPONENTS  OF REGION 1/2 FIELD IN SPHERICAL COORDS.  DERIVED FROM THE S/R DIPDEF2C (WHICH
      DOES THE SAME JOB, BUT INPUT/OUTPUT THERE WAS IN SPHERICAL COORDS, WHILE HERE WE USE CARTESIAN ONES)

    INPUT:  NUMB=1 (2) FOR REGION 1 (2) CURRENTS
             MODE=1 YIELDS SIMPLE SINUSOIDAL MLT VARIATION, WITH MAXIMUM CURRENT AT DAWN/DUSK MERIDIAN
       WHILE MODE=2 YIELDS THE SECOND HARMONIC.


    SEE N
    (1) DPHI:   HALF-DIFFERENCE (IN RADIANS) BETWEEN DAY AND NIGHT LATITUDE OF FAC OVAL AT IONOSPHERIC ALTITUDE
                TYPICAL VALUE: 0.06
    (2) B:      AN ASYMMETRY FACTOR AT HIGH-ALTITUDES;  FOR B=0, THE ONLY ASYMMETRY IS THAT FROM DPHI
                TYPICAL VALUES: 0.35-0.70
    (3) RHO_0:  A FIXED PARAMETER, DEFINING THE DISTANCE RHO, AT WHICH THE LATITUDE SHIFT GRADUALLY SATURATES AND
                  STOPS INCREASING
                  ITS VALUE WAS ASSUMED FIXED, EQUAL TO 7.0.
    (4) XKAPPA: AN OVERALL SCALING FACTOR, WHICH CAN BE USED FOR CHANGING THE SIZE OF THE F.A.C. OVAL
    '''

    beta = 0.9
    rh = 10.0
    eps = 3.0

    a11 = np.array([0.161806835,   -0.1797957553, 2.999642482,     -0.9322708978,
                    -0.681105976,  0.2099057262,  -8.358815746,    -14.8603355,
	            0.3838362986,  -16.30945494,  4.537022847,     2.685836007,
                    27.97833029,   6.330871059,   1.876532361,     18.95619213,
                    0.96515281,    0.4217195118,  -0.0895777002,   -1.823555887,
                    0.7457045438,  -0.5785916524, -1.010200918,    0.01112389357,
                    0.09572927448, -0.3599292276, 8.713700514,     0.9763932955,
                    3.834602998,   2.492118385,   0.7113544659])

    a12 = np.array([0.705802694,   -0.2845938535, 5.715471266,     -2.47282088,
                    -0.7738802408, 0.347829393,   -11.37653694,    -38.64768867,
	            0.6932927651,  -212.4017288,  4.944204937,     3.071270411,
                    33.05882281,   7.387533799,   2.366769108,     79.22572682,
                    0.6154290178,  0.5592050551,  -0.1796585105,   -1.65493221,
                    0.7309108776,  -0.4926292779, -1.130266095,    -0.009613974555,
                    0.1484586169,  -0.2215347198, 7.883592948,     0.02768251655,
	            2.950280953,   1.212634762,   0.5567714182])

    a21 = np.array([0.1278764024,  -0.2320034273, 1.805623266,     -32.3724144,
                    -0.9931490648, 0.317508563,   -2.492465814,    -16.21600096,
	            0.2695393416,  -6.752691265,  3.971794901,     14.54477563,
                    41.10158386,   7.91288973,    1.258297372,     9.583547721,
                    1.014141963,   0.5104134759,  -0.1790430468,   -1.756358428,
                    0.7561986717,  -0.6775248254, -0.0401401642,   0.01446794851,
                    0.1200521731,  -0.2203584559, 4.50896385,      0.8221623576,
	            1.77993373,    1.102649543,   0.886788002])

    a22 = np.array([0.4036015198,  -0.3302974212, 2.82773093,      -45.4440583,
                    -1.611103927,  0.4927112073,  -0.003258457559, -49.59014949,
	            0.3796217108,  -233.7884098,  4.31266698,      18.05051709,
                    28.95320323,   11.09948019,   0.7471649558,    67.10246193,
                    0.5667096597,  0.6468519751,  -0.1560665317,   -1.460805289,
                    0.7719653528,  -0.6658988668, 2.515179349e-6,  0.02426021891,
                    0.1195003324,  -0.2625739255, 4.377172556,     0.2421190547,
	            2.503482679,   1.071587299,   0.724799743])

    # THESE PARAMETERS CONTROL
    # parameters of the tilt-depend
    b = 0.5
    rho_0 = 7.0
    modenum = mode

    dphi = 0.0
    dtheta = 0.0

    if (numb == 1):
        dphi = .055
        dtheta = .06
    if (numb == 2):
        dphi = .03
        dtheta = .09

    xsc = x * xkappa
    ysc = y * xkappa
    zsc = z * xkappa

    rho = np.sqrt(xsc**2 + zsc**2)
    rsc = np.sqrt(xsc**2 + ysc**2 + zsc**2)

    rho2 = rho_0**2
    if (xsc == 0.0 and zsc == 0.0):
        phi = 0.0
    else:
        phi = np.arctan2(-zsc, xsc)

    # FROM CARTESIAN TO CYLINDRICAL (RHO,PHI)
    sphic = np.sin(phi)
    cphic = np.cos(phi)

    # "C" means "CYLINDRICAL", TO DISTINGUISH FROM S
    brack = dphi + b * rho2 / (rho2 + 1.) * (rho**2 - 1.0) / (rho2 + rho**2)
    r1rh = (rsc - 1.0) / rh
    psias = beta * ps / (r1rh**eps + 1.0)**(1.0 / eps)
    phis = phi - brack * np.sin(phi) - psias
    dphisphi = 1.0 - brack * np.cos(phi)

    dphisrho = b * -2.0 * rho2 * rho / (rho2 + rho**2)**2 * np.sin(phi) + beta * ps * r1rh**(eps - 1.0) * rho / (rh * rsc * (r1rh**eps + 1.0)**(1.0 / eps + 1.0));
    dphisdy = beta * ps * r1rh**(eps - 1.0) * ysc / (rh * rsc * (r1rh**eps + 1.0)**(1.0 / eps + 1.0))

    sphics = np.sin(phis)
    cphics = np.cos(phis)

    xs = rho * cphics
    zs = -rho * sphics

    if (numb == 1):
        if (mode == 1):
            bxs, byas, bzs = twoconss(a11, xs, ysc, zs, dtheta, modenum)
        if (mode == 2):
            bxs, byas, bzs = twoconss(a12, xs, ysc, zs, dtheta, modenum)
    else:
        if (mode == 1):
            bxs, byas, bzs = twoconss(a21, xs, ysc, zs, dtheta, modenum)
        if (mode == 2):
            bxs, byas, bzs = twoconss(a22, xs, ysc, zs, dtheta, modenum)

    brhoas = bxs * cphics - bzs * sphics
    bphias = -bxs * sphics - bzs * cphics
    brho_s = brhoas * dphisphi * xkappa
    bphi_s = (bphias - rho * (byas * dphisdy + brhoas * dphisrho)) * xkappa
    by_s = byas * dphisphi * xkappa

    bx = brho_s * cphic - bphi_s * sphic
    by = by_s
    bz = -brho_s * sphic - bphi_s * cphic

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs a, x, y, z, modenum
# Returns bx, by, bz
@njit
def twoconss(a, x, y, z, dtheta, modenum):
    '''
    DIFFERS FROM TWOCONES:  THIS S/R IS FOR THE "SYMMETRIC" MODE OF BIRKELAND CURRENTS IN THAT */
                            HERE THE FIELD IS ROTATED BY 90 DEGS FOR M=1 AND BY 45 DEGS FOR M=2 */

    ADDS FIELDS FROM TWO CONES (NORTHERN AND SOUTHERN), WITH A PROPER SYMMETRY OF THE CURRENT AND FIELD, */
    CORRESPONDING TO THE REGION 1 BIRKELAND CURRENTS. (SEE NB #6, P.58). */
    '''

    hsqr2 = 0.707106781

    if (modenum == 1):
        # ROTATION BY 90 DEGS 
        xas = y
        yas = -x
    else:
        # ROTATION BY 45 DEGS
        xas = (x + y) * hsqr2
        yas = (y - x) * hsqr2
    
    bxn, byn, bzn = one_cone(a, xas, yas, z, dtheta, modenum)
    bxs, bys, bzs = one_cone(a, xas, -yas, -z, dtheta, modenum)
    bxas = bxn - bxs
    byas = byn + bys
    bz = bzn + bzs

    if (modenum == 1):
        # ROTATION BY 90 DEGS
        bx = -byas
        by = bxas
    else:
        bx = (bxas - byas) * hsqr2
        by = (bxas + byas) * hsqr2

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs a, x, y, z, dtheta, modenum
# Returns bx, by, bz
@njit
def one_cone(a, x, y, z, dtheta, modenum):
    '''
    RETURNS FIELD COMPONENTS FOR A DEFORMED CONICAL CURRENT SYSTEM, FITTED TO A BIOSAVART FIELD
    BY SIM_14.FOR.  HERE ONLY THE NORTHERN CONE IS TAKEN INTO ACCOUNT.
    '''

    dr = 1e-6
    dt = 1e-6

    # JUST FOR NUMERICAL DIFFERENTIATION
    theta0 = a[31 - 1]

    rho2 = x**2 + y**2
    rho = np.sqrt(rho2)
    r = np.sqrt(rho2 + z**2)
    theta = np.arctan2(rho, z)
    phi = np.arctan2(y, x)

    # MAKE THE DEFORMATION OF COORDINATES:
    rs = r_s(a, r, theta)
    thetas = theta_s(a, r, theta)
    phis = phi

    # CALCULATE FIELD COMPONENTS AT THE NEW POSITION (ASTERISKED):
    btast, bfast = fialcos(rs, thetas, phis, modenum, theta0, dtheta)

    # NOW TRANSFORM B{R,T,F}_AST BY THE DEFORMATION TENSOR:
    # FIRST OF ALL, FIND THE DERIVATIVES:

    drsdr = (r_s(a, r + dr, theta) - r_s(a, r - dr, theta)) / (dr * 2.0)
    drsdt = (r_s(a, r, theta + dt) - r_s(a, r, theta - dt)) / (dt * 2.0)
    dtsdr = (theta_s(a, r + dr, theta) - theta_s(a, r - dr, theta)) / (dr * 2.0)
    dtsdt = (theta_s(a, r, theta + dt) - theta_s(a, r, theta - dt)) / (dt * 2.0)
    stsst = np.sin(thetas) / np.sin(theta)
    rsr = rs / r
    br = -rsr / r * stsst * btast * drsdt

    btheta = rsr * stsst * btast * drsdr
    bphi = rsr * bfast * (drsdr * dtsdt - drsdt * dtsdr)

    s = rho / r
    c = z / r
    sf = y / rho
    cf = x / rho
    be = br * s + btheta * c

    bx = a[1 - 1] * (be * cf - bphi * sf)
    by = a[1 - 1] * (be * sf + bphi * cf)
    bz = a[1 - 1] * (br * c - btheta * s)

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs r, theta, phi, n, theta0, dt
# Returns btheta, bphi
@njit
def fialcos(r, theta, phi, n, theta0, dt):
    '''
    CONICAL MODEL OF BIRKELAND CURRENT FIELD; BASED ON THE OLD S/R FIALCO (OF 1990-91) */
    SEE THE OLD NOTEBOOK 1985-86-88, NOTE OF MARCH 5, BUT HERE BOTH INPUT AND OUTPUT ARE IN SPHERICAL CDS. */
    BTN, AND BPN ARE THE ARRAYS OF BTHETA AND BPHI (BTN(i), BPN(i) CORRESPOND TO i-th MODE). */
    ONLY FIRST  N  MODE AMPLITUDES ARE COMPUTED (N<=10). */
    THETA0 IS THE ANGULAR HALF-WIDTH OF THE CONE, DT IS THE ANGULAR H.-W. OF THE CURRENT LAYER */
    NOTE:  BR=0  (BECAUSE ONLY RADIAL CURRENTS ARE PRESENT IN THIS MODEL) */
    '''
    bpn = np.empty(10)
    btn = np.empty(10)
    ccos = np.empty(10)
    ssin = np.empty(10)

    sinte = np.sin(theta)
    ro = r * sinte
    coste = np.cos(theta)
    sinfi = np.sin(phi)
    cosfi = np.cos(phi)
    tg = sinte / (coste + 1.0)

    # TAN(THETA/2)
    ctg = sinte / (1.0 - coste)

    # CTG(THETA/2)
    tetanp = theta0 + dt
    tetanm = theta0 - dt

    if (theta >= tetanm):
        tgp = np.tan(tetanp * 0.5)
        tgm = np.tan(tetanm * 0.5)
        tgm2 = tgm * tgm
        tgp2 = tgp * tgp

    cosm1 = 1.0
    sinm1 = 0.0
    tm = 1.0
    tgm2m = 1.0
    tgp2m = 1.0
    i = n

    for m in np.arange(1, n + 1):
        tm *= tg
        ccos[m - 1] = cosm1 * cosfi - sinm1 * sinfi
        ssin[m - 1] = sinm1 * cosfi + cosm1 * sinfi
        cosm1 = ccos[m - 1]
        sinm1 = ssin[m - 1]
        if (theta < tetanm):
            t = tm
            dtt = m * 0.5 * tm * (tg + ctg)
            dtt0 = 0.0
        elif (theta < tetanp):
            tgm2m *= tgm2
            fc = 1.0 / (tgp - tgm)
            fc1 = 1.0 / (2.0 * m + 1)
            tgm2m1 = tgm2m * tgm
            tg21 = tg**2 + 1.0
            t = fc * (tm * (tgp - tg) + fc1 * (tm * tg - tgm2m1 / tm))
            dtt = m * 0.5 * fc * tg21 * (tm / tg * (tgp - tg) - fc1 * (tm - tgm2m1 / (tm * tg)))
            dtt0 = fc * 0.5 * ((tgp + tgm) * (tm * tg - fc1 * (tm * tg - tgm2m1 / tm)) + tm * (1.0 - tgp * tgm) - (tgm2 + 1.0) * tgm2m / tm)
        else:
            tgp2m *= tgp2
            tgm2m *= tgm2
            fc = 1.0 / (tgp - tgm)
            fc1 = 1.0 / (2.0 * m + 1)
            t = fc * fc1 * (tgp2m * tgp - tgm2m * tgm) / tm
            dtt = -t * m * 0.5 * (tg + ctg)

        btn[m - 1] = m * t * ccos[m - 1] / ro
        bpn[m - 1] = -dtt * ssin[m - 1] / r

    btheta = btn[n - 1] * 800.0
    bphi = bpn[n - 1] * 800.0

    return btheta, bphi


# =========
# = WORKS =
# =========

# Needs a, r, theta
# Returns ret_val
@njit
def r_s(a, r, theta):
    ret_val = r + a[2 - 1] / r + a[3 - 1] * r / np.sqrt(r**2 + a[11 - 1]**2) + a[4 - 1] * r / (r**2 + a[12 - 1]**2) + (a[5 - 1] + a[6 - 1] / r + a[7 - 1] * r / np.sqrt(r**2 + a[13 - 1]**2) + a[8 - 1] * r / (r**2 + a[14 - 1]**2)) * np.cos(theta) + (a[9 - 1] * r / np.sqrt(r**2 + a[15 - 1]**2) + a[10 - 1] * r / (r**2 + a[16 - 1]**2)**2) * np.cos(theta * 2.0)

    return ret_val


# =========
# = WORKS =
# =========

# Needs a, r, theta
# Returns ret_val
@njit
def theta_s(a, r, theta):
    ret_val = theta + (a[17 - 1] + a[18 - 1] / r + a[19 - 1] / r**2 + a[20 - 1] * r / np.sqrt(r**2 + a[27 - 1]**2)) * np.sin(theta) + (a[21 - 1] + a[22 - 1] * r / np.sqrt(r**2 + a[28 - 1]**2) + a[23 - 1] * r / (r**2 + a[29 - 1]**2)) * np.sin(theta * 2.0) + (a[24 - 1] + a[25 - 1] / r + a[26 - 1] * r / (r**2 + a[30 - 1]**2)) * np.sin(theta * 3.0)

    return ret_val


# =========
# = WORKS =
# =========

# Needs a, ps, x_sc, x, y, z
# Returns bx, by, bz
@njit
def birsh_sy(a, ps, x_sc, x, y, z):
    # this s/r is quite similar to birk_shl, but it is for the symmetric mode of birkeland current field */
    # and for that reason the field components have a different kind of symmetry with respect to y_gsm */

    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = cps * 2.0

    pst1 = ps * a[85 - 1]
    pst2 = ps * a[86 - 1]
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2

    l = 0
    gx = 0.0
    gy = 0.0
    gz = 0.0

    for m in np.arange(1, 2 + 1):
        for i in np.arange(1, 3 + 1):
            p = a[i + 72 - 1]
            q = a[i + 78 - 1]
            cypi = np.cos(y / p)
            cyqi = np.cos(y / q)
            sypi = np.sin(y / p)
            syqi = np.sin(y / q)

            for k in np.arange(1, 3 + 1):
                r = a[k + 75 - 1]
                s = a[k + 81 - 1]
                szrk = np.sin(z1 / r)
                czsk = np.cos(z2 / s)
                czrk = np.cos(z1 / r)
                szsk = np.sin(z2 / s)

                sqpr = np.sqrt(1. / (p * p) + 1. / (r * r))
                sqqs = np.sqrt(1. / (q * q) + 1. / (s * s))

                epr = np.exp(x1 * sqpr)
                eqs = np.exp(x2 * sqqs)

                for n in np.arange(1, 2 + 1):
                    for nn in np.arange(1, 2 + 1):
                        if (m == 1):
                            fx = sqpr * epr * sypi * szrk
                            fy = epr * cypi * szrk / p
                            fz = epr * sypi * czrk / r
                            if (n == 1):
                                if (nn == 1):
                                    hx = fx
                                    hy = fy
                                    hz = fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if (nn == 1):
                                    hx = fx * cps
                                    hy = fy * cps
                                    hz = fz * cps
                                else:
                                    hx = fx * cps * x_sc
                                    hy = fy * cps * x_sc
                                    hz = fz * cps * x_sc
                        else:
                            fx = sps * sqqs * eqs * syqi * czsk
                            fy = sps / q * eqs * cyqi * czsk
                            fz = -sps / s * eqs * syqi * szsk
                            if (n == 1):
                                if (nn == 1):
                                    hx = fx
                                    hy = fy
                                    hz = fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if (nn == 1):
                                    hx = fx * s3ps
                                    hy = fy * s3ps
                                    hz = fz * s3ps
                                else:
                                    hx = fx * s3ps * x_sc
                                    hy = fy * s3ps * x_sc
                                    hz = fz * s3ps * x_sc
                        l += 1
                        if (m == 1):
                            hxr = hx * ct1 + hz * st1
                            hzr = -hx * st1 + hz * ct1
                        else:
                            hxr = hx * ct2 + hz * st2
                            hzr = -hx * st2 + hz * ct2

                        gx += hxr * a[l - 1]
                        gy += hy * a[l - 1]
                        gz += hzr * a[l - 1]

    bx = gx
    by = gy
    bz = gz

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs ps, x, y, z
# Returns bx, by, bz
@njit
def dipole(ps, x, y, z):
    '''
    a double precision routine

    calculates gsm components of a geodipole field with the dipole moment
    corresponding to the epoch of 2000.

    ----input parameters:
    ps - geodipole tilt angle in radians,
    x,y,z - gsm coordinates in re (1 re = 6371.2 km)

    ----output parameters:
    bx,by,bz - field components in gsm system, in nanotesla.

    last modification: jan. 5, 2001. the value of the dipole moment was updated to 2000.
    and a "save" statement has been added, to avoid potential problems with some
    fortran compilers

    written by: n. a. tsyganenko
    '''

    m = 0.0
    psi = 5.0

    if (m != 1 or np.abs(ps - psi) >= 1e-5):
        sps = np.sin(ps)
        cps = np.cos(ps)
        psi = ps
        m = 1

    p = x**2
    t = y**2
    u = z**2
    v = z * 3.0 * x

    q = 30115.0 / np.sqrt(p + t + u)**5

    bx = q * ((t + u - p * 2.0) * sps - v * cps)
    by = y * -3.0 * q * (x * sps + z * cps)
    bz = q * ((p + t - u * 2.0) * cps - v * sps)

    return bx, by, bz


# =========
# = WORKS =
# =========

# Needs xn_pd, vel, xgsm, ygsm, zgsm
# Returns xmgnp, ymgnp, zmgnp, dist, id_
@njit
def t96_mgnp_d(xn_pd, vel, xgsm, ygsm, zgsm):
    '''
    double-precision version !!!!!!!!   hence the suffix "d" in the name

    for any point of space with given coordinates (xgsm,ygsm,zgsm), this subroutine defines
    the position of a point (xmgnp,ymgnp,zmgnp) at the t96 model magnetopause, having the
    same value of the ellipsoidal tau-coordinate, and the distance between them.  this is
    not the shortest distance d_min to the boundary, but dist asymptotically tends to d_min,
    as the observation point gets closer to the magnetopause.

    input: xn_pd - either solar wind proton number density (per c.c.) (if vel>0)
    or the solar wind ram pressure in nanopascals   (if vel<0)
    vel - either solar wind velocity (km/sec)
    or any negative number, which indicates that xn_pd stands
    for the solar wind pressure, rather than for the density

    xgsm,ygsm,zgsm - coordinates of the observation point in earth radii

    output: xmgnp,ymgnp,zmgnp - gsm position of the boundary point, having the same
    value of tau-coordinate as the observation point (xgsm,ygsm,zgsm)
    dist -  the distance between the two points, in re,
    id -    position flag; id=+1 (-1) means that the point (xgsm,ygsm,zgsm)
    lies inside (outside) the model magnetopause, respectively.

    the pressure-dependent magnetopause is that used in the t96_01 model
    (tsyganenko, jgr, v.100, p.5599, 1995; esa sp-389, p.181, oct. 1996)

    author:  n.a. tsyganenko
    date:    aug.1, 1995, revised april 3, 2003.
    '''

    # define solar wind dynamic pressure (nanopascals, assuming 4% of alpha-particles),
    # if not explicitly specified in the input:
    if (vel < 0.):
        pd = xn_pd
    else:
        pd = xn_pd * 1.94e-6 * vel**2

    rat = 0.5 * pd
    rat16 = rat**0.14

    # (the power index 0.14 in the scaling factor is the best-fit value obtained from data
    # and used in the t96_01 version)

    # values of the magnetopause parameters for  pd = 2 npa:
    a0 = 34.586
    s00 = 1.196
    x00 = 3.4397

    # values of the magnetopause parameters, scaled by the actual pressure:

    a = a0 / rat16
    s0 = s00
    x0 = x00 / rat16
    xm = x0 - a

    # (xm is the x-coordinate of the "seam" between the ellipsoid and the cylinder)

    # (for details on the ellipsoidal coordinates, see the paper:
    # n.a.tsyganenko, solution of chapman-ferraro problem for an
    # ellipsoidal magnetopause, planet.space sci., v.37, p.1037, 1989).

    if (ygsm != 0.0 or zgsm != 0.0):
        phi = np.arctan2(ygsm, zgsm)
    else:
        phi = 0.0

    rho = np.sqrt(ygsm**2 + zgsm**2)

    if (xgsm < xm):
        xmgnp = xgsm
        rhomgnp = a * sqrt(s0**2 - 1.0)
        ymgnp = rhomgnp * np.sin(phi)
        zmgnp = rhomgnp * np.cos(phi)

        dist = sqrt((xgsm - xmgnp)**2 + (ygsm - ymgnp)**2 + (zgsm - zmgnp)**2)

        id_ = 0
        if (rhomgnp > rho):
            id_ = 1
        else:
            id_ = -1

        return xmgnp, ymgnp, zmgnp, dist, id_

    xksi = (xgsm - x0) / a + 1.0
    xdzt = rho / a
    sq1 = np.sqrt((xksi + 1.0)**2 + xdzt**2)
    sq2 = np.sqrt((1.0 - xksi)**2 + xdzt**2)
    sigma = (sq1 + sq2) * .5
    tau = (sq1 - sq2) * .5

    # now calculate (x,y,z) for the closest point at the magnetopause

    xmgnp = x0 - a * (1. - s0 * tau)
    arg = (s0**2 - 1.0) * (1. - tau**2)

    if (arg < 0.0):
        arg = 0.0

    rhomgnp = a * np.sqrt(arg)
    ymgnp = rhomgnp * np.sin(phi)
    zmgnp = rhomgnp * np.cos(phi)

    # now calculate the distance between the points {xgsm,ygsm,zgsm} and {xmgnp,ymgnp,zmgnp}:
    # (in general, this is not the shortest distance d_min, but dist asymptotically tends
    # to d_min, as we are getting closer to the magnetopause):

    dist = np.sqrt((xgsm - xmgnp)**2 + (ygsm - ymgnp)**2 + (zgsm - zmgnp)**2)

    id_ = 0
    if (sigma > s0):
        # id_ = -1 means that the point lies outside the magnetosphere
        id_ = -1
    else:
        # id_ = +1 means that the point lies inside the magnetosphere
        id_ = 1

    return xmgnp, ymgnp, zmgnp, dist, id_
