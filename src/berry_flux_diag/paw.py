"""PAW reconstruction of VASP wavefunctions, without pawpyseed.

VASPParser needs each occupied state with the PAW augmentation added back,
expressed on a single G grid shared by every k-point:

    C(b,k,G) = C~(b,k,G)
             + 4 pi / sqrt(V) * sum_{s,n,l,m} <p_snlm|psi~_bk>  i^l  e^{-2 pi i G.tau_s}
                                 * F_snl(|k+G|) * Y_lm(-(k+G))

where C~ are the WAVECAR (pseudo) coefficients, p the PAW projectors, tau_s
the site positions, and F the spherical Bessel transform of the partial-wave
difference phi_ae - phi_ps. This used to come from pawpyseed's
MomentumMatrix.get_reciprocal_fullfw, whose C core needs the Intel MKL and
has not been maintained since 2021.

This module is a NumPy port of exactly that call path and nothing else.
The C function each piece reproduces is named in its docstring:

    PWFPointer(wavecar, vr)          read_wavecar, setup                 (reader.c)
    Wavefunction(..., True)          get_projector_list -> _element_setup (projector.c)
                                       spline_coeff, spherical_bessel_transform_setup,
                                       wave_spherical_bessel_transform  (utils.c, sbt.c)
                                     setup_projections  -> _site_setup, _projections
                                       setup_site, proj_value, min_cart_path, Ylm,
                                       legendre, fft3d, onto_projector_helper
    MomentumMatrix(wf, encut)        momentum_grid_size, get_momentum_grid (momentum.c)
    get_reciprocal_fullfw(b, k, 0)   fullwf_reciprocal, kwave_value, wave_interpolate

It was checked against pawpyseed's own C, compiled against a stand-in for
the few MKL calls it makes (FFTs, one zdotc). On BaTiO3 (6x6x6 mesh, 20
occupied bands, 4833 G vectors) the momentum grid is identical and every
coefficient agrees to 1.4e-14 relative. The tolerance is rounding error
from summing in a different order; the WAVECAR itself is single precision.

Not ported, because nothing on this path reads their output: pawpyseed's
smooth partial waves and partial-wave overlap matrices (get_projector_list),
get_all_transforms / grid_bounds / list_to_grid_map (MomentumMatrix), and
symmetry expansion. Projections are computed only for the bands asked for,
where pawpyseed computes them for every band in the WAVECAR. That does not
change the numbers for the bands that are asked for.

It is also much faster than the C it replaces: about 6 s per BaTiO3
WAVECAR against pawpyseed's 115 s. pawpyseed's fullwf_reciprocal
recomputes the radial-times-Ylm factor for every band, although it depends
only on k and G. Here that factor is built once per k-point, and all bands
come out of one matrix product (MomentumMatrix._aug_basis).

Spin-polarized and noncollinear WAVECARs are rejected. VASPParser's PAW
path has only ever used spin channel 0.

Pseudopotential and CoreRegion are copied verbatim from pawpyseed, and the
rest is a translation of its C. pawpyseed is distributed under the
following license, which applies to those parts:

    Copyright (c) 2017 Kyle Bystrom

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import math

import numpy as np
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Vasprun


# ------------------------------------------------- POTCAR (pawpyseed, verbatim) --

class Pseudopotential:
    """
    Contains important attributes from a VASP pseudopotential files. POTCAR
    "settings" can be read from the pymatgen POTCAR object

    If you use pymatgen, you can think of this as correlating with
    the PotcarSingle object.

    Note: for the following attributes, 'index' refers to an energy
    quantum number epsilon and angular momentum quantum number l,
    which define one set consisting of a projector function, all electron
    partial waves, and pseudo partial waves.

    Attributes:
            rmax (np.float64): Maximum radius of the projection operators
            grid (np.array): radial grid on which partial waves are defined
            aepotential (np.array): All electron potential defined radially on grid
            aecorecharge (np.array): All electron core charge defined radially
                    on grid (i.e. charge due to core, and not valence, electrons)
            kinetic (np.array): Core kinetic energy density, defined raidally on grid
            pspotential (np.array): pseudopotential defined on grid
            pscorecharge (np.array): pseudo core charge defined on grid
            ls (list): l quantum number for each index
            pswaves (list of np.array): pseudo partial waves for each index
            aewaves (list of np.array): all electron partial waves for each index
            projgrid (np.array): radial grid on which projector functions are defined
            recipprojs (list of np.array): reciprocal space projection operators
                    for each index
            realprojs (list of np.array): real space projection operators
                    for each index
    """

    def __init__(self, data):
        """
        Initializer for Pseudopotential.
        Should only be used by CoreRegion.

        Arguments:
                data (str): single-element pseudopotential
                        (POTCAR) as a string
        """
        nonradial, radial = data.split("PAW radial sets", 1)
        partial_waves = radial.split("pseudo wavefunction")
        gridstr, partial_waves = partial_waves[0], partial_waves[1:]
        self.pswaves = []
        self.aewaves = []
        self.recipprojs = []
        self.realprojs = []
        self.nonlocalprojs = []
        self.ls = []

        auguccstr, gridstr = gridstr.split("grid", 1)
        gridstr, aepotstr = gridstr.split("aepotential", 1)
        aepotstr, corechgstr = aepotstr.split("core charge-density", 1)
        try:
            corechgstr, kenstr = corechgstr.split("kinetic energy-density", 1)
            kenstr, pspotstr = kenstr.split("pspotential", 1)
        except:
            kenstr = "0 0"
            corechgstr, pspotstr = corechgstr.split("pspotential", 1)
        pspotstr, pscorechgstr = pspotstr.split("core charge-density (pseudized)", 1)
        self.grid = self.make_nums(gridstr)
        # self.aepotential = self.make_nums(aepotstr)
        # self.aecorecharge = self.make_nums(corechgstr)
        # self.kinetic = self.make_nums(kenstr)
        # self.pspotential = self.make_nums(pspotstr)
        # self.pscorecharge = self.make_nums(pscorechgstr)

        augstr, uccstr = auguccstr.split("uccopancies in atom", 1)
        head, augstr = augstr.split("augmentation charges (non sperical)", 1)
        self.augs = self.make_nums(augstr)

        for pwave in partial_waves:
            lst = pwave.split("ae wavefunction", 1)
            self.pswaves.append(self.make_nums(lst[0]))
            self.aewaves.append(self.make_nums(lst[1]))

        projstrs = nonradial.split("Non local Part")
        topstr, projstrs = projstrs[0], projstrs[1:]
        self.T = float(topstr[-22:-4])
        topstr, atpschgstr = topstr[:-22].split("atomic pseudo charge-density", 1)
        try:
            topstr, corechgstr = topstr.split("core charge-density (partial)", 1)
            settingstr, localstr = topstr.split("local part", 1)
        except:
            corechgstr = "0 0"
            settingstr, localstr = topstr.split("local part", 1)
        """
		if "gradient corrections used for XC" in localstr:
			localstr, self.gradxc = localstr.split("gradient corrections used for XC", 1)
			self.gradxc = int(self.gradxc)
		else:
			self.gradxc = None
		self.localpart = self.make_nums(localstr)
		self.localnum = self.localpart[0]
		self.localpart = self.localpart[1:]
		self.coredensity = self.make_nums(corechgstr)
		self.atomicdensity = self.make_nums(atpschgstr)
		"""

        for projstr in projstrs:
            lst = projstr.split("Reciprocal Space Part")
            nonlocalvals, projs = lst[0], lst[1:]
            self.rmax = self.make_nums(nonlocalvals.split()[2])[0]
            nonlocalvals = self.make_nums(nonlocalvals)
            l = nonlocalvals[0]
            count = nonlocalvals[1]
            self.nonlocalprojs.append(nonlocalvals[2:])
            for proj in projs:
                recipproj, realproj = proj.split("Real Space Part")
                self.recipprojs.append(self.make_nums(recipproj))
                self.realprojs.append(self.make_nums(realproj))
                self.ls.append(l)

        settingstr, projgridstr = settingstr.split("STEP   =")
        self.ndata = int(settingstr.split()[-1])
        # projgridstr = projgridstr.split("END")[0]
        self.projgrid = (
            np.arange(len(self.realprojs[0])) * self.rmax / len(self.realprojs[0])
        )
        self.step = (self.projgrid[0], self.projgrid[1])

    def make_nums(self, numstring):
        return np.fromstring(numstring, dtype=np.float64, sep=" ")


class CoreRegion:
    """
    List of Pseudopotential objects to describe the core region of a structure.

    Attributes:
            pps (dict of Pseudopotential): keys are element symbols,
                    values are Pseudopotential objects
    """

    def __init__(self, potcar):
        """
        Returns a new CoreRegion object from a
        pymatgen.io.vasp.inputs.Potcar class.

        Arguments:
                potcar (pymatgen.io.vasp.inputs.Potcar): Potcar file for
                        the VASP calculation

        Returns:
                CoreRegion object based on potcar
        """
        self.pps = {}
        for potsingle in potcar:
            self.pps[potsingle.element] = Pseudopotential(potsingle.data[:-15])


# ------------------------------------------------------------------ constants --

C = 0.262465831  # 2m/hbar^2 in 1/(eV A^2), as in pawpyseed
PI = 3.14159265358979323846


# ---------------------------------------------------------------- utils.c --

def _det(m):
    m = m.ravel()
    return (m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
            - m[2] * m[4] * m[6] - m[1] * m[3] * m[8] - m[0] * m[5] * m[7])


def _fac(n):
    return float(math.factorial(n))


def _legendre(l, m, x):
    if m < 0:
        return (-1.0) ** m * _fac(l + m) / _fac(l - m) * _legendre(l, -m, x)
    total = np.zeros_like(x)
    n = l
    while n >= 0 and 2 * n - l - m >= 0:
        total = total + (x ** (2 * n - l - m) * _fac(2 * n) / _fac(2 * n - l - m)
                         / _fac(n) / _fac(l - n) * (-1.0) ** (l - n))
        n -= 1
    return total * (-1.0) ** m * (1 - x * x) ** (m / 2.0) / 2.0 ** l


def _ylm(l, m, theta, phi):
    return (math.sqrt((2 * l + 1) / (4 * PI) * _fac(l - m) / _fac(l + m))
            * _legendre(l, m, np.cos(theta)) * np.exp(1j * m * phi))


def _angles(v, r):
    """theta, phi of vectors v (..., 3) with norms r, as utils.c computes them.

    Points with r == 0 get theta = phi = 0, which is what the r == 0 branches
    of proj_value_helper and kwave_value evaluate Ylm at.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.where(r == 0, 0.0, np.arccos(v[..., 2] / r))
        rxy = (v[..., 0] ** 2 + v[..., 1] ** 2) ** 0.5
        phi = np.where(r - np.abs(v[..., 2]) == 0, 0.0, np.arccos(v[..., 0] / rxy))
    phi = np.where(v[..., 1] < 0, 2 * PI - phi, phi)
    phi = np.where(r == 0, 0.0, phi)
    return theta, phi


def spline_coeff(x, y):
    """utils.c spline_coeff (adapted there from VASP): returns (3, N)."""
    N = len(x)
    c0, c1, c2 = np.zeros(N), np.zeros(N), np.zeros(N)
    d1p1 = (y[1] - y[0]) / (x[1] - x[0])
    if d1p1 > 0.99e30:
        c1[0] = c0[0] = 0
    else:
        c1[0] = -0.5
        c0[0] = (3 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - d1p1)
    for i in range(1, N - 1):
        s = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        r = s * c1[i - 1] + 2
        c1[i] = (s - 1) / r
        c0[i] = (6 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                      - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
                 / (x[i + 1] - x[i - 1]) - s * c0[i - 1]) / r
    c0[N - 1] = c1[N - 1] = c2[N - 1] = 0
    for i in range(N - 2, -1, -1):
        c1[i] = c1[i] * c1[i + 1] + c0[i]
    for i in range(N - 1):
        s = x[i + 1] - x[i]
        r = (c1[i + 1] - c1[i]) / 6
        c2[i] = r / s
        c1[i] = c1[i] / 2
        c0[i] = (y[i + 1] - y[i]) / s - (c1[i] + r) * s
    return np.array([c0, c1, c2])


def _cubic(f, s, ind, rem):
    return f[ind] + rem * (s[0][ind] + rem * (s[1][ind] + rem * s[2][ind]))


def proj_interpolate(r, rmax, x, f, s):
    size = len(x)
    ind = np.minimum((r / rmax * size).astype(np.int64), size - 2)
    val = _cubic(f, s, ind, r - x[ind])
    val = np.where(r < x[0], f[0], val)
    return np.where(r > x[size - 1], 0.0, val)


def wave_interpolate(r, x, f, s):
    size = len(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.log(r / x[0]) / np.log(x[1] / x[0])
    raw = np.where(r < x[0], 0, raw)  # those points take f[0] below
    ind = np.minimum(raw.astype(np.int64), size - 2)
    val = _cubic(f, s, ind, r - x[ind])
    val = np.where(r < x[0], f[0], val)
    return np.where(r > x[size - 1], 0.0, val)


def _cround(x):
    """C round(): half away from zero (np.round is half to even)."""
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5)).astype(np.int64)


# ---------------------------------------------------------------- sbt.c ----

def sbt_setup(encut, enbuf, lmax, N, r):
    """spherical_bessel_transform_setup (Talman's NumSBT). Returns (desc, kgrid)."""
    N = 2 * N
    lmax = max(lmax, 1)
    drho = math.log(r[1] / r[0])
    dt = 2 * PI / N / drho
    rmin = r[0]
    kmin = math.sqrt((encut + enbuf) * C) * math.exp(-(N // 2 - 1) * drho)
    kappamin = math.log(kmin)
    i = np.arange(N)
    ks = kmin * np.exp(i * drho)
    rs = rmin * np.exp((i - N // 2) * drho)
    rhomin = math.log(rs[0])
    t = i * dt
    rad = (10.5 * 10.5 + t * t) ** 0.5
    phi3 = (kappamin + rhomin) * t
    phi = np.arctan((2 * t) / 21)
    phi1 = (-10 * phi - t * np.log(rad) + t + np.sin(phi) / (12 * rad)
            - np.sin(3 * phi) / (360 * rad ** 3) + np.sin(5 * phi) / (1260 * rad ** 5)
            - np.sin(7 * phi) / (1680 * rad ** 7))
    for j in range(1, 11):
        phi1 = phi1 + np.arctan((2 * t) / (2 * j - 1))
    phi2 = -np.arctan(np.tanh(PI * t / 2))
    phi = phi1 + phi2 + phi3
    M = [None] * (lmax + 1)
    M[0] = (PI / 2) ** 0.5 * np.exp(1j * phi) / N
    M[0][0] *= 0.5
    M[1] = np.exp(2j * (-phi2 - np.arctan(2 * t))) * M[0]
    for l in range(1, lmax):
        M[l + 1] = np.exp(2j * -np.arctan(2 * t / (2 * l + 1))) * M[l - 1]
    return {"N": N, "M": M, "ks": ks, "rs": rs}, ks[: N // 2].copy()


def wave_sbt(d, f, l):
    """wave_spherical_bessel_transform. MKL backward DFT == N * numpy ifft."""
    N, M, ks, r = d["N"], d["M"], d["ks"], d["rs"]
    fs = np.empty(N)
    fs[: N // 2] = f[0] / r[N // 2] ** (l + 1) * r[: N // 2] ** (l + 1)
    fs[N // 2:] = f
    x = N * np.fft.ifft(r ** 0.5 * fs)
    x = x * M[l]
    x[N // 2:] = 0
    x = N * np.fft.ifft(x)
    return x[: N // 2].real * 2 / ks[: N // 2] ** 1.5


# ------------------------------------------------------------- reader.c ----

def _setup(encut, lattice):
    """reader.c setup: reciprocal lattice and the G-sphere half-extents."""
    L = lattice
    b = np.array([np.cross(L[1], L[2]), np.cross(L[2], L[0]), np.cross(L[0], L[1])])
    b = b * (2.0 * PI / _det(L))
    m = np.linalg.norm(b, axis=1)
    s = (encut * C) ** 0.5

    def sinphi(top, a, bb, mt):
        v = np.cross(a, bb)
        return np.dot(top, v) / (np.linalg.norm(v) * mt)

    phi12 = math.acos(np.dot(b[0], b[1]) / (m[0] * m[1]))
    sp = sinphi(b[2], b[0], b[1], m[2])
    A = (s / (m[0] * abs(math.sin(phi12))) + 1, s / (m[1] * abs(math.sin(phi12))) + 1,
         s / (m[2] * abs(sp)) + 1)
    phi13 = math.acos(np.dot(b[0], b[2]) / (m[0] * m[2]))
    sp = sinphi(b[1], b[0], b[2], m[1])
    B = (s / (m[0] * abs(math.sin(phi13))) + 1, s / (m[1] * abs(sp)) + 1,
         s / (m[2] * abs(math.sin(phi13))) + 1)
    phi23 = math.acos(np.dot(b[2], b[1]) / (m[2] * m[1]))
    sp = sinphi(b[0], b[1], b[2], m[0])
    Cc = (s / (m[0] * abs(sp)) + 1, s / (m[1] * abs(math.sin(phi23))) + 1,
          s / (m[2] * abs(math.sin(phi23))) + 1)
    return b, tuple(max(A[i], B[i], Cc[i]) for i in range(3))


def _g_sphere(k, nbmax, b, encut):
    """The ig3/ig2/ig1 loop shared by read_wavecar and get_momentum_grid."""
    axes = []
    for nb in nbmax:
        ig = np.arange(0, int(math.floor(2 * nb)) + 1)
        igp = np.where(ig > nb, np.trunc(ig - 2 * nb - 1), ig).astype(np.int64)
        axes.append(igp)
    g3, g2, g1 = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    G = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
    sumkg = ((k[0] + G[:, 0, None]) * b[0] + (k[1] + G[:, 1, None]) * b[1]
             + (k[2] + G[:, 2, None]) * b[2])
    etot = (np.sqrt(np.einsum("ij,ij->i", sumkg, sumkg))) ** 2.0 / C
    return G[etot <= encut].astype(np.int32)


def read_wavecar(filename, nbands=None):
    """read_wavecar: pawpyseed's own G ordering and G_bounds, single precision C."""
    raw = np.memmap(filename, dtype=np.float64, mode="r")
    nrecl, nspin, nprec = (int(round(v)) for v in raw[:3])
    if nprec == 45210:
        raise ValueError("double-precision WAVECAR: pawpyseed reads these as single")
    rec = nrecl // 8
    nwk, nband = int(round(raw[rec])), int(round(raw[rec + 1]))
    encut = float(raw[rec + 2])
    lattice = np.array(raw[rec + 3: rec + 12]).reshape(3, 3)
    b, nbmax = _setup(encut, lattice)
    nb = nband if nbands is None else nbands
    cplx = np.memmap(filename, dtype=np.complex64, mode="r")
    kpts, Gs, Cs, G_bounds = [], [], [], np.zeros(6, dtype=np.int64)
    for iwk in range(nwk * nspin):
        irec = iwk * (1 + nband)
        hdr = raw[(irec + 2) * rec:(irec + 3) * rec]
        nplane = int(round(hdr[0]))
        k = np.array(hdr[1:4])
        G = _g_sphere(k, nbmax, b, encut)
        if len(G) != nplane:
            raise ValueError(f"k-point {iwk}: {len(G)} G vectors for {nplane} plane "
                             "waves (noncollinear or gamma-only WAVECAR?)")
        G_bounds[0::2] = np.minimum(G_bounds[0::2], G.min(axis=0))
        G_bounds[1::2] = np.maximum(G_bounds[1::2], G.max(axis=0))
        start = (irec + 3) * rec
        Cs.append(np.array([cplx[start + i * rec: start + i * rec + nplane]
                            for i in range(nb)]))
        kpts.append(k)
        Gs.append(G)
    return dict(nspin=nspin, nwk=nwk, nband=nband, encut=encut, lattice=lattice,
                reclattice=b, kpts=kpts, Gs=Gs, Cs=Cs, G_bounds=G_bounds)


# ---------------------------------------------------------- projector.c ----

def _element_setup(pp):
    """The fields of get_projector_list that fullwf_reciprocal and setup_site read."""
    if any(len(p) != pp.ndata for p in pp.realprojs):
        raise ValueError("projector length differs from POTCAR NDATA")
    wave_grid = pp.grid
    proj_grid = pp.rmax / pp.ndata * np.arange(pp.ndata)
    ls = [int(l) for l in pp.ls]
    funcs = []
    for i, l in enumerate(ls):
        proj = pp.realprojs[i]
        funcs.append({"l": l, "proj": proj, "proj_spline": spline_coeff(proj_grid, proj),
                      "diffwave": pp.aewaves[i] - pp.pswaves[i]})
    d, kgrid = sbt_setup(1e7, 0, max(ls + [0]), len(wave_grid), wave_grid)
    for f in funcs:
        f["kwave"] = wave_sbt(d, f["diffwave"], f["l"])
        f["kwave_spline"] = spline_coeff(kgrid, f["kwave"])
    lm = [(n, f["l"], m) for n, f in enumerate(funcs) for m in range(-f["l"], f["l"] + 1)]
    return {"rmax": pp.rmax, "proj_grid": proj_grid, "kwave_grid": kgrid,
            "funcs": funcs, "lm": lm}


def _site_setup(coord, el, lattice, fftg):
    """setup_site with pr0_pw1 = 0: real-space projector values around one site."""
    vol = _det(lattice)
    rmax = el["rmax"]
    grid = [int(np.linalg.norm(np.cross(lattice[a], lattice[b])) * rmax / vol * fftg[i]) + 1
            for i, (a, b) in enumerate([(1, 2), (0, 2), (0, 1)])]
    center = _cround(coord * fftg)
    size = len(el["proj_grid"])
    R0 = (size - 1) * rmax / size
    rng = [np.arange(-grid[i] + center[i], grid[i] + center[i] + 1) for i in range(3)]
    I, J, K = np.meshgrid(*rng, indexing="ij")
    ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1)
    test = (ijk / fftg - coord) @ lattice
    keep = np.sqrt(np.einsum("ij,ij->i", test, test)) < R0
    ijk, test = ijk[keep], test[keep]
    w = ijk % fftg
    index = w[:, 0] * fftg[1] * fftg[2] + w[:, 1] * fftg[2] + w[:, 2]
    # min_cart_path: first strictly-shortest of the 27 images, in i, j, k order
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    cand = ((w / fftg)[:, None, :] + shifts[None] - coord) @ lattice
    dist = np.sqrt(np.einsum("ijk,ijk->ij", cand, cand))
    best = np.argmin(dist, axis=1)
    temp = cand[np.arange(len(cand)), best]
    r = dist[np.arange(len(cand)), best]
    theta, phi = _angles(temp, r)
    values = np.empty((len(el["lm"]), len(index)), dtype=np.complex128)
    for p, (n, l, m) in enumerate(el["lm"]):
        f = el["funcs"][n]
        rad = proj_interpolate(r, rmax, el["proj_grid"], f["proj"], f["proj_spline"])
        values[p] = rad * _ylm(l, m, theta, phi)
    return {"index": index, "paths": test, "values": values}


def _projections(wav, sites, k_idx, fftg):
    """fft3d + onto_projector_helper for every requested band at one k-point.

    Returns (nbands, sum_s total_projs_s), site-major, matching the order
    fullwf_reciprocal walks band->projections[s].overlaps.
    """
    k, G, Cs = wav["kpts"][k_idx], wav["Gs"][k_idx], wav["Cs"][k_idx]
    lattice = wav["lattice"]
    n = int(np.prod(fftg))
    grid = np.zeros((len(Cs),) + tuple(fftg), dtype=np.complex128)
    g = (G + fftg) % fftg
    grid[:, g[:, 0], g[:, 1], g[:, 2]] = Cs
    x = np.fft.ifftn(grid, axes=(1, 2, 3)) * (n * _det(lattice) ** -0.5)
    x = x.reshape(len(Cs), -1)
    dv = _det(lattice) / n
    kcart = k @ wav["reclattice"]
    out = []
    for s in sites:
        xv = x[:, s["index"]] * dv * np.exp(1j * (s["paths"] @ kcart))
        out.append(xv @ s["values"].conj().T)
    return np.concatenate(out, axis=1)


# ----------------------------------------------------------- momentum.c ----

class Wavefunction:
    """A WAVECAR with its PAW projections <p|psi~> computed.

    Stands in for pawpyseed's Wavefunction(structure, PWFPointer(wavecar, vr),
    CoreRegion(potcar), dim, symprec, setup_projectors=True), for the
    attributes VASPParser reads.

    Parameters
    ----------
    structure : pymatgen Structure
        Site positions the projectors are centred on - vasprun's
        final_structure, as pawpyseed used.
    wavecar_file : path
    cr : CoreRegion
    dim : length-3 sequence of int
        NGX, NGY, NGZ: the real-space grid the projections are taken on.
    nbands : int, optional
        Load and project only the lowest nbands bands. Default all.
    """

    @classmethod
    def from_files(cls, wavecar_file, potcar_file, vasprun_file, nbands=None):
        vr = Vasprun(str(vasprun_file))
        dim = [vr.parameters["NGX"], vr.parameters["NGY"], vr.parameters["NGZ"]]
        return cls(vr.final_structure, wavecar_file,
                   CoreRegion(Potcar.from_file(str(potcar_file))), dim, nbands)

    def __init__(self, structure, wavecar_file, cr, dim, nbands=None):
        self.structure = structure
        self.dim = np.array(dim, dtype=np.int64)
        self.wav = read_wavecar(str(wavecar_file), nbands)
        if self.wav["nspin"] != 1:
            raise NotImplementedError(
                "spin-polarized WAVECAR: the PAW reconstruction is only "
                "implemented for ISPIN = 1")
        self.nwk, self.nband, self.encut = self.wav["nwk"], self.wav["nband"], self.wav["encut"]
        self.nbands_loaded = len(self.wav["Cs"][0])

        elements = {e: _element_setup(pp) for e, pp in cr.pps.items()}
        self.site_elems = [elements[s.specie.symbol] for s in self.structure]
        self.coords = np.array([s.frac_coords for s in self.structure])
        self.sites = [_site_setup(self.coords[i], self.site_elems[i], self.wav["lattice"],
                                  self.dim) for i in range(len(self.structure))]
        self.projections = [_projections(self.wav, self.sites, k, self.dim)
                            for k in range(self.nwk)]


class MomentumMatrix:
    """Reconstructed coefficients on one G grid shared by every k-point.

    Stands in for pawpyseed's MomentumMatrix(wf, encut). The grid holds every
    G with hbar^2 |G|^2 / 2m <= encut (in eV), defaulting, as pawpyseed's did,
    to 4 x the run's ENCUT.
    """

    def __init__(self, wf, encut=None):
        if encut is None:
            encut = 4 * wf.encut
        self.wf = wf
        b, nbmax = _setup(encut, wf.wav["lattice"])
        self._grid = _g_sphere(np.zeros(3), nbmax, b, encut)

    @property
    def momentum_grid(self):
        return self._grid.copy()

    def _aug_basis(self, k_idx):
        """The band-independent factor of fullwf_reciprocal's augmentation sum,
        (numg, sum_s total_projs_s). pawpyseed recomputes this for every band."""
        wav = self.wf.wav
        G = self._grid
        gcart = -(G + wav["kpts"][k_idx]) @ wav["reclattice"]
        r = np.sqrt(np.einsum("ij,ij->i", gcart, gcart))
        theta, phi = _angles(gcart, r)
        pref = 4 * PI * _det(wav["lattice"]) ** -0.5
        cols = []
        for coord, el in zip(self.wf.coords, self.wf.site_elems):
            phase = np.exp(-2 * PI * 1j * (G @ coord))
            radial = [wave_interpolate(r, el["kwave_grid"], f["kwave"], f["kwave_spline"])
                      for f in el["funcs"]]
            for n, l, m in el["lm"]:
                cols.append(radial[n] * _ylm(l, m, theta, phi) * (1j ** l) * phase * pref)
        return np.array(cols).T

    def _pseudo(self, k_idx, bands):
        wav = self.wf.wav
        gb = wav["G_bounds"]
        f = np.array([gb[1] - gb[0] + 1, gb[3] - gb[2] + 1, gb[5] - gb[4] + 1])
        x = np.zeros((len(bands),) + tuple(f), dtype=np.complex64)
        g = (wav["Gs"][k_idx] % f + f) % f
        x[:, g[:, 0], g[:, 1], g[:, 2]] = wav["Cs"][k_idx][bands]
        G = self._grid
        inside = np.all((G >= gb[0::2]) & (G <= gb[1::2]), axis=1)
        gi = (G[inside] + f) % f
        out = np.zeros((len(bands), len(G)), dtype=np.complex128)
        out[:, inside] = x[:, gi[:, 0], gi[:, 1], gi[:, 2]]
        return out

    def get_reciprocal_fullfw_kpoint(self, k, nbands):
        """All of get_reciprocal_fullfw(b, k, 0) for b < nbands, one matrix product."""
        bands = np.arange(nbands)
        return self._pseudo(k, bands) + self.wf.projections[k][bands] @ self._aug_basis(k).T

    def get_reciprocal_fullfw(self, b, k, s=0):
        if s != 0 or not 0 <= b < self.wf.nbands_loaded:
            raise ValueError(f"band {b}, spin {s} not loaded")
        # VASPParser asks band by band; build the whole k-point once.
        if getattr(self, "_cached_k", None) != k:
            self._cached = self.get_reciprocal_fullfw_kpoint(k, self.wf.nbands_loaded)
            self._cached_k = k
        return self._cached[b]
