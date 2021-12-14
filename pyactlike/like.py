# ACT likelihood, ported 11/6/2016 Zack Li, updated for DR4 on 4/11/2020
# original fortran by E. Calabrese, J. Dunkley 2016

import os
import sys
from typing import Optional, Sequence

import numpy as np
import pkg_resources
from scipy.io import FortranFile  # need this to read the fortran data format

# load in cobaya class if it's there
try:
    from cobaya.likelihood import Likelihood
except:

    class Likelihood:  # dummy class to inherit if cobaya is missing
        pass


# NOTE: all bin indices are i-1 if they are i in the fortran likelihood, like b0


class ACTPowerSpectrumData:
    data_dir = pkg_resources.resource_filename("pyactlike", "data/")

    def __init__(
        self,
        print_version=False,  # whether we print out stuff when initializing
        use_tt=True,
        use_te=True,
        use_ee=True,
        use_wide=True,
        use_deep=True,
        lmax=5000,
        bmin=0,  # 0 for ACTPol only or ACTPol+WMAP, 24 for ACTPol+Planck
        nbintt=40,
        nbinte=45,
        nbinee=45,
        b0=5,
    ):

        # set up all the config variables
        self.use_tt = use_tt
        self.use_te = use_te
        self.use_ee = use_ee
        self.use_deep = use_deep
        self.use_wide = use_wide
        self.lmax = lmax
        print("Use wide", self.use_wide)
        print("Use deep", self.use_deep)

        self.b0 = b0  # first bin in TT theory selection
        self.nbintt = nbintt
        self.nbinte = nbinte
        self.nbinee = nbinee
        self.nbinw = nbintt + nbinte + nbinee  # total bins in single patch
        self.nbin = 2 * self.nbinw  # total bins
        self.lmax_win = 7925  # total ell in window functions
        self.bmax_win = 520  # total bins in window functions
        self.bmax = 52  # total bins in windows for one spec

        self.version = "ACTPollite dr4 v4"
        if print_version:
            print("Initializing ACTPol likelihood, version", self.version)

        # set up the data file names
        like_file = os.path.join(self.data_dir, "cl_cmb_ap.dat")
        cov_file = os.path.join(self.data_dir, "c_matrix_ap.dat")

        # like_file loading, contains bandpowers
        try:
            self.bval, self.X_data, self.X_sig = np.genfromtxt(
                like_file, max_rows=self.nbin, delimiter=None, unpack=True
            )
        except IOError:
            print("Couldn't load file", like_file)
            sys.exit()

        # cov_file loading
        try:
            f = FortranFile(cov_file, "r")
            cov = f.read_reals(dtype=float).reshape((self.nbin, self.nbin))
            for i_index in range(self.nbin):
                for j_index in range(i_index, self.nbin):
                    cov[i_index, j_index] = cov[j_index, i_index]
            # important: arrays in Fortran are 1-indexed,
            # but arrays in Python are 0-indexed. :(
        except IOError:
            print("Couldn't load file", cov_file)
            sys.exit()

        # cull lmin in TT
        if bmin > 0:
            nbinw = self.nbinw
            for i in range(bmin):
                cov[i, :nbintt] = 0.0  # deep
                cov[:nbintt, i] = 0.0  # deep
                cov[i, i] = 1e10  # deep
                cov[nbinw + i, nbinw : nbinw + nbintt] = 0.0  # wide
                cov[nbinw : nbinw + nbintt, nbinw + i] = 0.0  # wide
                cov[nbinw + i, nbinw + i] = 1e10  # wide

        # covmat selection
        idx = np.array([], dtype=int)
        if self.use_tt:
            idx = np.concatenate([idx, np.arange(self.nbintt)])
        if self.use_te:
            idx = np.concatenate([idx, self.nbintt + np.arange(self.nbinte)])
        if self.use_ee:
            idx = np.concatenate([idx, self.nbintt + self.nbinte + np.arange(self.nbinee)])

        self.idx = np.array([], dtype=int)
        if self.use_deep:
            self.idx = np.concatenate([self.idx, idx])
        if self.use_wide:
            self.idx = np.concatenate([self.idx, self.nbinw + idx])
        print("Number of selected bins", len(self.idx))
        print("Selected bins", self.idx)
        self.X_data = self.X_data[self.idx]
        self.inv_cov = np.linalg.inv(cov[np.ix_(self.idx, self.idx)])

        # read window functions
        for field, fn in zip(
            ["deep", "wide"],
            ["coadd_bpwf_15mJy_191127_lmin2.npz", "coadd_bpwf_100mJy_191127_lmin2.npz"],
        ):
            try:
                bmax_win, lmax_win = self.bmax_win, self.lmax_win
                bbl_file = os.path.join(self.data_dir, fn)
                bbl = np.load(bbl_file)["bpwf"]
                setattr(self, f"win_func_{field[0]}", np.zeros((bmax_win, lmax_win)))
                win_func = getattr(self, f"win_func_{field[0]}")
                win_func[:bmax_win, 1:lmax_win] = bbl[:bmax_win, :lmax_win]
            except IOError:
                print("Couldn't load file", fn)
                sys.exit()

    def loglike(self, dell_tt, dell_te, dell_ee, yp2):
        """
        Pass in the dell_tt, dell_te, dell_ee, and yp values, get 2 * log L out.
        """

        # ----- coding notes -----
        # python is ZERO indexed, so l = 1 corresponds to an index i = 0
        # fortran indices start at ONE
        #
        # general rule for indexing in fortran to python:
        # array(a:b, c:d) in Fortran --> array[a-1:b, c-1:d] in Python
        # all of our data starts with l = 2

        ell = np.arange(2, self.lmax + 1)

        cltt = np.zeros(self.lmax_win)
        clte = np.zeros(self.lmax_win)
        clee = np.zeros(self.lmax_win)

        # convert to regular C_l, get rid of weighting
        cltt[1 : self.lmax] = dell_tt[: self.lmax - 1] / ell / (ell + 1.0) * 2.0 * np.pi
        clte[1 : self.lmax] = dell_te[: self.lmax - 1] / ell / (ell + 1.0) * 2.0 * np.pi
        clee[1 : self.lmax] = dell_ee[: self.lmax - 1] / ell / (ell + 1.0) * 2.0 * np.pi

        # use 150x150 windows
        bmax, lmax_win = self.bmax, self.lmax_win
        win_tt_d = self.win_func_d[2 * bmax : 3 * bmax, 1:lmax_win]
        win_te_d = self.win_func_d[6 * bmax : 7 * bmax, 1:lmax_win]
        win_ee_d = self.win_func_d[9 * bmax : 10 * bmax, 1:lmax_win]
        # use 150x150 windows
        win_tt_w = self.win_func_w[2 * bmax : 3 * bmax, 1:lmax_win]
        win_te_w = self.win_func_w[6 * bmax : 7 * bmax, 1:lmax_win]
        win_ee_w = self.win_func_w[9 * bmax : 10 * bmax, 1:lmax_win]

        # Select ell range in theory
        b0, nbintt, nbinte, nbinee = self.b0, self.nbintt, self.nbinte, self.nbinee

        X_model_d, X_model_w = [], []
        if self.use_tt:
            cl_tt_d = win_tt_d @ cltt[1:lmax_win]
            cl_tt_w = win_tt_w @ cltt[1:lmax_win]
            cl_tt_d, cl_tt_w = cl_tt_d[b0 : b0 + nbintt], cl_tt_w[b0 : b0 + nbintt]
            X_model_d = np.concatenate([X_model_d, cl_tt_d])
            X_model_w = np.concatenate([X_model_w, cl_tt_w])
        if self.use_te:
            cl_te_d = win_te_d @ clte[1:lmax_win]
            cl_te_w = win_te_w @ clte[1:lmax_win]
            cl_te_d, cl_te_w = cl_te_d[:nbinte], cl_te_w[:nbinte]
            X_model_d = np.concatenate([X_model_d, cl_te_d * yp2])
            X_model_w = np.concatenate([X_model_w, cl_te_w * yp2])
        if self.use_ee:
            cl_ee_d = win_ee_d @ clee[1:lmax_win]
            cl_ee_w = win_ee_w @ clee[1:lmax_win]
            cl_ee_d, cl_ee_w = cl_ee_d[:nbinee], cl_ee_w[:nbinee]
            X_model_d = np.concatenate([X_model_d, cl_ee_d * yp2 ** 2])
            X_model_w = np.concatenate([X_model_w, cl_ee_w * yp2 ** 2])

        # Maybe we can do less calculations before !!
        X_model = []
        if self.use_deep:
            X_model = np.concatenate([X_model, X_model_d])
        if self.use_wide:
            X_model = np.concatenate([X_model, X_model_w])
        diff_vec = self.X_data - X_model
        chi2 = diff_vec @ self.inv_cov @ diff_vec
        log_like_result = -0.5 * chi2

        return log_like_result


# cobaya interface for the ACT Likelihood
class ACTPol_lite_DR4(Likelihood):
    name: str = "ACTPol_lite_DR4"
    components: Optional[Sequence] = ["tt", "te", "ee"]
    lmax: int = 7000
    bmin: int = 0
    use_wide: bool = True
    use_deep: bool = True

    def initialize(self):
        self.components = [c.lower() for c in self.components]

        # if not (len(self.components) in (1, 3)):
        #     raise ValueError(
        #         "components can be: [tt,te,ee], or a single component of tt, te, or ee"
        #     )

        self.data = ACTPowerSpectrumData(
            use_tt=("tt" in self.components),
            use_te=("te" in self.components),
            use_ee=("ee" in self.components),
            use_wide=self.use_wide,
            use_deep=self.use_deep,
            nbintt=40,
            nbinte=45,
            nbinee=45,
            b0=5,
            bmin=self.bmin,
        )

    def get_requirements(self):
        # State requisites to the theory code
        return {"yp2": None, "Cl": {cl: self.lmax for cl in self.components}}

    def logp(self, **params_values):
        Cl = self.theory.get_Cl(ell_factor=True)
        return self.data.loglike(Cl["tt"][2:], Cl["te"][2:], Cl["ee"][2:], yp2=params_values["yp2"])


# convenience class for combining with Planck
class ACTPol_lite_DR4_for_combining_with_Planck(ACTPol_lite_DR4):
    name: str = "ACTPol_lite_DR4_for_combining_with_Planck"
    bmin: int = 24


# single channel convenience classes
class ACTPol_lite_DR4_onlyTT(ACTPol_lite_DR4):
    name: str = "ACTPol_lite_DR4_onlyTT"
    components: Optional[Sequence] = ["tt"]


class ACTPol_lite_DR4_onlyTE(ACTPol_lite_DR4):
    name: str = "ACTPol_lite_DR4_onlyTE"
    components: Optional[Sequence] = ["te"]


class ACTPol_lite_DR4_onlyEE(ACTPol_lite_DR4):
    name: str = "ACTPol_lite_DR4_onlyEE"
    components: Optional[Sequence] = ["ee"]
