# act likelihood, ported 11/6/2016 Zack Li
# original fortran by E. Calabrese, J. Dunkley 2016

import os, sys
import numpy as np
from scipy.io import FortranFile  # need this to read the fortran data format
from scipy import linalg  # need this for cholesky decomposition and inverse
import pkg_resources

# NOTE: all bin indices are i-1 if they are i in the fortran likelihood, like b0 and bmin


class ACTPowerSpectrumLikelihood:
    def __init__(
        self,
        print_version=False,  # whether we print out stuff when initializing
        use_tt=True,
        use_te=True,
        use_ee=True,
        tt_lmax=5000,
        bmin=0,  # for ACTPol only or ACTPol+WMAP, 24 for ACTPol+Planck
        b0=5,  # first bin in TT theory selection
        nbin=260,  # total bins
        nbinw=130,  # total bins in single patch
        nbintt=40,
        nbinte=45,
        nbinee=45,
        lmax_win=7925,  # total ell in window functions
        bmax_win=520,  # total bins in window functions
        bmax=52,  # total bins in windows for one spec
    ):

        # set up all the config variables
        self.data_dir = pkg_resources.resource_filename("pyactlike", "data/")
        self.use_tt = use_tt
        self.use_te = use_te
        self.use_ee = use_ee
        self.tt_lmax = tt_lmax
        self.nbin = nbin
        self.b0 = b0
        self.nbinw = nbinw
        self.nbintt = nbintt
        self.nbinte = nbinte
        self.nbinee = nbinee
        self.lmax_win = lmax_win
        self.bmax_win = bmax_win
        self.bmax = bmax

        self.version = "ACTPollite dr4"
        if print_version:
            print("Initializing ACTPol likelihood, version", self.version)

        # set up the data file names
        like_file = os.path.join(self.data_dir, "cl_cmb_ap.dat")
        cov_file = os.path.join(self.data_dir, "c_matrix_ap.dat")
        bbldeep_file = os.path.join(self.data_dir, "coadd_bpwf_15mJy_191127_lmin2.npz")
        bblwide_file = os.path.join(self.data_dir, "coadd_bpwf_100mJy_191127_lmin2.npz")

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

        # covmat selection
        if (use_tt) and (not use_te) and (not use_ee):  # TT only
            bin_no = nbintt + nbintt
            subcov = np.zeros((bin_no, bin_no))
            subcov[:nbintt, :nbintt] = cov[:nbintt, :nbintt]
            subcov[nbintt:bin_no, nbintt:bin_no] = cov[
                nbinw : nbinw + nbintt, nbinw : nbinw + nbintt
            ]
            subcov[:nbintt, nbintt:bin_no] = cov[:nbintt, nbinw : nbinw + nbintt]
            subcov[nbintt:bin_no, :nbintt] = cov[nbinw : nbinw + nbintt, :nbintt]
        elif (not use_tt) and (use_te) and (not use_ee):  # TE only
            bin_no = nbinte + nbinte
            subcov = np.zeros((bin_no, bin_no))
            subcov[:nbinte, :nbinte] = cov[
                nbintt : nbintt + nbinte, nbintt : nbintt + nbinte
            ]
            subcov[nbinte:bin_no, nbinte:bin_no] = cov[
                nbinw + nbintt : nbinw + nbintt + nbinte,
                nbinw + nbintt : nbinw + nbintt + nbinte,
            ]  # wide
            subcov[:nbinte, nbinte:bin_no] = cov[
                nbintt : nbintt + nbinte, nbinw + nbintt : nbinw + nbintt + nbinte,
            ]  # offdiag
            subcov[nbinte:bin_no, :nbinte] = cov[
                nbinw + nbintt : nbinw + nbintt + nbinte, nbintt : nbintt + nbinte,
            ]  # offdiag
        elif (not use_tt) and (not use_te) and (use_ee):  # EE only
            bin_no = nbinee + nbinee
            subcov = np.zeros((bin_no, bin_no))
            subcov[:nbinee, :nbinee] = cov[
                nbintt + nbinte : nbintt + nbinte + nbinee,
                nbintt + nbinte : nbintt + nbinte + nbinee,
            ]  # deep
            subcov[nbinee:bin_no, nbinee:bin_no] = cov[
                nbinw + nbintt + nbinte : nbinw + nbinw,
                nbinw + nbintt + nbinte : nbinw + nbinw,
            ]  # wide
            subcov[:nbinee, nbinee:bin_no] = cov[
                nbintt + nbinte : nbinw, nbinw + nbintt + nbinte : nbinw + nbinw
            ]  # offdiag
            subcov[nbinee:bin_no, :nbinee] = cov[
                nbinw + nbintt + nbinte : nbinw + nbinw, nbintt + nbinte : nbinw
            ]  # offdiag
        elif use_tt and use_te and use_ee:  # all
            bin_no = nbin
            subcov = cov
        else:
            raise ValueError(
                "Options are: (TT+TE+EE), (only TT), (only TE), or (only EE)"
            )

        self.bin_no = bin_no
        self.fisher = linalg.cho_solve(linalg.cho_factor(subcov), b=np.identity(bin_no))

        # read window functions
        try:
            bbldeep = np.load(bbldeep_file)["bpwf"]
            self.win_func_d = np.zeros((bmax_win, lmax_win))
            self.win_func_d[:bmax_win, 1:lmax_win] = bbldeep[:bmax_win, :lmax_win]
        except IOError:
            print("Couldn't load file", bbldeep_file)
            sys.exit()

        try:
            bblwide = np.load(bblwide_file)["bpwf"]
            self.win_func_w = np.zeros((bmax_win, lmax_win))
            self.win_func_w[:bmax_win, 1:lmax_win] = bblwide[:bmax_win, :lmax_win]
        except IOError:
            print("Couldn't load file", bblwide_file)
            sys.exit()

    def loglike(self, cell_tt, cell_te, cell_ee, yp2):
        """
	    Pass in the cell_tt, cell_te, cell_ee, and yp values, get 2 * log L out.
        """

        # ----- coding notes -----
        # python is ZERO indexed, so l = 1 corresponds to an index i = 0
        # fortran indices start at ONE
        #
        # general rule for indexing in fortran to python:
        # array(a:b, c:d) in Fortran --> array[a-1:b, c-1:d] in Python
        # all of our data starts with l = 2

        X_model = np.zeros(self.nbin)
        Y = np.zeros(self.nbin)

        l_list = np.array(range(2, self.tt_lmax + 1))

        cltt = np.zeros(self.lmax_win)
        clte = np.zeros(self.lmax_win)
        clee = np.zeros(self.lmax_win)

        # convert to regular C_l, get rid of weighting
        cltt[1 : self.tt_lmax] = cell_tt / l_list / (l_list + 1.0) * 2.0 * np.pi
        clte[1 : self.tt_lmax] = cell_te / l_list / (l_list + 1.0) * 2.0 * np.pi
        clee[1 : self.tt_lmax] = cell_ee / l_list / (l_list + 1.0) * 2.0 * np.pi

        # use 150x150 windows
        bmax, lmax_win = self.bmax, self.lmax_win
        cl_tt_d = self.win_func_d[2 * bmax : 3 * bmax, 1:lmax_win] @ cltt[1:lmax_win]
        cl_te_d = self.win_func_d[6 * bmax : 7 * bmax, 1:lmax_win] @ clte[1:lmax_win]
        cl_ee_d = self.win_func_d[9 * bmax : 10 * bmax, 1:lmax_win] @ clee[1:lmax_win]
        # use 150x150 windows
        cl_tt_w = self.win_func_w[2 * bmax : 3 * bmax, 1:lmax_win] @ cltt[1:lmax_win]
        cl_te_w = self.win_func_w[6 * bmax : 7 * bmax, 1:lmax_win] @ clte[1:lmax_win]
        cl_ee_w = self.win_func_w[9 * bmax : 10 * bmax, 1:lmax_win] @ clee[1:lmax_win]

        # Select ell range in theory
        b0, nbintt, nbinte, nbinee = self.b0, self.nbintt, self.nbinte, self.nbinee
        X_model[:nbintt] = cl_tt_d[b0 : b0 + nbintt]  # TT
        X_model[nbintt : nbintt + nbinte] = cl_te_d[:nbinte] * yp2  # TE
        X_model[nbintt + nbinte : nbintt + nbinte + nbinee] = (
            cl_ee_d[:nbinee] * yp2 * yp2
        )  # EE
        X_model[nbintt + nbinte + nbinee : 2 * nbintt + nbinte + nbinee] = cl_tt_w[
            b0 : b0 + nbintt
        ]  # TT
        X_model[2 * nbintt + nbinte + nbinee : 2 * nbintt + 2 * nbinte + nbinee] = (
            cl_te_w[:nbinte] * yp2
        )  # TE
        X_model[
            2 * nbintt + 2 * nbinte + nbinee : 2 * nbintt + 2 * nbinte + 2 * nbinee
        ] = (
            cl_ee_w[:nbinee] * yp2 * yp2
        )  # EE

        Y = self.X_data - X_model

        # covmat selection
        nbinw = self.nbinw
        nbintt, nbinte, nbinee = self.nbintt, self.nbinte, self.nbinee
        use_tt, use_te, use_ee = self.use_tt, self.use_te, self.use_ee
        diff_vec = np.zeros(self.bin_no)
        if (self.use_tt) and (not self.use_te) and (not self.use_ee):  # TT only
            diff_vec[:nbintt] = Y[:nbintt]  # deep
            diff_vec[nbintt:bin_no] = Y[self.nbinw : self.nbinw + nbintt]  # wide
        elif (not use_tt) and (use_te) and (not use_ee):  # TE only
            diff_vec[:nbinte] = Y[nbintt : nbintt + nbinte]  # deep
            diff_vec[nbinte:bin_no] = Y[
                nbinw + nbintt : nbinw + nbintt + nbinte
            ]  # wide
        elif (not use_tt) and (not use_te) and (use_ee):  # EE only
            diff_vec[:nbinee] = Y[nbintt + nbinte : nbintt + nbinte + nbinee]  # deep
            diff_vec[nbinee:bin_no] = Y[
                nbinw + nbintt + nbinte : nbinw + nbintt + nbinte + nbinee
            ]  # wide
        elif use_tt and use_te and use_ee:  # all
            diff_vec = Y
        else:
            raise ValueError(
                "Options are: (TT+TE+EE), (only TT), (only TE), or (only EE)"
            )

        ptemp = np.dot(self.fisher, diff_vec)
        like = np.sum(ptemp * diff_vec) / 2.0

        return like
