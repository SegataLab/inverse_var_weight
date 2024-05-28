#!/usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
from scipy import stats as sts
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection


class generalized_meta_analysis(object):
    def __init__( self, \
            effects, \
            variances, \
            study_pvalues, \
            study_names, \
            n_controls, \
            n_cases, \
            response_var, \
            HET="PM", \
            overlap_mat=None, \
            cov="standardized_mean_diff"):

        print("Caller of meta-analysis on %s" %response_var)
        print("Heterogeneity: %s" %HET)
        print("Number studies: %i" %len(effects))
        print("Correlation matrix foreseen: ", not(overlap_mat is None))

        self.effects = np.array(effects, dtype=np.float64)
        self.n_cases = n_cases
        self.n_controls = n_controls
 
        if not any([(x is None) for x in self.n_cases]):
            self.tot_n_cases = np.sum(self.n_cases)
        if not any([(x is None) for x in self.n_controls]):
            self.tot_n_ctrs = np.sum(self.n_controls)

        self.study_names = study_names
        print("Studies: " + " ".join(self.study_names))

        self.n = len(self.study_names)
        self.n_studies = self.n
        self.variances = np.array(variances, dtype=np.float64)
        print("Variances: is the sum zero? -> ", self.variances)
        self.HET = HET
        self.response_var = response_var
        self.devs = np.sqrt(self.variances)
        self.var_covar = None

        if not any([(x is None) for x in study_pvalues]):
            self.study_pvalues = np.array(study_pvalues, dtype=np.float64)

        if overlap_mat is None:
            self.w = np.array( [(1./v) for v in self.variances], dtype=np.float64 )
            self.effects_are_iid = True

        else:
            self.overlap_mat = np.array(overlap_mat, dtype=np.float64)
            self.var_covar = np.eye( len( self.variances ) ) * self.variances
            print("Overlap of samples across studies in a two-by-two matrix: ", self.overlap_mat)

            if cov == "standardized_mean_difference":
                for ith in range(len(self.effects)):
                    for jth in range(len(self.effects)):
                        if ith != jth:
                            d_ith, d_jth = self.effects[ith], self.effects[jth]
                            n0 = self.overlap_mat[ith, jth]
                            if n0 > 0:
                                N = self.n_cases[ith] + self.n_cases[jth] + ( 2 * n0 )
                                self.var_covar[ith, jth] += ((d_ith*d_jth) / (2*( N-3 ))) + (1./n0)

            elif cov == "linsullivan":
                for ith in range(len(self.effects)):
                    for jth in range(len(self.effects)):
                        if ith != jth:
                            se_ij = self.devs[ith] * self.devs[jth]
                            n0 = self.overlap_mat[ith, jth]
                            if n0 > 0:
                                term_a = self.n_cases[ith] * self.n_cases[jth]
                                term_b = self.n_controls[ith] * self.n_controls[jth]
                                N = (self.n_controls[ith] + self.n_cases[ith]) * (self.n_controls[jth] + self.n_cases[jth])
                                self.var_covar[ith, jth] += ((n0 * np.sqrt(term_a/term_b)) / np.sqrt(N)) * (se_ij)

            elif cov == "precomputed":
                for ith in range(len(self.effects)):
                    for jth in range(len(self.effects)):
                        if ith != jth:
                            self.var_covar[ith, jth] += self.overlap_mat[ith, jth]

            else:
                raise NotImplementedError("Cov = %s is not implemented." %cov)

            print("Parameters following the correlated structure: ")
            print("Reconstructed Var-Covar matrix: ", self.var_covar)
            #### THESE TWO ARE FOR THE COVARIANCE MATRIX OF THE WEIGTHS
            self.e = np.ones(len(self.variances), dtype=np.float64)
 
            #print(self.e)
            #print(self.var_covar)
            #print(np.dot(self.e, self.var_covar))
            #print("/", np.dot(np.dot(self.e, self.var_covar), self.e))

            inv_cov_mat = np.linalg.inv(self.var_covar)

            self.w = np.dot(self.e, inv_cov_mat) / np.dot(np.dot(self.e, inv_cov_mat), self.e)
            self.effects_are_iid = False
            print("Weights: ", " ".join(list(map(str, self.w))))

        mu_bar = np.sum(a*b for a,b in zip(self.w, self.effects))/np.sum(self.w)
        self.Q = np.sum(a*b for a,b in zip(self.w, [(x - mu_bar)**2 for x in self.effects]))
        self.Qtest = 2.*(1 - sts.chi2.cdf(np.abs(self.Q), len(self.effects)-1))
        #### H = np.sqrt(self.Q/(self.n - 1))
 
        #print("variances: ", self.variances)
        #print(self.Q, " Q")
        #print(len(variances) - 1, "len var minus one")

        self.I2 = np.max([0., (self.Q-(len(self.variances)-1))/float(self.Q)])
        self.t2_PM, self.t2PM_conv = paule_mandel_tau(self.effects, self.variances)
        self.t2_DL = ((self.Q - self.n + 1) / self.scaling( self.w )) if (self.Q > (self.n-1)) else 0.
  
        if self.effects_are_iid:
            if self.HET == "PM": self.W = [(1./float(v+self.t2_PM)) for v in self.variances]
            elif self.HET.startswith("FIX"): self.W = [(1./float(v)) for v in self.variances]
            else: self.W = [(1./float(v+self.t2_DL)) for v in self.variances]
            print("Weights: ", " ".join(list(map(str, self.W))))
            self.RE = np.sum(self.W*self.effects)/float(np.sum(self.W))
            self.RE_Var = 1./float(np.sum(self.W))

        else:
            self.W = self.w
            self.RE = np.sum(self.W*self.effects)
            self.RE_Var = 0.
            for ith in range(len(self.effects)):
                for jth in range(len(self.effects)):
                    #if ith != jth: ## EITHER WE JUST SUM THE UPPER TRIANGLE

                    if ith != jth:
                        self.RE_Var += (self.W[ith] * self.W[jth] * self.var_covar[ith, jth]) ## WE DO NOT MULTIPLY BY TWO BECAUSE WE ARE ADDING DOUBLE THE TABLE
                    else:
                        self.RE_Var += (self.W[ith] * self.W[jth] * self.variances[ith])

        print("Random/Fixed Effect model main coefficient: ", self.RE)
        print("First round of computation of variance led to: ", self.RE_Var, end="\n" if self.effects_are_iid else " ")

        print("The effect variance: ", self.RE_Var)
        self.stdErr = np.sqrt(self.RE_Var)
        self.Zscore = self.RE/self.stdErr
        print("Meta-analysis Zeta score: ", self.Zscore)
        self.Pval = 2.*(1 - sts.norm.cdf(np.abs(self.Zscore)))
        print("Meta-analysis p value: ", self.Pval)
        self.conf_int = [self.RE - 1.96*self.stdErr, self.RE + 1.96*self.stdErr]
        print("\n*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*.\n")


    def tot_var(self, Effects, Weights):
        Q = np.sum(Weights * [x**2 for x in Effects]) - ((np.sum(Weights*Effects)**2)/np.sum(Weights))
        return Q

    def scaling(self, W):
        C = np.sum(W) - (np.sum([w**2 for w in W])/float(np.sum(W)))
        return C

    def tau_squared_DL(self, Q, df, C):
        return (Q-df)/float(C) if (Q>df) else 0.

    def CombinedEffect(self):
        return np.sum(self.W*self.effects)/float(np.sum(self.W))

    def pretty_one_feat_print(self):
        _,fdr = fdrcorrection(self.study_pvalues)
        pp = { \
            "effect": list(self.effects) + [self.RE],
            "se": list(self.devs) + [self.stdErr], 
            "p-val": list(self.study_pvalues) + [self.Pval], 
            "q-val": list(fdr) + [self.Pval],
            "n_ctrs": list(self.n_controls) + [self.tot_n_ctrs], 
            "n_cases": list(self.n_cases) + [self.tot_n_cases],
            "response": self.response_var, 
        }
        return pd.DataFrame(pp, index=list(self.study_names) + ["summary"])
        
    def pretty_print(self):
        NS = {}
        for eff,std,P,study in zip(self.effects, self.devs, self.study_pvalues, self.study_names):
            NS[str(study) + "_Effect"] = eff
            NS[str(study) + "_Pvalue"] = P
            NS[str(study) + "_SE"] = std

        NS["RE_Effect"] = self.RE
        NS["RE_Pvalue"] = self.Pval
        NS["RE_stdErr"] = self.stdErr
        NS["RE_conf_int"] = ";".join(list(map(str,self.conf_int)))
        NS["RE_Var"] = self.RE_Var
        NS["Zscore"] = self.Zscore

        if self.effects_are_iid: 
            NS["Tau2_DL"] = self.t2_DL
            NS["Tau2_PM"] = self.t2_PM
            NS["I2"] = self.I2
            NS["Q"] = self.Qtest
        NS = pd.DataFrame(NS, index=[self.response_var])
        return NS


class correlation_meta_analysis(object):
    def __init__(self, rhos, ers, n_studies, pvals, studies, response_name, het="PM"):
        self.HET = het
        self.responseName = response_name
        self.studies = studies
        self.study_pvalues = pvals

        self.effects = np.arctanh(np.array(rhos, dtype=np.float64)) if not ers else np.array(rhos, dtype=np.float64)

        self.n_studies = n_studies
        self.n = float(len(studies))

        if not ers:
            self.vi = np.array([(1./float(n-3)) for n in self.n_studies], dtype=np.float64)
            self.devs = np.sqrt(self.vi)
        else:
            self.devs = np.array(ers, dtype=np.float64)
            self.vi = self.devs**2.

            #self.vi = np.array([(1./float(n-1)) for n in self.n_studies], dtype=np.float64)

        self.w = np.array([(1./float(v)) for v in self.vi], dtype=np.float64)
        mu_bar = np.sum(a*b for a,b in zip(self.w, self.effects))/np.sum(self.w)
        self.Q = np.sum(a*b for a,b in zip(self.w, [(x - mu_bar)**2 for x in self.effects]))
        self.Qtest =  2.*(1 - sts.chi2.cdf(np.abs(self.Q), len(self.effects)-1))

        H = np.sqrt(self.Q/(self.n - 1))
        self.I2 = np.max([0., (self.Q-(len(self.vi)-1))/float(self.Q)])
        self.t2_PM, self.t2PM_conv = paule_mandel_tau(self.effects, self.vi)
        self.t2_DL = ((self.Q - self.n + 1) / self.scaling( self.w )) if (self.Q > (self.n-1)) else 0.

        if self.HET == "PM":
            self.W = [(1./float(v+self.t2_PM)) for v in self.vi]
        elif self.HET.startswith("FIX"):
            self.W = [(1./float(v)) for v in self.vi]
        else:
            self.W = [(1./float(v+self.t2_DL)) for v in self.vi]

        print("Weights: ", " ".join(list(map(str, self.W))))
        self.RE = np.sum(self.W*self.effects)/float(np.sum(self.W))
        self.RE_Var = 1./float(np.sum(self.W))

        print("Random/Fixed Effect model main coefficient: ", self.RE)
        print("First round of computation of variance led to: ", self.RE_Var, end="\n")

        print("The effect variance: ", self.RE_Var)
        self.stdErr = np.sqrt(self.RE_Var)
        self.Zscore = self.RE/self.stdErr
        print("Meta-analysis Zeta score: ", self.Zscore)
        self.Pval = 2.*(1 - sts.norm.cdf(np.abs(self.Zscore)))
        print("Meta-analysis p value: ", self.Pval)
        self.conf_int = [self.RE - 1.96*self.stdErr, self.RE + 1.96*self.stdErr]

        #self.result = self.nice_shape(True)

    def scaling(self, W):
        C = np.sum(W) - (np.sum([w**2 for w in W])/float(np.sum(W)))
        return C

    def tau_squared_DL(self, Q, df, C):
        return (Q-df)/float(C) if (Q>df) else 0.

    def pretty_one_feat_print(self):
        _,fdr = fdrcorrection(self.study_pvalues)
        pp = { \
            "effect": list(self.effects) + [self.RE],
            "se": list(self.devs) + [self.stdErr],
            "p-val": list(self.study_pvalues) + [self.Pval],
            "q-val": list(fdr) + [self.Pval],
            "n_s": self.n_studies + [np.sum(self.n_studies)],
            "response": self.response_var,
        }
        return pd.DataFrame(pp, index=list(self.study_names) + ["summary"])

    def pretty_print(self):
        NS = {} 
        for rho,std,P,study in zip(self.effects, self.devs, self.study_pvalues, self.studies):
            NS[study + "_Correlation"] = np.tanh(rho)
            NS[study + "_Pvalue"] = P
            NS[study + "_SE"] = np.tanh(std)

        NS["RE_Correlation"] = np.tanh(self.RE)
        NS["RE_Pvalue"] = self.Pval
        NS["RE_stdErr"] = np.tanh(self.stdErr)
        NS["RE_conf_int"] = ";".join(list(map(str, [np.tanh(c) for c in self.conf_int])))
        NS["RE_Var"] = np.tanh(self.RE_Var)
        NS["Zscore"] = self.Zscore
        NS["Tau2_DL"] = self.t2_DL
        NS["Tau2_PM"] = self.t2_PM
        NS["I2"] = self.I2
        NS["Q"] = self.Qtest
        NS = pd.DataFrame(NS, index=[self.responseName])
        return NS

## DISCLAIMER ##
## FOLLOWING CODE FOR PAULE MANDEL TAU WAS TAKEN DIRECTLY FROM statsmodels library
## I DON T OWN THIS CODE
def paule_mandel_tau(eff, var_eff, tau2_start=0, atol=1e-5, maxiter=50):
    tau2 = tau2_start
    k = eff.shape[0]
    converged = False
    for i in range(maxiter):
        w = 1 / (var_eff + tau2)
        m = w.dot(eff) / w.sum(0)
        resid_sq = (eff - m)**2
        q_w = w.dot(resid_sq)
        # estimating equation
        ee = q_w - (k - 1)
        if ee < 0:
            tau2 = 0
            converged = 0
            break
        if np.allclose(ee, 0, atol=atol):
            converged = True
            break
        # update tau2
        delta = ee / (w**2).dot(resid_sq)
        tau2 += delta
    return tau2, converged

