#!/usr/bin/env python

import pandas as pd
import numpy as np
import os,sys
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats as sts
sys.path.insert(1, '../')
from meta_analyses import generalized_meta_analysis as GMA

#    effects, variances, study_names, n_cases, n_controls, response_var, HET="PM", overlap_mat
#    effect  std_err p-val   lenght  95% CI  nctrs   ncases  mean_ctr        mean_css

SIGN_TH=0.05

class pooled_analyses(object):
    analyses = {
        "cardio_minusone_to_one": "../input_tables_scores/cardio_minusone_to_one_cardiometabolic_and_gut_related.tsv",
        "diet_minusone_to_one": "../input_tables_scores/diet_minusone_to_one_cardiometabolic_and_gut_related.tsv",
        "cardio_minusone_to_one_weig": "../input_tables_scores/cardio_minusone_to_one_weig_cardiometabolic_and_gut_related.tsv",
        "diet_minusone_to_one_weig": "../input_tables_scores/diet_minusone_to_one_weig_cardiometabolic_and_gut_related.tsv",
        "cardio_zero_to_one_bad": "../input_tables_scores/cardio_zero_to_one_bad_cardiometabolic_and_gut_related.tsv",    
        "diet_zero_to_one_bad": "../input_tables_scores/diet_zero_to_one_bad_cardiometabolic_and_gut_related.tsv",
        "cardio_zero_to_one_bad_weig": "../input_tables_scores/cardio_zero_to_one_bad_weig_cardiometabolic_and_gut_related.tsv",
        "diet_zero_to_one_bad_weig": "../input_tables_scores/diet_zero_to_one_bad_weig_cardiometabolic_and_gut_related.tsv",
        "cardio_zero_to_one_good": "../input_tables_scores/cardio_zero_to_one_good_cardiometabolic_and_gut_related.tsv",
        "diet_zero_to_one_good": "../input_tables_scores/diet_zero_to_one_good_cardiometabolic_and_gut_related.tsv",
        "cardio_zero_to_one_good_weig": "../input_tables_scores/cardio_zero_to_one_good_weig_cardiometabolic_and_gut_related.tsv",
        "diet_zero_to_one_good_weig": "../input_tables_scores/diet_zero_to_one_good_weig_cardiometabolic_and_gut_related.tsv",
        "cardio_minusone_to_one_arcsin": "../input_tables_scores/cardio_minusone_to_one_arcsin_cardiometabolic_and_gut_related.tsv",
        "diet_minusone_to_one_arcsin": "../input_tables_scores/diet_minusone_to_one_arcsin_cardiometabolic_and_gut_related.tsv",
        "cardio_zero_to_one_bad_arcsin": "../input_tables_scores/cardio_zero_to_one_bad_arcsin_cardiometabolic_and_gut_related.tsv",
        "diet_zero_to_one_bad_arcsin": "../input_tables_scores/diet_zero_to_one_bad_arcsin_cardiometabolic_and_gut_related.tsv",
        "cardio_zero_to_one_good_arcsin": "../input_tables_scores/cardio_zero_to_one_good_arcsin_cardiometabolic_and_gut_related.tsv",
        "diet_zero_to_one_good_arcsin": "../input_tables_scores/diet_zero_to_one_good_arcsin_cardiometabolic_and_gut_related.tsv"
    }

    CRC = ["FengQ_2015_in_AUT-CRC", "GuptaA_2019_in_IND-CRC", "HanniganGD_2017_in_USA-CRC", "ThomasAM_2018a_in_ITA-CRC", "ThomasAM_2018b_in_ITA-CRC", \
        "VogtmannE_2016_in_USA-CRC", "WirbelJ_2018_in_DEU-CRC", "YachidaS_2019_in_JPN-CRC", "YuJ_2015_in_CHN-CRC", "ZellerG_2014_in_FRA-CRC"]
    T2D = ["MetaCardis_2020_a_in_FRA-T2D", "MetaCardis_2020_a_in_DEU-T2D", "KarlssonFH_2013_in_SWE-T2D", "QinJ_2012_in_CHN-T2D", "SankaranarayananK_2015_in_USA-T2D", "XuQ_2021_in_CHN-T2D"]
    IGT = ["MetaCardis_2020_a_in_FRA-IGT", "MetaCardis_2020_a_in_DEU-IGT", "KarlssonFH_2013_in_SWE-IGT"]
    IBD = ["NielsenHB_2014_in_ESP-UC", "NielsenHB_2014_in_ESP-CD", "HeQ_2017_in_CHN-CD"]
    ACVD = ["MetaCardis_2020_a_in_FRA-CAD", "MetaCardis_2020_a_in_FRA-HF", "JieZ_2017_in_CHN-ACVD"]


    def __init__(self, analysis, SMD):
        the_biggest_ever = []
        ma = self.analyses[analysis].replace("_cardiometabolic", "_SMD_cardiometabolic" if SMD else "_MND_cardiometabolic")
        cov_met = "standardized_mean_difference" if SMD else "linsullivan"
 
        #cov_met = "linsullivan"


        OUTFILE = "latest_analyses/summary_%s_%s.tsv" %(analysis, "SMD" if SMD else "MND")
        self.ma = pd.read_csv(ma, sep="\t", header=0, index_col=0, low_memory=False)
        print(self.ma)

        self.ma_ACVD = GMA( \
            self.ma.loc[self.ACVD, "effect"].values.astype(float), \
            self.ma.loc[self.ACVD, "std_err"].values.astype(float) ** 2., \
            self.ma.loc[self.ACVD, "p-val"].values.astype(float), \
            self.ACVD, \
            self.ma.loc[self.ACVD, "nctrs"].values.astype(float), \
            self.ma.loc[self.ACVD, "ncases"].values.astype(float), \
            "ACVD", "FIX", np.array(\
                [[0., self.ma.loc["MetaCardis_2020_a_in_FRA-CAD", "nctrs"], 0.],
                 [self.ma.loc["MetaCardis_2020_a_in_FRA-HF", "nctrs"], 0., 0.],
                 [0., 0., 0.]], dtype=np.float64), \
            cov=cov_met )

        self.ma_IBD = GMA( \
            self.ma.loc[self.IBD, "effect"].values.astype(float), \
            self.ma.loc[self.IBD, "std_err"].values.astype(float) ** 2., \
            self.ma.loc[self.IBD, "p-val"].values.astype(float), \
            self.IBD, \
            self.ma.loc[self.IBD, "nctrs"].values.astype(float), \
            self.ma.loc[self.IBD, "ncases"].values.astype(float), \
            "IBD", "FIX", np.array(\
                [[0, self.ma.loc["NielsenHB_2014_in_ESP-UC", "nctrs"], 0], \
                 [self.ma.loc["NielsenHB_2014_in_ESP-UC", "nctrs"], 0, 0], \
                 [0, 0, 0]], dtype=np.float64), \
            cov=cov_met )
                       
        self.ma_CRC = GMA( \
            self.ma.loc[self.CRC, "effect"].values.astype(float), \
            self.ma.loc[self.CRC, "std_err"].values.astype(float) ** 2., \
            self.ma.loc[self.CRC, "p-val"].values.astype(float), \
            self.CRC, \
            self.ma.loc[self.CRC, "nctrs"].values.astype(float), \
            self.ma.loc[self.CRC, "ncases"].values.astype(float), \
            "CRC", "PM", None, cov=cov_met )

        self.ma_T2D = GMA( \
            self.ma.loc[self.T2D, "effect"].astype(float), \
            self.ma.loc[self.T2D, "std_err"].values.astype(float) ** 2., \
            self.ma.loc[self.T2D, "p-val"].astype(float), \
            self.T2D, \
            self.ma.loc[self.T2D, "nctrs"].tolist(), \
            self.ma.loc[self.T2D, "ncases"].tolist(), \
            "T2D", "PM", cov=cov_met ) 

        self.ma_IGT = GMA( \
            self.ma.loc[self.IGT, "effect"], \
            self.ma.loc[self.IGT, "std_err"]** 2., \
            self.ma.loc[self.IGT, "p-val"], \
            self.IGT, \
            self.ma.loc[self.IGT, "nctrs"].tolist(), \
            self.ma.loc[self.IGT, "ncases"].tolist(), \
            "IGT", "PM", cov=cov_met )

        self.all_analyses = [self.ma_CRC, self.ma_IGT, self.ma_T2D, self.ma_IBD, self.ma_ACVD]
        self.analyses_by_name = ["CRC", "IGT", "T2D", "IBD", "ACVD"]
         
        cad_mc_ctr = float(self.ma.loc["MetaCardis_2020_a_in_FRA-CAD", "nctrs"])

        ctr_mc_fran = self.ma.loc["MetaCardis_2020_a_in_FRA-IGT", "nctrs"]
        ctr_mc_deut = self.ma.loc["MetaCardis_2020_a_in_DEU-IGT", "nctrs"]
        ctr_karls = self.ma.loc["KarlssonFH_2013_in_SWE-T2D", "nctrs"]

        # ****************************************

        print("Ultima")

        
        self.random_stage = GMA( \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "effect"].astype(float), \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "std_err"].values.astype(float) ** 2., \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "p-val"].astype(float), \
            self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "nctrs"].tolist(), \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "ncases"].tolist(), \
            "Pooled analysis of diseases", "PM", cov=cov_met )

        
        self.second_stage_fix = GMA( \
            [ma.RE for ma in self.all_analyses], \
            [ma.RE_Var for ma in self.all_analyses], \
            [ma.Pval for ma in self.all_analyses], \
            ["CRC", "IGT", "T2D", "IBD", "ACVD"], \
            [ma.tot_n_ctrs for ma in self.all_analyses], \
            [ma.tot_n_cases for ma in self.all_analyses], \
            "Pooled analysis of diseases", "FIX", \
            np.array([\
                [0., 0., 0., 0., 0.],
                [0., 0., ctr_mc_fran + ctr_mc_deut + ctr_karls, 0., cad_mc_ctr],
                [0., ctr_mc_fran + ctr_mc_deut + ctr_karls, 0., 0., cad_mc_ctr],
                [0., 0., 0., 0., 0.],
                [0., cad_mc_ctr, cad_mc_ctr, 0., 0.]], dtype=np.float64), \
            cov=cov_met)
        
        self.second_stage = GMA( \
            [ma.RE for ma in self.all_analyses], \
            [ma.RE_Var for ma in self.all_analyses], \
            [ma.Pval for ma in self.all_analyses], \
            ["CRC", "IGT", "T2D", "IBD", "ACVD"], \
            [ma.tot_n_ctrs for ma in self.all_analyses], \
            [ma.tot_n_cases for ma in self.all_analyses], \
            "Pooled analysis of diseases", "PM", None, cov=cov_met)

        the_biggest_ever = []
    
        for name,anl in zip( self.analyses_by_name, self.all_analyses ):
            pp = anl.pretty_one_feat_print()
            pp.rename(index={"summary": pp["response"].tolist()[0] + " summary"}, inplace=True)

            if not len(the_biggest_ever):
                the_biggest_ever = pp
            else:
                the_biggest_ever = pd.concat([the_biggest_ever, pp])

        pp = self.second_stage_fix.pretty_one_feat_print()
        pp = pp.rename(index={"summary": "covarying effect"})
        the_biggest_ever = pd.concat([the_biggest_ever, pp.loc["covarying effect"].to_frame().T])

        pw = self.random_stage.pretty_one_feat_print()
        pw = pw.rename(index={"summary": "tot.random effect"})
        the_biggest_ever = pd.concat([the_biggest_ever, pw.loc["tot.random effect"].to_frame().T])

        ppp = self.second_stage.pretty_one_feat_print()
        ppp = ppp.rename(index={"summary": "random effect"})
        the_biggest_ever = pd.concat([the_biggest_ever, ppp.loc["random effect"].to_frame().T])

        idxs = [i for i in the_biggest_ever.index if (not "effect" in i) and (not "summary" in i)] + [i for i in the_biggest_ever.index if "summary" in i] + \
            ["covarying effect", "tot.random effect", "random effect"]
        ##[i for i in the_biggest_ever.index if ("effect" in i)]
        the_biggest_ever = the_biggest_ever.loc[idxs]
        the_biggest_ever.to_csv(OUTFILE, sep="\t", header=True, index=True)





def main():
    for score in [ \
        "cardio_zero_to_one_good", "cardio_zero_to_one_bad", "cardio_minusone_to_one", \
        "cardio_zero_to_one_good_weig", "cardio_zero_to_one_bad_weig", "cardio_minusone_to_one_weig", \
        "diet_zero_to_one_good", "diet_zero_to_one_bad", "diet_minusone_to_one", \
        "diet_zero_to_one_good_weig", "diet_zero_to_one_bad_weig", "diet_minusone_to_one_weig", \
        "cardio_zero_to_one_good_arcsin", "cardio_zero_to_one_bad_arcsin", "cardio_minusone_to_one_arcsin", \
        "diet_zero_to_one_good_arcsin", "diet_zero_to_one_bad_arcsin", "diet_minusone_to_one_arcsin"]:

        p = pooled_analyses(score, True)
        ###p = pooled_analyses(score, False)


if __name__ == "__main__":
    main()
