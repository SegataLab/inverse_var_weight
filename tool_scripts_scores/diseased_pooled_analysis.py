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

    CRC = ["FengQ_2015_in_AUT-CRC", "GuptaA_2019_in_IND-CRC", "HanniganGD_2017_in_USA-CRC", "ThomasAM_2018a_in_ITA-CRC", "ThomasAM_2018b_in_ITA-CRC", \
        "VogtmannE_2016_in_USA-CRC", "WirbelJ_2018_in_DEU-CRC", "YachidaS_2019_in_JPN-CRC", "YuJ_2015_in_CHN-CRC", "ZellerG_2014_in_FRA-CRC"]
    T2D = ["MetaCardis_2020_a_in_FRA-T2D", "MetaCardis_2020_a_in_DEU-T2D", "KarlssonFH_2013_in_SWE-T2D", "QinJ_2012_in_CHN-T2D", "SankaranarayananK_2015_in_USA-T2D", "XuQ_2021_in_CHN-T2D"]
    IGT = ["MetaCardis_2020_a_in_FRA-IGT", "MetaCardis_2020_a_in_DEU-IGT", "KarlssonFH_2013_in_SWE-IGT"]
    IBD = ["NielsenHB_2014_in_ESP-UC", "NielsenHB_2014_in_ESP-CD", "HeQ_2017_in_CHN-CD"]
    ACVD = ["MetaCardis_2020_a_in_FRA-CAD", "MetaCardis_2020_a_in_FRA-HF", "JieZ_2017_in_CHN-ACVD"]

    def __init__(self, top=True, cardio=True, counts=True, crude=False):
        the_biggest_ever = []
        top_bot = "TOP" if top else "BOT"
        cardio_diet = "Cardio" if cardio else "Diet"
        counts_or_not = "counts" if counts else "cumul"
        type_ = "ORDINARY" if not crude else "CRUDE" 

        if counts_or_not == "counts":
            cov_met = "linsullivan"
        else:
            cov_met = "standardized_mean_difference"
 
        ma = "../input_tables_scores/aggregated_%s_table_of_meta_analysis_on_OUT_hier_%s_on-%s_%s.tsv" %(top_bot, type_, cardio_diet, counts_or_not)

        OUTFILE = lambda name : "latest_analyses/%s_%s_table_of_meta_analysis_on_OUT_%s_%s_%s_final_proposal.tsv" %(name, top_bot, cardio_diet, counts_or_not, type_)
        self.ma = pd.read_csv(ma, sep="\t", header=0, index_col=0, low_memory=False)

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
            self.ma.loc[self.CRC, "p-val"].values.astype(float), self.CRC, \
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


        self.all_analyses = [self.ma_CRC, self.ma_IGT, self.ma_T2D, self.ma_IBD, self.ma_ACVD] #, self.ma_OTHER]
        self.analyses_by_name = ["CRC", "IGT", "T2D", "IBD", "ACVD"] #, "others"]
        
        cad_mc_ctr = float(self.ma.loc["MetaCardis_2020_a_in_FRA-CAD", "nctrs"])
        ctr_mc_fran = self.ma.loc["MetaCardis_2020_a_in_FRA-IGT", "nctrs"]
        ctr_mc_deut = self.ma.loc["MetaCardis_2020_a_in_DEU-IGT", "nctrs"]
        ctr_karls = self.ma.loc["KarlssonFH_2013_in_SWE-T2D", "nctrs"]

        # ****************************************

        self.random_stage = GMA( \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "effect"].astype(float), \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "std_err"].values.astype(float) ** 2., \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "p-val"].astype(float), \
            self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "nctrs"].tolist(), \
            self.ma.loc[self.T2D + self.IGT + self.ACVD + self.CRC + self.IBD, "ncases"].tolist(), \
            "Pooled analysis of diseases", "PM", cov=cov_met )

        self.second_stage = GMA( \
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
            cov=cov_met )

        for name,anl in zip( self.analyses_by_name, self.all_analyses ):
            outf = OUTFILE(name)
            pp = anl.pretty_one_feat_print()
            pp.to_csv(outf, sep="\t", header=True, index=True)
            pp.rename(index={"summary": pp["response"].tolist()[0] + " summary"}, inplace=True)

            if not len(the_biggest_ever):
                the_biggest_ever = pp
            else:
                the_biggest_ever = pd.concat([the_biggest_ever, pp])

        self.second_stage.pretty_one_feat_print().to_csv(OUTFILE("Overall"), sep="\t", header=True, index=True)
        self.OUTFILE = OUTFILE
        print(OUTFILE("Overall"))
        pp = self.second_stage.pretty_one_feat_print()
        the_biggest_ever = pd.concat([the_biggest_ever, pp.loc["summary"].to_frame().T])


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
        the_biggest_ever.to_csv(OUTFILE("Overall"), sep="\t", header=True, index=True)
 





def main():
    paa = pooled_analyses(top=True, cardio=True, counts=True)
    pab = pooled_analyses(top=False, cardio=True, counts=True) 
    pac = pooled_analyses(top=True, cardio=False, counts=True)
    pad = pooled_analyses(top=False, cardio=False, counts=True)

    pae = pooled_analyses(top=True, cardio=True, counts=False)
    paf = pooled_analyses(top=False, cardio=True, counts=False)
    pag = pooled_analyses(top=True, cardio=False, counts=False)
    pah = pooled_analyses(top=False, cardio=False, counts=False)

if __name__ == "__main__":
    main()
