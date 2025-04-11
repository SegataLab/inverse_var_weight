#!/urs/bin/env python

import pandas as pd
import numpy as np
import sys, os, glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats as sts
import pingouin as pg
sys.path.append("/shares/CIBIO-Storage/CM/scratch/users/paolo.manghi/metaSinC/metasinc/")
from meta_analyses import generalized_meta_analysis, correlation_meta_analysis

SIGN_TH=0.05

class Abundance(object):
    relab_BMI = pd.read_csv("../public_data_profiles/Healthy_Subject_Data_Table_Jan21.tsv", sep="\t", header=0, index_col=0, low_memory=False)
    relab_OUT = pd.read_csv("../public_data_profiles/CaseControl_Subject_Data_Table_Jan21.tsv", sep="\t", header=0, index_col=0, low_memory=False)

    relab_BMI.loc["study_identifier"] = [((dt + "_in_" + cn) if dt!="KarlssonFH_2013" else dt+"_in_SWE") for dt,cn in zip(\
      relab_BMI.loc["study_name"].tolist(), relab_BMI.loc["country"].tolist())]

    #for x in relab_BMI.loc["study_identifier"].unique(): print(x)
    #exit(1)

    relab_BMI = relab_BMI.rename(index={"gender": "sex"})

    relab_OUT.loc["study_identifier"] = [((dt + "_in_" + cn) if dt!="KarlssonFH_2013" else dt+"_in_SWE") for dt,cn in zip(\
      relab_OUT.loc["study_name"].tolist(), relab_OUT.loc["country"].tolist())]
    relab_OUT.loc["target_condition"] = [("case" if cd!="control" else "control") for cd in relab_OUT.loc["study_condition"].tolist()]
    relab_OUT = relab_OUT.rename(index={"gender": "sex"})
    relab_OUT.loc["study_condition"] = [(std if std!="IBD" else disub) for std,disub in zip(\
      relab_OUT.loc["study_condition"].tolist(), relab_OUT.loc["disease_subtype"].tolist())]

    relab_OUT = relab_OUT.rename(index=dict([(i, i.split("|")[-1][3:]) for i in relab_OUT.index if ("t__" in i)]))
    relab_BMI = relab_BMI.rename(index=dict([(i, i.split("|")[-1][3:]) for i in relab_BMI.index if ("t__" in i)]))
 
    relab_BMI.loc["BMI_class"] = [("normal" if (b<25) else ("overweight" if (b>=25 and b<30) else "obese")) for b in relab_BMI.loc["BMI"].values.astype(float)]

    spps_BMI = [i for i in relab_BMI.index if i.startswith("SGB") or i.startswith("EUK")]
    spps_OUT = [i for i in relab_OUT.index if i.startswith("SGB") or i.startswith("EUK")]

    def __init__(self):
        self.relab_BMI.loc["richness"] = [np.count_nonzero( self.relab_BMI.loc[ self.spps_BMI, s].astype(float)) for s in self.relab_BMI]
        self.relab_OUT.loc["richness"] = [np.count_nonzero( self.relab_OUT.loc[ self.spps_OUT, s].astype(float)) for s in self.relab_OUT]


class Ranks(object):
    
    if (not os.path.exists('../cardiometabolic_scores_zoe.tsv')) or (not os.path.exists('../diet_scores_zoe.tsv')):
        raise FileNotFoundError('the files cardiometabolic_scores_zoe.tsv and diet_scores_zoe.tsv msut be found in the parent directory.')
        exit(1)

    cardi0_sc0res = pd.read_csv( 'cardiometabolic_scores_zoe.tsv', sep="\t", header=0, index_col=0)["mean_atleast2"].dropna()
    diet_sc0res = pd.read_csv( 'diet_scores_zoe.tsv', sep="\t", header=0, index_col=0)["mean_atleast2"].dropna()

    cardi0_sc0res = cardi0_sc0res.sort_values()
    diet_sc0res   = diet_sc0res.sort_values()

    N = 50
 
    #relab = self.relab_OUT.copy(deep=True)

    self.relab_OUT = self.relab_OUT.rename(index=dict([(i, i.split("t__")[1]) for i in self.relab_OUT.index if ("t__" in i)]))

    self.relab_OUT.loc["count_of_good_cardio"] = np.count_nonzero( self.relab_OUT.loc[ cardi0_sc0res[ :N ].index ].values.astype(float), axis=0)
    self.relab_OUT.loc["count_of_bad_cardio"]  = np.count_nonzero( self.relab_OUT.loc[ cardi0_sc0res[ -N: ].index ].values.astype(float), axis=0)
    self.relab_OUT.loc["count_of_good_diet"]   = np.count_nonzero( self.relab_OUT.loc[ diet_sc0res[ :N ].index ].values.astype(float), axis=0)
    self.relab_OUT.loc["count_of_bad_diet"]    = np.count_nonzero( self.relab_OUT.loc[ diet_sc0res[ -N: ].index ].values.astype(float), axis=0)

    self.relab_OUT.loc["percentile_count_of_good_cardio"] = [sts.percentileofscore( self.relab_OUT.loc["count_of_good_cardio", self.relab_OUT.loc["study_identifier"]==std].values.astype(float) \
        , x)/100. for x,std in zip(self.relab_OUT.loc["count_of_good_cardio"].values.astype(float), self.relab_OUT.loc["study_identifier"].tolist())    ]
    self.relab_OUT.loc["percentile_count_of_bad_cardio"] = [sts.percentileofscore( self.relab_OUT.loc["count_of_bad_cardio", self.relab_OUT.loc["study_identifier"]==std].values.astype(float) \
        , x)/100. for x,std in zip(self.relab_OUT.loc["count_of_bad_cardio"].values.astype(float), self.relab_OUT.loc["study_identifier"].tolist())    ]
    self.relab_OUT.loc["percentile_count_of_good_diet"] = [sts.percentileofscore( self.relab_OUT.loc["count_of_good_diet", self.relab_OUT.loc["study_identifier"]==std].values.astype(float) \
        , x)/100. for x,std in zip(self.relab_OUT.loc["count_of_good_diet"].values.astype(float), self.relab_OUT.loc["study_identifier"].tolist())    ]
    self.relab_OUT.loc["percentile_count_of_bad_diet"] = [sts.percentileofscore( self.relab_OUT.loc["count_of_bad_diet", self.relab_OUT.loc["study_identifier"]==std].values.astype(float) \
        , x)/100. for x,std in zip(self.relab_OUT.loc["count_of_bad_diet"].values.astype(float), self.relab_OUT.loc["study_identifier"].tolist())    ]

    self.relab_OUT.loc["cumul_of_good_cardio"] = np.sum( self.relab_OUT.loc[ cardi0_sc0res[ :N ].index ].values.astype(float), axis=0) #/ 100.
    self.relab_OUT.loc["cumul_of_bad_cardio"]  = np.sum( self.relab_OUT.loc[ cardi0_sc0res[ -N: ].index ].values.astype(float), axis=0) #/ 100.
    self.relab_OUT.loc["cumul_of_good_diet"]   = np.sum( self.relab_OUT.loc[ diet_sc0res[ :N ].index ].values.astype(float), axis=0) #/ 100.
    self.relab_OUT.loc["cumul_of_bad_diet"]    = np.sum( self.relab_OUT.loc[ diet_sc0res[ -N: ].index ].values.astype(float), axis=0) #/ 100.

    all_cols = ["count_of_good_cardio", "count_of_bad_cardio", "count_of_good_diet", "count_of_bad_diet", "percentile_count_of_good_cardio", "percentile_count_of_bad_cardio", \
        "percentile_count_of_good_diet", "percentile_count_of_bad_diet", "cumul_of_good_cardio", "cumul_of_bad_cardio", "cumul_of_good_diet", "cumul_of_bad_diet" ]
 
    cardio_zeroone_good = ((cardi0_sc0res.sort_values() - cardi0_sc0res.max()) / (cardi0_sc0res.min() - cardi0_sc0res.max())).to_dict()
    cardio_zeroone_bad  = ((cardi0_sc0res.sort_values() - cardi0_sc0res.min()) / (cardi0_sc0res.max() - cardi0_sc0res.min())).to_dict()
    cardio_oneone       = (((1 - (-1)) / (cardi0_sc0res.max() - cardi0_sc0res.min()) * (cardi0_sc0res.sort_values() - cardi0_sc0res.max()) + 1) * -1).to_dict()

    diet_zeroone_good   = ((diet_sc0res.sort_values() - diet_sc0res.max()) / (diet_sc0res.min() - diet_sc0res.max())).to_dict()
    diet_zeroone_bad    = ((diet_sc0res.sort_values() - diet_sc0res.min()) / (diet_sc0res.max() - diet_sc0res.min())).to_dict()
    diet_oneone         = (((1 - (-1)) / (diet_sc0res.max() - diet_sc0res.min()) * (diet_sc0res.sort_values() - diet_sc0res.max()) + 1) * -1).to_dict()

    relab = self.relab_OUT.copy(deep=True)

    studies = ["FengQ_2015_in_AUT", "GuptaA_2019_in_IND", "HanniganGD_2017_in_USA", "HanniganGD_2017_in_CAN", "HeQ_2017_in_CHN", "JieZ_2017_in_CHN", "KarlssonFH_2013_in_SWE", \
        "MetaCardis_2020_a_in_FRA", "MetaCardis_2020_a_in_DEU", "NielsenHB_2014_in_DNK", "NielsenHB_2014_in_ESP", "QinJ_2012_in_CHN", "QinN_2014_in_CHN", "ThomasAM_2018a_in_ITA", \
        "ThomasAM_2018b_in_ITA", "VogtmannE_2016_in_USA", "WirbelJ_2018_in_DEU", "XuQ_2021_in_CHN", "YachidaS_2019_in_JPN", "YuJ_2015_in_CHN", "ZellerG_2014_in_FRA", \
        "SankaranarayananK_2015_in_USA"]

    def __init__(self):
        print(self.cardi0_sc0res.shape, "\n-->Number of cardio SGBs")
        print(self.diet_sc0res.shape, "\n-->Number of diet SGBs")


class ZOE_scores_fundamental_analyses(object):
    def __init__(self):
        self.starting_scores = Ranks()
        self.abs = Abundance() ## self.abs.relab_OUT

        ## BMI
        self.all_datasets_BMI = pd.read_csv('../public_data_profiles/Healthy_Subject_Data_Table_Jan21.tsv', sep="\t", header=0, index_col=0, low_memory=False)
 
        self.sgbs = [i for i in self.all_datasets_BMI.index if ("t__" in i)]
        self.sgbs_codes = [s.split("|")[-1][3:] for s in self.sgbs]
        self.sgbs_namer = dict([(i.split("|")[-1][3:], "|".join(i.split("|")[-2:])) for i in self.sgbs])
        self.all_datasets_BMI.rename(index=dict([(i, i.split("|")[-1][3:]) for i in self.sgbs]), inplace=True)
        self.all_datasets_BMI = self.all_datasets_BMI.loc[ ["age", "BMI", "gender", "country", "study_name"] + self.sgbs_codes ] 

        print(len(self.all_datasets_BMI.loc["study_name"].tolist()))
        print(len(self.all_datasets_BMI.loc["country"].tolist()))

        self.all_datasets_BMI.loc["study_identifier"] = [((dt + "_in_" + cn) if dt!="KarlssonFH_2013" else dt+"_in_SWE") for dt,cn in zip(\
            self.all_datasets_BMI.loc["study_name"].tolist(), self.all_datasets_BMI.loc["country"].tolist())]

        self.all_datasets_BMI = self.all_datasets_BMI.rename(index={"gender": "sex"})
        self.metadata = ["sex", "age", "BMI"]
        self.all_datasets_BMI.loc["sex"] = [(1. if s=="male" else 0.0) for s in self.all_datasets_BMI.loc["sex"].tolist()]

        ## OUTCOMES
        self.all_datasets_OUT = pd.read_csv('../public_data_profiles/CaseControl_Subject_Data_Table_Jan21.tsv', sep="\t", header=0, index_col=0, low_memory=False)
 
        self.OUT_sgbs = [i for i in self.all_datasets_OUT.index if ("t__" in i)]
        self.OUT_sgbs_codes = [s.split("|")[-1][3:] for s in self.OUT_sgbs]
        self.OUT_sgbs_namer = dict([(i.split("|")[-1][3:], "|".join(i.split("|")[-2:])) for i in self.OUT_sgbs])
        self.all_datasets_OUT.rename(index=dict([(i, i.split("|")[-1][3:]) for i in self.OUT_sgbs]), inplace=True)
        self.all_datasets_OUT = self.all_datasets_OUT.loc[ [ "age", "BMI", "gender", "country", "study_condition", "study_name", "disease_subtype" ] + self.OUT_sgbs_codes ]

        print(len(self.all_datasets_OUT.loc["study_name"].tolist()))
        print(len(self.all_datasets_OUT.loc["country"].tolist()))

        self.all_datasets_OUT.loc["study_identifier"] = [((dt + "_in_" + cn) if dt!="KarlssonFH_2013" else dt+"_in_SWE") for dt,cn in zip(\
          self.all_datasets_OUT.loc["study_name"].tolist(), self.all_datasets_OUT.loc["country"].tolist())]
        self.all_datasets_OUT.loc["target_condition"] = [("case" if cd!="control" else "control") for cd in \
          self.all_datasets_OUT.loc["study_condition"].tolist()]

        self.all_datasets_OUT = self.all_datasets_OUT.rename(index={"gender": "sex"})
        self.metadata_OUT = ["BMI", "target_condition", "study_condition", "sex", "age"]
        self.all_datasets_OUT.loc["sex"] = [(1. if s=="male" else 0.0) for s in self.all_datasets_OUT.loc["sex"].tolist()]
        self.all_datasets_OUT.loc["study_condition"] = [(std if std!="IBD" else disub) for std,disub in zip(\
            self.all_datasets_OUT.loc["study_condition"].tolist(), self.all_datasets_OUT.loc["disease_subtype"].tolist())]

        self.Cardio_SGBs = self.starting_scores.cardi0_sc0res.index.tolist()
        self.Diet_SGBs   = self.starting_scores.diet_sc0res.index.tolist()

    def get_a_smd(self, dataset_id, condition, SGB):
        datas = self.all_datasets_OUT.loc[ self.metadata_OUT + [SGB], \
            (self.all_datasets_OUT.loc["study_identifier"]==dataset_id) & \
            (self.all_datasets_OUT.loc["study_condition"].isin([condition, "control"])) ]
        datas.loc[SGB] = np.arcsin(np.sqrt(datas.loc[SGB].values.astype(float)/100.))
        covars = ["BMI", "sex", "age"]
 
        if datas.loc[SGB].astype(bool).astype(int).sum() >= 15:
            datast = datas.loc[self.metadata_OUT + [SGB]].T
            datast = datast.astype({"BMI": float, SGB: float, "target_condition": str, "study_condition": str, "sex": str, "age": float})
            covars = [c for c in covars if (len(datast[c].unique())>1)]
            
            #md = smf.ols('Q("%s") ~ C(target_condition, Treatment("control"))'%(SGB), data=datast)

            md = smf.ols('Q("%s") ~ C(target_condition, Treatment("control")) + BMI + C(sex) + age' %(SGB), data=datast)
            model_fit = md.fit()
            disease = condition ##"|".join(sorted([c for c in datast["study_condition"].unique().tolist() if c!="control"]))
            if 'C(target_condition, Treatment("control"))[T.case]' in model_fit.params.index.tolist():
                #print(model_fit.summary())
                t = model_fit.tvalues.loc[ 'C(target_condition, Treatment("control"))[T.case]' ]
                n1 = float(len(datast.loc[ (datast["target_condition"]=="control") ]))
                n2 = float(len(datast.loc[ (datast["target_condition"]=="case") ]))
                d = (t*(n1+n2))/float(np.sqrt(n1*n2)*np.sqrt(n1+n2-2))
                SEd = np.sqrt(((n1+n2-1)/float(n1+n2-3)) * ((4./float(n1+n2))*(1+((d**2.)/8.))))
                pval = model_fit.pvalues.loc['C(target_condition, Treatment("control"))[T.case]']
                rs = pd.DataFrame({\
                    "d": d, "se": SEd, "p-val": pval, "n_cases": n2, "n_ctrs": n1, "SGB": SGB, "study": dataset_id, "disease": disease}, index=["effect_size"])
                return rs
        return "ABSENT"

    def get_a_smd_no_BMI_correction(self, dataset_id, condition, SGB):
        datas = self.all_datasets_OUT.loc[ self.metadata_OUT + [SGB], \
            (self.all_datasets_OUT.loc["study_identifier"]==dataset_id) & \
            (self.all_datasets_OUT.loc["study_condition"].isin([condition, "control"])) ]
        datas.loc[SGB] = np.arcsin(np.sqrt(datas.loc[SGB].values.astype(float)/100.))
        covars = ["sex", "age"]

        if datas.loc[SGB].astype(bool).astype(int).sum() >= 15:
            datast = datas.loc[self.metadata_OUT + [SGB]].T
            datast = datast.astype({"BMI": float, SGB: float, "target_condition": str, "study_condition": str, "sex": str, "age": float})

            covars = [c for c in covars if (len(datast[c].unique())>1)]
            md = smf.ols('Q("%s") ~ C(target_condition, Treatment("control")) + C(sex) + age' %(SGB), data=datast)
            model_fit = md.fit()
            disease = condition ##"|".join(sorted([c for c in datast["study_condition"].unique().tolist() if c!="control"]))
            if 'C(target_condition, Treatment("control"))[T.case]' in model_fit.params.index.tolist():
                #print(model_fit.summary())
                t = model_fit.tvalues.loc[ 'C(target_condition, Treatment("control"))[T.case]' ]
                n1 = float(len(datast.loc[ (datast["target_condition"]=="control") ]))
                n2 = float(len(datast.loc[ (datast["target_condition"]=="case") ]))
                d = (t*(n1+n2))/float(np.sqrt(n1*n2)*np.sqrt(n1+n2-2))
                SEd = np.sqrt(((n1+n2-1)/float(n1+n2-3)) * ((4./float(n1+n2))*(1+((d**2.)/8.))))
                pval = model_fit.pvalues.loc['C(target_condition, Treatment("control"))[T.case]']
                rs = pd.DataFrame({\
                    "d": d, "se": SEd, "p-val": pval, "n_cases": n2, "n_ctrs": n1, "SGB": SGB, "study": dataset_id, "disease": disease}, index=["effect_size"])
                return rs
        return "ABSENT"

    def get_a_smd_Crude(self, dataset_id, condition, SGB):
        datas = self.all_datasets_OUT.loc[ self.metadata_OUT + [SGB], \
            (self.all_datasets_OUT.loc["study_identifier"]==dataset_id) & \
            (self.all_datasets_OUT.loc["study_condition"].isin([condition, "control"])) ]
        datas.loc[SGB] = np.arcsin(np.sqrt(datas.loc[SGB].values.astype(float)/100.))
        covars = [] #"sex", "age"]

        if datas.loc[SGB].astype(bool).astype(int).sum() >= 15:
            datast = datas.loc[self.metadata_OUT + [SGB]].T
            datast = datast.astype({"BMI": float, SGB: float, "target_condition": str, "study_condition": str, "sex": str, "age": float})
            #covars = [c for c in covars if (len(datast[c].unique())>1)]
            md = smf.ols('Q("%s") ~ C(target_condition, Treatment("control"))' %(SGB), data=datast)
            model_fit = md.fit()
            disease = condition ##"|".join(sorted([c for c in datast["study_condition"].unique().tolist() if c!="control"]))
            if 'C(target_condition, Treatment("control"))[T.case]' in model_fit.params.index.tolist():
                #print(model_fit.summary())
                t = model_fit.tvalues.loc[ 'C(target_condition, Treatment("control"))[T.case]' ]
                n1 = float(len(datast.loc[ (datast["target_condition"]=="control") ]))
                n2 = float(len(datast.loc[ (datast["target_condition"]=="case") ]))
                d = (t*(n1+n2))/float(np.sqrt(n1*n2)*np.sqrt(n1+n2-2))
                SEd = np.sqrt(((n1+n2-1)/float(n1+n2-3)) * ((4./float(n1+n2))*(1+((d**2.)/8.))))
                pval = model_fit.pvalues.loc['C(target_condition, Treatment("control"))[T.case]']
                rs = pd.DataFrame({\
                    "d": d, "se": SEd, "p-val": pval, "n_cases": n2, "n_ctrs": n1, "SGB": SGB, "study": dataset_id, "disease": disease}, index=["effect_size"])
                return rs
        return "ABSENT"

    def get_diff_rec_Crude(self, on_diet=False):
        OUTFILE = "complete_corre_table_for_OUTCOMES_Crude_on-Cardio.tsv" if not on_diet else "complete_corre_table_for_OUTCOMES_Crude_on-Diet.tsv"
        res_on_OUT = []
        scored_sgbs = self.Cardio_SGBs if not on_diet else self.Diet_SGBs
        for SGB in scored_sgbs:
            for dataset in self.all_datasets_OUT.loc["study_identifier"].unique():
                diseases = [c for c in self.all_datasets_OUT.loc[ "study_condition", \
                    (self.all_datasets_OUT.loc["study_identifier"]==dataset) ].unique().tolist() if c!="control" ]
                for disease in diseases:
                    cor_frame = self.get_a_smd_Crude( dataset, disease, SGB )
                    if not isinstance(cor_frame, str):
                        print(cor_frame)
                        if not len(res_on_OUT):
                            res_on_OUT = cor_frame
                        else:
                            res_on_OUT = pd.concat([res_on_OUT, cor_frame])
        _,fdr = fdrcorrection(res_on_OUT["p-val"].values.astype(float), alpha=0.05)
        res_on_OUT.insert(4, "FDR-q-val", fdr)
        res_on_OUT.to_csv(os.path.join("results/", OUTFILE), sep="\t", header=True, index=True)
 
    def get_diff_rec_NoBMI(self, on_diet=False):
        OUTFILE = "complete_corre_table_for_OUTCOMES_NoBMI_Corr_on-Cardio.tsv" if not on_diet else "complete_corre_table_for_OUTCOMES_NoBMI_Corr_on-Diet.tsv"
        res_on_OUT = []
        scored_sgbs = self.Cardio_SGBs if not on_diet else self.Diet_SGBs
        for SGB in scored_sgbs:
            for dataset in self.all_datasets_OUT.loc["study_identifier"].unique():
                diseases = [c for c in self.all_datasets_OUT.loc[ "study_condition", \
                    (self.all_datasets_OUT.loc["study_identifier"]==dataset) ].unique().tolist() if c!="control" ]
                for disease in diseases:
                    cor_frame = self.get_a_smd_no_BMI_correction( dataset, disease, SGB )
                    if not isinstance(cor_frame, str):
                        print(cor_frame)
                        if not len(res_on_OUT):
                            res_on_OUT = cor_frame
                        else:
                            res_on_OUT = pd.concat([res_on_OUT, cor_frame])
        _,fdr = fdrcorrection(res_on_OUT["p-val"].values.astype(float), alpha=0.05)
        res_on_OUT.insert(4, "FDR-q-val", fdr)
        res_on_OUT.to_csv(os.path.join("results/", OUTFILE), sep="\t", header=True, index=True)

    def get_diff_rec(self, on_diet=False):
        OUTFILE = "complete_corre_table_for_OUTCOMES_on-Cardio.tsv" if not on_diet else "complete_corre_table_for_OUTCOMES_on-Diet.tsv"
        res_on_OUT = []
        scored_sgbs = self.Cardio_SGBs if not on_diet else self.Diet_SGBs
        for SGB in scored_sgbs:
            for dataset in self.all_datasets_OUT.loc["study_identifier"].unique():
                diseases = [c for c in self.all_datasets_OUT.loc[ "study_condition", \
                    (self.all_datasets_OUT.loc["study_identifier"]==dataset) ].unique().tolist() if c!="control" ]
                for disease in diseases:
                    cor_frame = self.get_a_smd( dataset, disease, SGB )
                    if not isinstance(cor_frame, str):
                        print(cor_frame)
                        if not len(res_on_OUT):
                            res_on_OUT = cor_frame
                        else:
                            res_on_OUT = pd.concat([res_on_OUT, cor_frame])
        _,fdr = fdrcorrection(res_on_OUT["p-val"].values.astype(float), alpha=0.05)
        res_on_OUT.insert(4, "FDR-q-val", fdr)
        res_on_OUT.to_csv(os.path.join("results/", OUTFILE), sep="\t", header=True, index=True)

    def get_a_pcorr(self, dataset_id, SGB):
        datas = self.all_datasets_BMI.loc[ self.metadata + [SGB], self.all_datasets_BMI.loc["study_identifier"]==dataset_id ]
        datas.loc[SGB] = np.arcsin(np.sqrt(datas.loc[SGB].values.astype(float)/100.))
        covars = ["sex", "age"]

        if datas.loc[SGB].astype(bool).astype(int).sum() >= 20:
            datast = datas.loc[self.metadata + [SGB]].T
            datast = datast.astype(float)
            covars = [c for c in covars if (len(datast[c].unique())>1)]

            pcrr = pg.partial_corr(data=datast, x="BMI", y=SGB, covar=covars, method="spearman")
            pcrr["SGB"] = SGB
            pcrr["study"] = dataset_id
            pcrr["CI95%"] = "-".join(list(map(str, pcrr["CI95%"])))

            return pcrr
        return "ABSENT"



#### ****** BLOCK ON BMI

    def perform_meta_analysis_on_one_dataset_and_BMI(self, study, frame, all_sgbs, counts, adj, class_a, class_b):
        if counts:
            vals = np.count_nonzero(frame.loc[ all_sgbs ].values.astype(float), axis=0)
            sams = frame.loc[ all_sgbs ].columns.tolist()
            if adj:
                formula = "count ~ C(BMI_class, Treatment(\"%s\")) + C(sex) + age" %class_a
            else:
                formula = "count ~ C(BMI_class, Treatment(\"%s\"))" %class_a
        else:
            vals = np.arcsin(np.sqrt(np.sum(frame.loc[ all_sgbs ].values.astype(float), axis=0)/100.))
            sams = frame.loc[ all_sgbs ].columns.tolist()
            if adj:
                formula = "cumulative ~ C(BMI_class, Treatment(\"%s\")) + C(sex) + age" %class_a
            else:
                formula = "count ~ C(BMI_class, Treatment(\"%s\"))" %class_a
 
        datast = pd.DataFrame({"count" if counts else "cumulative": list(vals), "sex": frame.loc["sex", sams].tolist(), \
            "age": frame.loc["age", sams].values.astype(float), "BMI": frame.loc["BMI", sams].values.astype(float), \
            "BMI_class": frame.loc["BMI_class", sams].values.astype(str), "richness": frame.loc["richness", sams].values.astype(float)}, index=sams)
 
        n_ctr = len(datast.loc[datast["BMI_class"] == class_a])
        n_css = len(datast.loc[datast["BMI_class"] == class_b])

        if n_ctr>=15 and n_css>=15:

            #self.class_numbers[ class_a ] += n_ctr

            datast = datast.astype({"count" if counts else "cumulative": float, "sex": str, "age": float, "BMI": float, "BMI_class": str, "richness": float})
            md = smf.ols(formula, data=datast)
            model_fit = md.fit()
 
            if ("C(BMI_class, Treatment(\"%s\"))[T.%s]" %(class_a, class_b)) in model_fit.params.index:
                the_var = "C(BMI_class, Treatment(\"%s\"))[T.%s]" %(class_a, class_b)
 
                eff = model_fit.params.loc[ the_var ]
                std = model_fit.bse.loc[ the_var ]
                pval = model_fit.pvalues.loc[ the_var ]

                if counts:
                    meta_an = {"effect": eff, "std_err": std, "p-val": pval, \
                        "95% CI": "|".join(list(map(str, [eff-std*1.96, eff+std*1.96]))), \
                        "n_ctrs": n_ctr, "n_cases": n_css}

                else:

                    eff = (model_fit.tvalues.loc[ the_var ] * (n_ctr + n_css)) / np.sqrt((n_ctr * n_css) * (n_ctr + n_css -2))
                    std = np.sqrt(((n_ctr + n_css -1)/(n_ctr + n_css -3))*((4/(n_ctr + n_css)) * (1 + ((eff**2.)/8.))))

                    meta_an = {"effect": eff, "std_err": std, "p-val": pval, \
                        "95% CI": "|".join(list(map(str, [eff-std*1.96, eff+std*1.96]))), \
                        "n_ctrs": n_ctr, "n_cases": n_css}

                return pd.DataFrame(meta_an, index=[study])

        return "TO_FEW" 


    def perform_meta_analysis_on_BMI_aggregated(self, title, on_diet, counts, adj):
        if not on_diet:
            top50, bot50 = self.Cardio_SGBs[ :50 ], self.Cardio_SGBs[ -50: ]
        else:
            top50, bot50 = self.Diet_SGBs[ :50 ], self.Diet_SGBs[ -50: ]

        analyses = [("normal", "overweight"), ("normal", "obese"), ("overweight", "obese")]
        meta_ana_def = "overall std. mean diff." if not counts else "overall mean difference"

        for class_a,class_b in analyses:
 
            OUTFILE_tt = "bmi_pooled_analyses/aggregated_FIRST_table_of_meta_analysis_on_BMI_%s_%s_%s_%svs%s.tsv" \
                %(title, "on-Cardio" if not on_diet else "on-Diet", "counts" if counts else "cumul", class_a, class_b)
            OUTFILE_bb = "bmi_pooled_analyses/aggregated_LAST_table_of_meta_analysis_on_BMI_%s_%s_%s_%svs%s.tsv" \
                %(title, "on-Cardio" if not on_diet else "on-Diet", "counts" if counts else "cumul", class_a, class_b)
 
            res_top, res_bot = [ ], [ ]

            relative_abs = self.abs.relab_BMI.copy(deep=True).loc[:, self.abs.relab_BMI.loc["BMI_class"].isin([class_a, class_b])]
            studies = relative_abs.loc["study_identifier"].unique().tolist()

            for study in studies:
                frame = relative_abs.loc[ :, relative_abs.loc["study_identifier"]==study ]
                ma = self.perform_meta_analysis_on_one_dataset_and_BMI( study, frame, top50, counts, adj, class_a, class_b )
 
                if not isinstance(ma, str):
                    if not len(res_top):
                        res_top = ma
                    else:
                        res_top = pd.concat([res_top, ma])
                else:
                    print(len(studies), study, " IS STRING")

            print(res_top)
 
            ma = generalized_meta_analysis( res_top["effect"], res_top["std_err"]**2., res_top["p-val"], res_top.index.tolist(), \
                res_top["n_ctrs"], res_top["n_cases"], "First 50", HET="PM" )

            #_,fdr = fdrcorrection(res_top["p-val"])
            #res_top["q-val"] = fdr
            #del res_top["p-val"]

            res_top.loc["meta_ana_def"] = pd.Series({\
                "effect": ma.RE, "std_err": ma.stdErr, \
                "p-val": ma.Pval, "95% CI": "|".join(list(map(str, [x for x in ma.conf_int]))), \
                "n_ctrs": res_top["n_ctrs"].sum(), "n_cases": res_top["n_cases"].sum()})

            res_top.index.name = "study"

            res_top.to_csv(OUTFILE_tt, sep="\t", header=True, index=True)
            #res_top.to_excel(OUTFILE_tt.replace(".tsv", ".xlsx"), header=True, index=True)

            for study in studies:
                frame = relative_abs.loc[ :, relative_abs.loc["study_identifier"]==study ]
                ma = self.perform_meta_analysis_on_one_dataset_and_BMI( study, frame, bot50, counts, adj, class_a, class_b )   

                if not isinstance(ma, str):
                    if not len(res_bot):
                        res_bot = ma
                    else:
                        res_bot = pd.concat([res_bot, ma])
 
            ma = generalized_meta_analysis( res_bot["effect"], res_bot["std_err"]**2., res_bot["p-val"], res_bot.index.tolist(), \
                res_bot["n_ctrs"], res_bot["n_cases"], "Last 50", HET="PM" )

            #_,fdr = fdrcorrection(res_bot["p-val"])
            #res_bot["q-val"] = fdr
            #del res_bot["p-val"]

            res_bot.loc["meta_ana_def"] = pd.Series({\
                "effect": ma.RE, "std_err": ma.stdErr, \
                "p-val": ma.Pval, "95% CI": "|".join(list(map(str, [x for x in ma.conf_int]))), \
                "n_ctrs": res_bot["n_ctrs"].sum(), "n_cases": res_bot["n_cases"].sum()})

            res_bot.index.name = "study"
 
            res_bot.to_csv(OUTFILE_bb, sep="\t", header=True, index=True)
            #res_bot.to_excel(OUTFILE_bb.replace(".tsv", ".xlsx"), header=True, index=True)




###**** END BLOCK OF BMI ***

    def get_cors_rec(self, on_diet=False):
        OUTFILE = "complete_corre_table_for_BMI_on-Cardio.tsv" if not on_diet else "complete_corre_table_for_BMI_on-Diet.tsv"
        res_on_BMI = []
        scored_sgbs = self.Cardio_SGBs if not on_diet else self.Diet_SGBs
        for SGB in scored_sgbs:
            for dataset in self.all_datasets_BMI.loc["study_identifier"].unique():
                cor_frame = self.get_a_pcorr( dataset, SGB )
                if not isinstance(cor_frame, str):
                    print(cor_frame)
                    if not len(res_on_BMI):
                        res_on_BMI = cor_frame
                    else:
                        res_on_BMI = pd.concat([res_on_BMI, cor_frame])
        _,fdr = fdrcorrection(res_on_BMI["p-val"].values.astype(float), alpha=0.05)
        res_on_BMI.insert(3, "FDR-q-val", fdr)
        res_on_BMI.to_csv(os.path.join("results/", OUTFILE), sep="\t", header=True, index=True)


    def perform_meta_analysis_on_one_SGB(self, SGB, dataframe):
        sgb_frame = dataframe.loc[dataframe["SGB"] == SGB]
        if len(sgb_frame) >= 3:
            median_rho = np.median(sgb_frame["r"].values.astype(float))
            ma = meta_analysis(np.arctanh(sgb_frame["r"].values.astype(float)), \
                sgb_frame["FDR-q-val"].values.astype(float), sgb_frame["study"].tolist(), None, None, SGB, EFF="precomputed", \
                variances_from_outside=[(1/(n-3)) for n in sgb_frame["n"].values.astype(float)], CI=False, HET="DL")
            meta_an = {"effect": np.arctanh(ma.RE), "std_err": np.arctanh(ma.stdErr), "p-val": ma.Pval, \
                "med.r": median_rho, "95% CI": "|".join(list(map(str, np.arctanh(ma.conf_int))))}
            return pd.DataFrame(meta_an, index=[SGB])
        return "TO_FEW"


    def perform_meta_analysis_on_one_dataset_and_one_disease_alternative_way(self, study, disease, dataframe, the_sgbs):
        dataset_frame = dataframe.loc[ (dataframe["disease"]==disease) & (dataframe["study"]==study) & (dataframe["SGB"].isin(the_sgbs)) ]
        if len(dataset_frame) >= 3:
            lenght = len(dataset_frame)

            ma = meta_analysis( \
                dataset_frame["d"].values.astype(float), \
                dataset_frame["FDR-q-val"].values.astype(float), dataset_frame["SGB"].tolist(), \
                dataset_frame["n_cases"].values.astype(float), \
                dataset_frame["n_ctrs"].values.astype(float), \
                disease, EFF="precomputed", \
                variances_from_outside=dataset_frame["se"].values.astype(float)**2., \
                CI=False, HET="FIX")

            meta_an = {"effect": ma.RE, "std_err": ma.stdErr, "p-val": ma.Pval, \
                "lenght": lenght, "95% CI": "|".join(list(map(str, ma.conf_int))) \
                }

            return pd.DataFrame(meta_an, index=[study + "-" + disease])
        return "TO_FEW"


    ##  perform_meta_analysis_on_one_dataset_and_one_disease(self, study, disease, dataset_frame, all_sgbs, counts, adj)
    def perform_meta_analysis_on_one_dataset_and_one_disease(self, study, disease, dataset_frame, all_sgbs, counts, adj):
        if counts: 
            ctrs = np.count_nonzero(self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]=="control") & \
                (self.abs.relab_OUT.loc["study_identifier"]==study) ].values.astype(float), axis=0)
            cases = np.count_nonzero(self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]==disease) & \
                (self.abs.relab_OUT.loc["study_identifier"]==study) ].values.astype(float), axis=0)
 
            sam_ctrs = self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]=="control") & (self.abs.relab_OUT.loc["study_identifier"]==study) ].columns.tolist()
            sam_cases= self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]==disease) & (self.abs.relab_OUT.loc["study_identifier"]==study) ].columns.tolist()
 
            #ctrs = [ (c if c>0. else 1 ) for c in ctrs ]
            #cases = [ (c if c>0. else 1 ) for c in cases ]

            #ctrs = np.log(ctrs)
            #cases = np.log(cases)

        else:
            ctrs = np.sum(self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]=="control") & \
                (self.abs.relab_OUT.loc["study_identifier"]==study) ].values.astype(float), axis=0)
            cases = np.sum(self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]==disease) & \
                (self.abs.relab_OUT.loc["study_identifier"]==study) ].values.astype(float), axis=0)

            sam_ctrs = self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]=="control") & (self.abs.relab_OUT.loc["study_identifier"]==study) ].columns.tolist()
            sam_cases= self.abs.relab_OUT.loc[ all_sgbs, (self.abs.relab_OUT.loc["study_condition"]==disease) & (self.abs.relab_OUT.loc["study_identifier"]==study) ].columns.tolist()

            #ctrs = [ (c if c>0. else 0.000005 ) for c in ctrs ]
            #cases = [ (c if c>0. else 0.000005 ) for c in cases ]

            #ctrs = np.log(ctrs)
            #cases = np.log(cases)

            ctrs = np.arcsin(np.sqrt(ctrs/100.))
            cases = np.arcsin(np.sqrt(cases/100.))

        if len(dataset_frame) >= 3:
            lenght = len(dataset_frame)

            if not adj:
                if not counts:
                    nctrs, ncases = len(ctrs), len(cases)
                    eff = pg.compute_effsize(cases, ctrs)
                    std = np.sqrt((nctrs+ncases)/(nctrs*ncases) + (eff**2.)/(2*(nctrs+ncases-2)))
                    _,pval = sts.ranksums(ctrs, cases)

                else:
                    nctrs, ncases = len(ctrs), len(cases)
                    datast = pd.DataFrame({\
                        "response": list(ctrs) + list(cases), "condition": ["no" for i in range(len(sam_ctrs))] + ["yes" for i in range(len(sam_cases))]}, \
                        index=sam_ctrs + sam_cases)
                    datast = datast.astype({"response": float, "condition": str})

                    md = smf.ols('response ~ C(condition, Treatment("no"))', data=datast)
                    model_fit = md.fit()

                    if 'C(condition, Treatment("no"))[T.yes]' in model_fit.params.index:
                        print(model_fit.summary())

                        eff = model_fit.params.loc[ 'C(condition, Treatment("no"))[T.yes]' ]
                        std = model_fit.bse.loc[ 'C(condition, Treatment("no"))[T.yes]' ]

                        pval = model_fit.pvalues.loc['C(condition, Treatment("no"))[T.yes]']

                    else: 
                        return "TO_FEW"

            else:
                nctrs, ncases = len(ctrs), len(cases)
                datast = pd.DataFrame({\
                    "response": list(ctrs) + list(cases), "condition": ["no" for i in range(len(sam_ctrs))] + ["yes" for i in range(len(sam_cases))], \
                    "sex": self.abs.relab_OUT.loc["sex", sam_ctrs + sam_cases].tolist(), "age": self.abs.relab_OUT.loc["age", sam_ctrs + sam_cases].values.astype(float), \
                    "BMI": self.abs.relab_OUT.loc["BMI", sam_ctrs + sam_cases].values.astype(float), \
                    "richness": self.abs.relab_OUT.loc["richness", sam_ctrs + sam_cases].values.astype(float)}, index=sam_ctrs + sam_cases)

                datast = datast.astype({"response": float, "condition": str, "sex": str, "age": float, "BMI": float, "richness": float})

                md = smf.ols('response ~ C(condition, Treatment("no")) + C(sex, Treatment(\"male\")) + BMI + age', data=datast)
                model_fit = md.fit()

                ## C(sex)

                if 'C(condition, Treatment("no"))[T.yes]' in model_fit.params.index:
                    model_fit = md.fit()
                    print(model_fit.summary())

                    if counts:
                        eff = model_fit.params.loc[ 'C(condition, Treatment("no"))[T.yes]' ]
                        std = model_fit.bse.loc[ 'C(condition, Treatment("no"))[T.yes]' ]

                    else: 
                        t = model_fit.tvalues.loc[ 'C(condition, Treatment("no"))[T.yes]' ]
                        eff = (t*(nctrs+ncases))/float(np.sqrt(nctrs*ncases)*np.sqrt(nctrs+ncases-2))
                        std = np.sqrt(((nctrs+ncases-1)/float(nctrs+ncases-3)) * ((4./float(nctrs+ncases))*(1+((eff**2.)/8.))))

                    pval = model_fit.pvalues.loc['C(condition, Treatment("no"))[T.yes]']

                else: 
                    return "TO_FEW"

            meta_an = {"effect": eff, "std_err": std, "p-val": pval, "# SGB. used": lenght, \
                "95% CI": "|".join(list(map(str, [eff-std*1.96, eff+std*1.96]))), \
                "nctrs": nctrs, "ncases": ncases, "mean_ctr": np.mean(ctrs), "mean_css": np.mean(cases), \
                "std_ctr": np.std(ctrs), "std_css": np.std(cases)}

            return pd.DataFrame(meta_an, index=[study + "-" + disease])
        return "TO_FEW"


    def perform_plain_meta_analysis_on_BMI(self, BMI_dataframe, on_diet=False):
        outfile = "results/synthetic_table_of_meta_analysis_on_BMI_plain_%s.tsv" %("on-Cardio" if not on_diet else "on-Diet")
        res = []
        for sgb in BMI_dataframe["SGB"].unique().tolist():
            mm = self.perform_meta_analysis_on_one_SGB(sgb, BMI_dataframe)
            if not isinstance(mm, str):
                if not len(res):
                    res = mm
                else:
                    res = res.append(mm)
        _,fdr = fdrcorrection(res["p-val"].values.astype(float), alpha=0.05)
        res.insert(4, "FDR-q-val", fdr)
        res["a"], res["b"] = [(0 if q<SIGN_TH else 1) for q in res["FDR-q-val"].values], np.abs(res["effect"].values)
        res.sort_values(by=["a", "b"], ascending=[True, False], inplace=True)
        del res["a"]; del res["b"]
        res.to_csv(outfile, sep="\t", header=True, index=True)


    def perform_meta_analysis_on_outcomes_aggregated(self, OUT_dataframe, title, on_diet, counts, adj):
        if not on_diet:
            top50, bot50 = self.Cardio_SGBs[ :50 ], self.Cardio_SGBs[ -50: ]
        else:
            top50, bot50 = self.Diet_SGBs[ :50 ], self.Diet_SGBs[ -50: ]
            
        OUTFILE_tt = "results/aggregated_TOP_table_of_meta_analysis_on_OUT_hier_%s_%s_%s.tsv" %(title, "on-Cardio" if not on_diet else "on-Diet", "counts" if counts else "cumul")
        OUTFILE_bb = "results/aggregated_BOT_table_of_meta_analysis_on_OUT_hier_%s_%s_%s.tsv" %(title, "on-Cardio" if not on_diet else "on-Diet", "counts" if counts else "cumul")

        combos = set()
        res_top, res_bot = [ ], [ ]

        for study,disease in zip(OUT_dataframe["study"].tolist(), OUT_dataframe["disease"].tolist()):
            if not (study,disease) in combos:
                combos.add((study,disease))
                frame = OUT_dataframe.loc[ (OUT_dataframe["study"] == study) & (OUT_dataframe["disease"] == disease) & (OUT_dataframe["SGB"].isin(top50)) ]
                ma = self.perform_meta_analysis_on_one_dataset_and_one_disease(study, disease, frame, top50, counts, adj)
                if not isinstance(ma, str):
                    if not len(res_top):
                        res_top = ma
                    else:
                        res_top = pd.concat([res_top, ma])
        res_top.to_csv(OUTFILE_tt, sep="\t", header=True, index=True)

        combos = set()
        for study,disease in zip(OUT_dataframe["study"].tolist(), OUT_dataframe["disease"].tolist()):
            if not (study,disease) in combos:
                combos.add((study,disease))
                frame = OUT_dataframe.loc[ (OUT_dataframe["study"] == study) & (OUT_dataframe["disease"] == disease) & (OUT_dataframe["SGB"].isin(bot50)) ]
                ma = self.perform_meta_analysis_on_one_dataset_and_one_disease(study, disease, frame, bot50, counts, adj)
                if not isinstance(ma, str):
                    if not len(res_bot):
                        res_bot = ma
                    else:
                        res_bot = pd.concat([res_bot, ma])
        res_bot.to_csv(OUTFILE_bb, sep="\t", header=True, index=True)


    def perform_meta_analysis_on_one_SGB_in_outcomes(self, sgb, OUT_dataframe):
        sgb_frame = OUT_dataframe.loc[ OUT_dataframe["SGB"] == sgb ]
        diseases = sgb_frame["disease"].unique().tolist()
        Studies  =  [ ]
        effects  =  [ ]
        std_errs =  [ ]
        pvals =     [ ]
        n_poss =    [ ]
        n_negs =    [ ]
  
        multiple_diseases = [d for d in ["CRC", "UC", "CD", "T2D", "IGT", "MS"] if d in diseases]
        single_diseases   = [d for d in diseases if not d in multiple_diseases]
 
        for disease_type in multiple_diseases:
            n_pos = sgb_frame.loc[sgb_frame["disease"]==disease_type, "n_cases"].values.astype(float)
            n_neg = sgb_frame.loc[sgb_frame["disease"]==disease_type, "n_ctrs"].values.astype(float)

            intermediate_meta_an = meta_analysis( \
              sgb_frame.loc[sgb_frame["disease"]==disease_type, "d"].values.astype(float), \
              sgb_frame.loc[sgb_frame["disease"]==disease_type, "FDR-q-val"].values.astype(float), \
              sgb_frame.loc[sgb_frame["disease"]==disease_type, "study"].tolist(), \
              n_pos, n_neg, sgb, EFF="precomputed", variances_from_outside= \
              sgb_frame.loc[sgb_frame["disease"]==disease_type, "se"].values.astype(float)**2., \
              CI=False, HET="PM")

            effects  += [ intermediate_meta_an.RE ]
            std_errs += [ intermediate_meta_an.stdErr ]
            pvals    += [ intermediate_meta_an.Pval ]
            n_poss   += [ np.sum(n_pos) ]
            n_negs   += [ np.sum(n_neg) ]
            Studies  += [ "all on %s" %disease_type]

        for disease_type in single_diseases:
            effects  += [ float(sgb_frame.loc[ sgb_frame["disease"]==disease_type, "d" ]) ]
            std_errs += [ float(sgb_frame.loc[ sgb_frame["disease"]==disease_type, "se" ]) ]
            pvals    += [ float(sgb_frame.loc[ sgb_frame["disease"]==disease_type, "FDR-q-val" ]) ]
            n_poss   += [ float(sgb_frame.loc[ sgb_frame["disease"]==disease_type, "n_cases" ]) ]
            n_negs   += [ float(sgb_frame.loc[ sgb_frame["disease"]==disease_type, "n_ctrs" ]) ]
            Studies  += [ "%s (%s)" %(sgb_frame.loc[ sgb_frame["disease"]==disease_type, "study" ], disease_type) ]
 
        if len(effects) >= 3:
            ma = meta_analysis(\
                np.array(effects, dtype=np.float64), \
                np.array(pvals, dtype=np.float64), Studies, 
                np.array(n_poss, dtype=np.float64), \
                np.array(n_negs, dtype=np.float64), \
                sgb, EFF="precomputed", \
                variances_from_outside=np.array(std_errs, dtype=np.float64)**2., \
                CI=False, HET="PM")
                        
            meta_an = {"effect": ma.RE, "std_err": ma.stdErr, "p-val": ma.Pval, "95% CI": "|".join(list(map(str, ma.conf_int)))}
            return pd.DataFrame(meta_an, index=[sgb])
        return "TO_FEW"

    def perform_meta_analysis_on_a_disease(self, OUT_dataframe, title, on_diet=False):
        outfile = "results/synthetic_table_of_meta_analysis_on_OUT_hier_%s_%s.tsv" %(title, "on-Cardio" if not on_diet else "on-Diet")
        res = []
        for sgb in OUT_dataframe["SGB"].unique().tolist():
            mm = self.perform_meta_analysis_on_one_SGB_in_outcomes(sgb, OUT_dataframe)
            if not isinstance(mm, str):
                if not len(res):
                    res = mm
                else:
                    res = res.append(mm)
        _,fdr = fdrcorrection(res["p-val"].values.astype(float), alpha=0.05)
        res.insert(4, "FDR-q-val", fdr)
        res["a"], res["b"] = [(0 if q<SIGN_TH else 1) for q in res["FDR-q-val"].values], np.abs(res["effect"].values)
        res.sort_values(by=["a", "b"], ascending=[True, False], inplace=True)
        del res["a"]; del res["b"]
        res.to_csv(outfile, sep="\t", header=True, index=True)

    def summarize_into_meta_analyses(self, \
        BMI_dataframe_Cardio, BMI_dataframe_Diet, \
        \
        OUT_dataframe_Cardio, OUT_dataframe_Diet, \
        OUT_dataframe_Cardio_NoBMI, OUT_dataframe_Diet_NoBMI, \
        OUT_dataframe_Cardio_Crude, OUT_dataframe_Diet_Crude):

        #self.perform_plain_meta_analysis_on_BMI(BMI_dataframe_Cardio, on_diet=False)
        #self.perform_plain_meta_analysis_on_BMI(BMI_dataframe_Diet, on_diet=True)

        self.perform_meta_analysis_on_a_disease(OUT_dataframe_Cardio, "Adj", on_diet=False)
        self.perform_meta_analysis_on_a_disease(OUT_dataframe_Diet, "Adj", on_diet=True)

        self.perform_meta_analysis_on_a_disease(OUT_dataframe_Cardio_NoBMI, "NoBMI", on_diet=False)
        self.perform_meta_analysis_on_a_disease(OUT_dataframe_Diet_NoBMI, "NoBMI", on_diet=True)

        self.perform_meta_analysis_on_a_disease(OUT_dataframe_Cardio_Crude, "Cru", on_diet=False)
        self.perform_meta_analysis_on_a_disease(OUT_dataframe_Diet_Crude, "Cru", on_diet=True)





    def main(self):

        ## BLOCK FOR OUTCOMES
        self.get_diff_rec(on_diet=False)
        self.get_diff_rec(on_diet=True)
 
        OUT_dataframe_Cardio = pd.read_csv("results/complete_corre_table_for_OUTCOMES_on-Cardio.tsv", sep="\t", header=0, index_col=None, low_memory=False)
        OUT_dataframe_Diet = pd.read_csv("results/complete_corre_table_for_OUTCOMES_on-Diet.tsv", sep="\t", header=0, index_col=None, low_memory=False)
 
        #self.perform_meta_analysis_on_outcomes_aggregated(OUT_dataframe_Cardio, "ORDINARY", on_diet=False, counts=True, adj=True)
        #self.perform_meta_analysis_on_outcomes_aggregated(OUT_dataframe_Diet, "ORDINARY", on_diet=True, counts=True, adj=True)

        #self.perform_meta_analysis_on_outcomes_aggregated(OUT_dataframe_Cardio, "ORDINARY", on_diet=False, counts=False, adj=True)
        #self.perform_meta_analysis_on_outcomes_aggregated(OUT_dataframe_Diet, "ORDINARY", on_diet=True, counts=False, adj=True)
 
        ## BLOCK FOR BMI
        self.perform_meta_analysis_on_BMI_aggregated("ORDINARY", on_diet=False, counts=True, adj=True)
        self.perform_meta_analysis_on_BMI_aggregated("ORDINARY", on_diet=True, counts=True, adj=True)

        self.perform_meta_analysis_on_BMI_aggregated("ORDINARY", on_diet=False, counts=False, adj=True)
        self.perform_meta_analysis_on_BMI_aggregated("ORDINARY", on_diet=True, counts=False, adj=True)


if __name__ == "__main__":
    ZsoB = ZOE_scores_fundamental_analyses()
    ZsoB.main()
