#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy import stats as sts
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys, os
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams["svg.fonttype"] = "none"
from scipy import stats as sts
sns.set_style("white")
import argparse as ap


def confIntMeanT(mn, se, n, conf=0.95):
    m = sts.t.ppf((1+conf)/2., n-1)
    return mn - m*se, mn + m*se


def get_meta_analyses(one, two):
    one = pd.read_csv(one, sep="\t", header=0, index_col=0)
    two = pd.read_csv(two, sep="\t", header=0, index_col=0)
    frames = []
    idx = ["IBD summary", "CRC summary", "IGT summary", "T2D summary", "ACVD summary", "summary"]
    
    for frame in [one, two]:
        frame["type"] = [("analysis" if (not i.endswith("summary") and not i.endswith("effect")) else ("disease summary" if "summary" in i else i)) for i in frame.index]

        sg = [ ]
        for ix,p in zip(frame.index, frame["p-val"].astype(float)):
            if "summary" in ix or "effect" in ix:
                if p<0.05:
                    sg += ["meta-an sg"]
                else:
                    sg += ["meta-an ns"]
            else:
                if p<0.05:
                    sg += ["sg"]
                else:
                    sg += ["ns"]

        frame["sg"] = sg
        frames += [frame]
    return tuple(frames)



#def get_meta_analyses(one, two):
#    one = pd.read_csv(one, sep="\t", header=0, index_col=0)
#    two = pd.read_csv(two, sep="\t", header=0, index_col=0)
##    studies = list(set().union(*[one.index.tolist(), two.index.tolist() ]))
#    frames = []
#    single_frames = []
#    idx = ["IBD summary", "CRC summary", "IGT summary", "T2D summary", "ACVD summary", "summary"]
#
#    for frame in [one, two]:
#        frame["type"] = [("analysis" if not i.endswith("summary") else ("disease summary" if not i=="summary" else "summary")) for i in frame.index]
#        sg = [ ]
#        for ix,p in zip(frame.index, frame["p-val"].astype(float)):
#            if "summary" in ix:
#                if p<0.05:
#                    sg += ["meta-an sg"]
#                else:
#                    sg += ["meta-an ns"]
#            else:
#                if p<0.05:
#                    sg += ["sg"]
#                else:
#                    sg += ["ns"]
#
#        frame["sg"] = sg
#        frames += [frame]
#    return tuple(frames)



def forest(frame, ax, x_lab, title, neg, pos, yticks=False, lims=None, displacement=""):

    #idx = ["IBD summary", "CRC summary", "IGT summary", "T2D summary", "ACVD summary", "covarying effect", "tot.random effect", "random effect"]
    idx = ["IBD", "CRC", "IGT", "T2D", "ACVD", "covarying effect", "tot.random effect", "random effect"]
    idx_n = [7, 6, 5, 4, 3, 2, 1, 0]
    idx_m = dict([(x,y) for x,y in zip(idx,idx_n)])

    print(idx_m)

    frame["response"] = [(r if not r.startswith("Pool") else ix) for r,ix in zip(frame["response"].tolist(), frame.index.tolist())]
    #frame["index"] = [ idx_m.get(r + " summary", idx_m[r]) for r in frame["response"].tolist() ]
    frame["index"] = [ idx_m.get(r, idx_m[r]) for r in frame["response"].tolist() ]

    frame = frame.sort_values("index", ascending=True)
    ax.set_yticks([7, 6, 5, 4, 3, 2, 1, 0])

    if displacement=="upp":
        frame["index"] += 0.2

    elif displacement=="dow":
        frame["index"] -= 0.2 

    elif displacement=="med":
        #frame["index"] = 0
        pass

    print(frame)
    print(ax.get_yticklabels(), ax.get_yticks())
   
    ax.axvline(0, ls="--", c="red")
    palette = {"meta-an sg": "red", "meta-an ns": "lightgrey", "sg": "orange", "ns": "darkcyan"}

    scatter = sns.scatterplot(data=frame, x="effect", y="index", ax=ax, hue="sg", style="type", edgecolor="black",
        palette=palette, markers={"analysis": "o", "disease summary": "D", "summary": "D", "covarying effect": "D", "tot.random effect": "D", "random effect": "D"})

    print(frame)

    for ix,ix2 in zip(idx_n, idx):
        if not "effect" in ix2:
            ix2 = ix2 + " summary"

        
        clr = palette[frame.loc[ix2, "sg"]]

        low,upp = confIntMeanT(float(frame.loc[ix2, "effect"]), float(frame.loc[ix2, "se"]), frame.loc[ix2, ["n_ctrs", "n_cases"]].astype(float).sum(), conf=0.95)

        if displacement=="upp":
            ix += 0.2 
        elif displacement=="dow":
            ix -= 0.2

        ax.plot([low, upp], [ix, ix], color=clr)

    scatter.get_legend().set_visible(False)

    for x in ax.get_xticks():
        if x!=0.0: ## and displacement!="med":
            ax.axvline(x, ls="--", c="lightgray", alpha=0.5)

    if bool(lims):
        ax.set_xlim(lims)

    if yticks:
        ax.set_yticklabels([t.replace(" summary", "") for t in idx])
    else:
        ax.set_yticklabels([])
 
    idx = ["IBD summary", "CRC summary", "IGT summary", "T2D summary", "ACVD summary", "covarying effect", "tot.random effect", "random effect"]

    if displacement=="med":
        ax.set_yticks(ax.get_yticks())
 
    if displacement=="dow":    
        ax2 = ax.twinx()
        print(frame.loc[idx, "index"].tolist(), ax.get_yticks(), "  <=================================================")
  
        ax2.set_yticks(ax.get_yticks()) ##[(i-0.2) for i in ax.get_yticks()])
        ax2.set_ylim(ax.get_ylim())

        if not yticks:
            ax2.set_yticklabels([("%i/%i" %(ct,cs)) for cs,ct in zip(frame.loc[idx, "n_cases"].astype(float), frame.loc[idx, "n_ctrs"].astype(float))])
        else:
            ax2.set_yticklabels([])

    if title:
        ax.set_title(title)
    ax.set_ylabel("")

    if x_lab:
        ax.set_xlabel(x_lab)



def generate_figure(X, Y, W, Z, XX, YY, title):
    one, two = X
    three, four = Y
    five, six = W
    seven, eight = Z
    nine, ten = XX
    eleven, twelve = YY
     
    fig = plt.figure(figsize=(30, 5.5))
    gs = gridspec.GridSpec(1, 6, wspace=0.35, width_ratios=[1, 1, 1, 1, 1, 1])

    ax_1 = plt.subplot(gs[0, 0])
    ax_2 = plt.subplot(gs[0, 1])
    ax_3 = plt.subplot(gs[0, 2])
    ax_4 = plt.subplot(gs[0, 3])
    ax_5 = plt.subplot(gs[0, 4])
    ax_6 = plt.subplot(gs[0, 5])

    print(one.shape, two.shape, three.shape, four.shape)
  
    forest(one, ax_1, "", "", "control", "disease", False, [-1.5, 0.5], "upp")
    forest(two, ax_1, "SMD", "ZOE MB Health/Diet (norm.)", "control", "disease", True, [-1.5, 0.5], "dow")

    forest(three, ax_2, "", "", "control", "disease", False, [-1.5, 0.5], "upp")
    forest(four, ax_2, "SMD", "ZOE MB Health/Diet (asin-norm)", "control", "disease", False, [-1.5, 0.5], "dow")
 
    forest(five, ax_3, "", "", "control", "disease", False, [-20, 5], "upp")
    forest(six, ax_3, "mean diff", "ZOE MB Health/Diet (count-fav)", "control", "disease", False, [-20, 5], "dow")

    forest( seven, ax_4, "", "", "control", "disease", False, [-6, 6], "upp" )
    forest( eight,  ax_4, "mean diff", "ZOE MB Health/Diet (count-unfav)", "control", "disease", False, [-6, 6], "dow" )

    forest( nine, ax_5, "", "", "control", "disease", False, [-1.5, .5], "upp" )
    forest( ten,  ax_5, "SMD", "ZOE MB Health/Diet (cumul-fav)", "control", "disease", False, [-1.5, .5], "dow" )

    forest( eleven, ax_6, "", "", "control", "disease", False, [-1., 1.], "upp" )
    forest( twelve,  ax_6, "SMD", "ZOE MB Health/Diet (cumul-unfav)", "control", "disease", False, [-1., 1.], "dow" )

    [plt.savefig("%s.%s" %(title, fmt), dpi=200) for fmt in ["svg", "png"]]



def main():

    generate_figure( \
    get_meta_analyses( \
        "../pooling_disease_meta_analyses/summary_cardio_minusone_to_one_SMD.tsv", 
        "../pooling_disease_meta_analyses/summary_diet_minusone_to_one_SMD.tsv"), 
    \
    get_meta_analyses( \
        "../pooling_disease_meta_analyses/summary_cardio_minusone_to_one_arcsin_SMD.tsv", 
        "../pooling_disease_meta_analyses/summary_diet_minusone_to_one_arcsin_SMD.tsv"), 
    \
    get_meta_analyses( \
        "../pooling_disease_meta_analyses/summary_count_good_cardio_MND.tsv", 
        "../pooling_disease_meta_analyses/summary_count_good_diet_MND.tsv"), 
    \
    get_meta_analyses( \
        "../pooling_disease_meta_analyses/summary_count_bad_cardio_MND.tsv", 
        "../pooling_disease_meta_analyses/summary_count_bad_diet_MND.tsv"), \
    \
    get_meta_analyses( \
        "../pooling_disease_meta_analyses/summary_cumul_good_cardio_SMD.tsv",
        "../pooling_disease_meta_analyses/summary_cumul_good_diet_SMD.tsv"), \
    \
    get_meta_analyses( \
        "../pooling_disease_meta_analyses/summary_cumul_bad_cardio_SMD.tsv",
        "../pooling_disease_meta_analyses/summary_cumul_bad_diet_SMD.tsv"), \
    \
    "disease_pooled_analysis_image")


if __name__ == "__main__":
    main()
