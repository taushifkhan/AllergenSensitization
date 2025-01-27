import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imblearn.combine import SMOTEENN
import random
random.seed(6)



bootIter = 100
p_alpha  = 0.005

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%D_%H:%M:%S").replace("/","-")
print("Current Time =", current_time)


# igE_clinic = pd.read_csv("../data_primary/clinicBool.csv").set_index("Unnamed: 0")
igE_clinic = pd.read_csv("../data_primary/clinicBool_34cutoff.csv").set_index("QBB_DUMMY_ID") # changed from 0.3 to 0.34 on 1June2022
covar = pd.read_csv("../data_primary/covarExted.csv").set_index("QBB_DUMMY_ID")

ps_Score = igE_clinic.sum(axis=1)/igE_clinic.shape[1]
Xmat_psCovar = pd.DataFrame(ps_Score).join(covar).rename({0:'psScore'},axis=1)

def PS_Score(hlaCalling,Xmat_psCovar,typing="HLA-HD",bootstrap=10,pval_alpha=0.05):
    hdm1 = pd.DataFrame()
    _errCombi = []

    for k in hlaCalling.columns:
        y = hlaCalling[k]
        Xmat   = Xmat_psCovar
        
        for itX in np.arange(bootstrap):
            smote_enn = SMOTEENN(random_state=random.seed(0))
            X_resampled, y_resampled = smote_enn.fit_resample(Xmat, y)
            print ("Resampled", X_resampled.shape, y_resampled.shape)
            X_resampled = sm.add_constant(X_resampled)
            try:
                mod = sm.Logit(y_resampled,X_resampled)
                res = mod.fit()
                tD = pd.read_html(res.summary().tables[1].as_html(),header=0,index_col=0)[0]
                tD['pvalue_raw']  = res.pvalues
                tD['features'] = X_resampled.columns
                tD['iter'] = itX + 1
                tD['allele'] = k
                tD['HLA_typing'] = typing
                hdm1 = pd.concat([hdm1,tD],axis=0).reset_index().drop('index',axis=1)
            except:
                _errCombi.append([k,itX])

    reject,pval_corrected,alphacSidak, alphacBonf = sm.stats.multipletests(hdm1.pvalue_raw,method='holm',alpha=pval_alpha)
    hdm1['pval_Holm']  = pval_corrected
    hdm1['nlog10_Pval'] = -1*np.log10(hdm1.pval_Holm)
    return hdm1,_errCombi


# HLA-HD
haploHD= pd.read_csv("../data_primary/HD_haplotype.csv").set_index("haplotype").T
_common = set(Xmat_psCovar.index).intersection(set(haploHD.index))
haploHDX = haploHD.loc[_common]
Xmat_psCovar_hd = Xmat_psCovar.loc[_common]
print (haploHDX.shape,Xmat_psCovar_hd.shape)

hdMS, hdHaploErr = PS_Score(haploHDX,Xmat_psCovar_hd,bootstrap=bootIter,pval_alpha=p_alpha) 


# HLA-LA
haploLA= pd.read_csv("../data_primary/prg_haplotype.csv").set_index("haplotype").T
_common = set(Xmat_psCovar.index).intersection(set(haploLA.index))
haploLAX = haploLA.loc[_common]
Xmat_psCovar_la = Xmat_psCovar.loc[_common]
print (haploLAX.shape,Xmat_psCovar_la.shape)

laMS, hdHaploErr = PS_Score(haploLAX,Xmat_psCovar_la,typing="HLA-LA",bootstrap=bootIter,pval_alpha=p_alpha) 

mSdf = pd.concat([hdMS,laMS],axis=0).reset_index().drop(['index'],axis=1)
mSdf = mSdf[mSdf.features=='psScore'].reset_index().drop('index',axis=1)

saveFlname = '../AssociationResults_34cutoff/Haplotype-PS_score_Model3_{}_{}.csv'.format(bootIter,current_time)
mSdf.to_csv(saveFlname)
print (mSdf.groupby("HLA_typing").size())
print ("Final result {}, saved in {}".format(mSdf.shape[0],saveFlname))