import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imblearn.combine import SMOTEENN
import random
random.seed(6)

alleleFrq = pd.read_csv("../data_primary/AlleleFrequency_protRes_2methods.csv")
alleleSelect = alleleFrq[alleleFrq.selectedForStudy==1].ProtRes.values
print ("Allele Selected for analysis =", alleleSelect.shape)
prgGT = pd.read_csv("../data_primary/HLA_LA_protGT.csv").set_index("QBB_DUMMY_ID")[alleleSelect]
hdGT = pd.read_csv("../data_primary/HLA_HD_protGT.csv").set_index("QBB_DUMMY_ID")[alleleSelect]
print ("Genotype Matrix", prgGT.shape, hdGT.shape)

# igE_clinic = pd.read_csv("../data_primary/clinicBool.csv").set_index("Unnamed: 0")
igE_clinic = pd.read_csv("../data_primary/clinicBool_34cutoff.csv").set_index("QBB_DUMMY_ID") # changed from 0.3 to 0.34 on 1June2022
covar = pd.read_csv("../data_primary/covarExted.csv").set_index("QBB_DUMMY_ID")
_commonIdx = set(igE_clinic.index).intersection(covar.index)
print (len(_commonIdx))

# make final matrix
prgGT_bool = prgGT.astype(bool).astype(int).loc[_commonIdx].sort_index()
hdGT_bool  = hdGT.astype(bool).astype(int).loc[_commonIdx].sort_index()
igE_clinic = igE_clinic.loc[_commonIdx].sort_index()
covarx = covar.loc[_commonIdx].sort_index()

print ("Genotypes Matrix", prgGT_bool.shape, hdGT_bool.shape)
print ("allergen phenotype", igE_clinic.shape)
print ("Covariates", covarx.shape)

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%D_%H:%M:%S").replace("/","-")
print("Current Time =", current_time)

### 2 Poly sensitization

# Test Association with all allergen at once for each allele
# allele1 = C + b1* FX1 + b2 * FX2 + b[3-8] * [FX22,WX1, GX2, phad] + b9 * covar 

def PolySensitization(hlaCalling,typing="HLA-HD",bootstrap=10,pval_alpha=0.05):
    hdm1 = pd.DataFrame()
    _errCombi = []

    for k in hlaCalling.columns:
        y = hlaCalling[k]
        Xmat   = igE_clinic.join(covarx)
        
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
                tD['iter'] = itX+1
                tD['allele'] = k
                tD['HLA_typing'] = typing
                hdm1 = pd.concat([hdm1,tD],axis=0).reset_index().drop('index',axis=1)
            except:
                _errCombi.append([k,itX])

    reject,pval_corrected,alphacSidak, alphacBonf = sm.stats.multipletests(hdm1.pvalue_raw,method='holm',alpha=pval_alpha)
    hdm1['pval_Holm']  = pval_corrected
    hdm1['nlog10_Pval'] = -1*np.log10(hdm1.pval_Holm)
    return hdm1,_errCombi


bootIter = 100
p_alpha  = 0.005
m1HD,_errHD = PolySensitization(hdGT_bool,typing="HLA-HD",bootstrap=bootIter,pval_alpha=p_alpha)
m1LA,_errLA =  PolySensitization(prgGT_bool,typing="HLA-PRG",bootstrap=bootIter,pval_alpha=p_alpha)

print ("HD associations", m1HD.shape, "\n LA associaition:", m1LA.shape)

mSdf = pd.concat([m1HD,m1LA],axis=0).reset_index().drop(['index'],axis=1)
mSdf = mSdf[mSdf.features.isin(igE_clinic.columns)].reset_index().drop('index',axis=1)

saveFlname = '../AssociationResults_34cutoff/PolySensitization_Model2_{}_{}.csv'.format(bootIter,current_time)
mSdf.to_csv(saveFlname)
print (mSdf.groupby("HLA_typing").size())
print ("Final result {}, saved in {}".format(mSdf.shape[0],saveFlname))