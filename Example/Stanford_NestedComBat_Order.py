import pandas as pd
import neuroCombat as nC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import ranksums, ttest_ind, ttest_rel, ks_2samp
import os
import numpy as np
from itertools import permutations
import OPNestedComBat as nested


# Loading in features
filepath = "C:/Users/horng/OneDrive/CBIG/Stanford/"
filepath2 = 'C:/Users/horng/OneDrive/CBIG/051221_Manuscript/GitHub/ex_test/'
if not os.path.exists(filepath2):
    os.makedirs(filepath2)

# Loading in batch effects
batch_df = pd.read_excel('C:/Users/horng/OneDrive/CBIG/Stanford/Sandy-CT-parameters.xlsx')
batch_df = batch_df[batch_df['Manufacturer'] != 'Philips'].reset_index(drop=True)  # Dropping Phillips Case
batch_df = batch_df[batch_df['Manufacturer'] != 'TOSHIBA'].reset_index(drop=True)  # Dropping Toshiba Case
batch_df = batch_df[batch_df['Manufacturer'] != 0].reset_index(drop=True)  # Dropping 0 Case
batch_df['clc'] = batch_df['clc'].apply(lambda x: x[:-1] if len(x) > 7 else x)

# batch_df = batch_df[batch_df['ID'].isin(caseno)].reset_index(drop=True)
batch_list = ['Manufacturer', 'KernelResolution', 'CE']

# Loading in clinical covariates
covars_df = pd.read_csv('C:/Users/horng/OneDrive/CBIG/Stanford/clinical-demographics.csv')
categorical_cols = ['event', 'sex', 'smoking', 'histology']
continuous_cols = ['days']

# CAPTK
data_df = pd.read_csv(filepath+'Sandy-Captk.csv')
data_df = data_df.reset_index(drop=True)
data_df = data_df.dropna()
data_df = data_df.rename(columns={"SubjectID": "Case"})
data_df = data_df[data_df['Case'] != 'R01-159'].reset_index(drop=True)  # Missing covariates
data_df = data_df.merge(batch_df['clc'], left_on='Case', right_on='clc')
dat = data_df.iloc[:, 1:-1]
dat = dat.T.apply(pd.to_numeric)
caseno = data_df['Case'].str.upper()

# Merging batch effects, clinical covariates
batch_df = data_df[['Case']].merge(batch_df, left_on='Case', right_on='clc')
covars_df = data_df[['Case']].merge(covars_df, left_on='Case', right_on='Unnamed: 0')
covars_string = pd.DataFrame()
covars_string[categorical_cols] = covars_df[categorical_cols].copy()
covars_string[batch_list] = batch_df[batch_list].copy()
covars_quant = covars_df[continuous_cols]

# Encoding categorical variables
covars_cat = pd.DataFrame()
for col in covars_string:
    stringcol = covars_string[col]
    le = LabelEncoder()
    le.fit(list(stringcol))
    covars_cat[col] = le.transform(stringcol)

covars = pd.concat([covars_cat, covars_quant], axis=1)

# # FOR GMM COMBAT VARIANTS:
# # Adding GMM Split to batch effects
gmm_df = nested.GMMSplit(dat, caseno, filepath2)
gmm_df_merge = covars_df.merge(gmm_df, right_on='Patient', left_on='Unnamed: 0')
covars['GMM'] = gmm_df_merge['Grouping']

# # EXECUTING OPNESTED+GMM COMBAT
# # Here we add the newly generated GMM grouping to the list of batch variables for harmonization
# batch_list = batch_list + ['GMM']

# EXECUTING OPNESTED-GMM COMBAT
# Here we add the newly generated GMM grouping to the list of categorical variables that will be protected during
# harmonization
categorical_cols = categorical_cols + ['GMM']

# Completing Nested ComBat
output_df = nested.OPNestedComBat(dat, covars, batch_list, filepath2, ad=True, categorical_cols=categorical_cols,
                                  continuous_cols=continuous_cols)
write_df = pd.concat([caseno, output_df], axis=1) # write results fo file
write_df.to_csv(filepath2+'/features_NestedComBat.csv')

# Compute the AD test p-values to measure harmonziation performance
nested.feature_ad(dat.T, output_df, covars, batch_list, filepath2)
# Plot kernel density plots to visualize distributions before and after harmonization
nested.feature_histograms(dat.T, output_df, covars, batch_list, filepath2)
