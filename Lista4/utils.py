import psycopg2
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import scipy

def check_normality(Data_df):
    
    # Generating absolute frequency
    table_df = Data_df.value_counts().reset_index(name='Fabs')
    
    # Generating cumulative frequency
    table_df['Fac'] = table_df['Fabs'].cumsum()
    
    # Computing fractionary column: cumulative frequency by element over total cumulative frequency
    table_df['Frac'] = table_df['Fac']/table_df['Fac'].max()
    
    # Computing z-score
    mean = Data_df.mean()
    std = Data_df.std()
    table_df['Zi'] = table_df.iloc[:, 0].apply(lambda x: (x - mean)/std)
    
    import scipy.special as scsp
    def zScoreToPvalue(z):
        # Retornar p-value a partir do z-score
        return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))
    
    # Computing expected value according to p-value
    table_df['FracEsp'] = table_df['Zi'].apply(lambda x: zScoreToPvalue(x))
    
    # Computing D-negative and D-positive
    table_df['D_neg'] = abs(table_df['FracEsp']-table_df['Frac'])
    
    table_df['D_pos'] = 0
    for i in range(table_df['Frac'].shape[0]):
        if i > 0:
            table_df.iloc[i, table_df.columns.get_loc("D_pos")] = table_df['FracEsp'].iloc[i] - table_df['Frac'].iloc[i-1]
        else:
            table_df.iloc[i, table_df.columns.get_loc("D_pos")] = table_df['FracEsp'].iloc[i]
    
    # Retrieving the maximum D
    D = ( table_df[['D_neg','D_pos']].max() ).max()
    
    from scipy.stats import ksone
    def ks_critical_value(n_trials, alpha):
        return ksone.ppf(1-alpha/2, n_trials)
    
    # Retrieving p-value
    p_value = ks_critical_value(Data_df.shape[0], 0.05)
    
    # Computing result
    if D < p_value:
        print('Os dados seguem uma distribuição normal.')
    else:
        print('Os dados não seguem uma distribuição normal.')


def check_distribution(dist_names, y_std):
    
    p_values = []
    distance = []
    D_less_p = []
    
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)

        if distribution != "norm":
            D, p = scipy.stats.kstest(y_std, distribution, args=param)
        else:
            D, p = scipy.stats.kstest(y_std, distribution,  alternative='greater')
            
        #p = np.around(p, 5)
        p_values.append(p)    
        
        #D = np.around(D, 5)
        distance.append(D)    
        
        if D<p: 
            D_less_p.append("yes") 
        else: 
            D_less_p.append("no")

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['Distance'] = distance
    results['p_value'] = p_values
    results['D<p'] = D_less_p
    
    results.sort_values(['p_value'], ascending=False, inplace=True)


    print ('\nDistributions sorted by goodness of fit:')
    print ('----------------------------------------')
    print (results)