import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import cvxpy 
from cvxpy import *
from  sklearn.preprocessing import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import sys

def snapshots_from_df(df):

    strains = ['wt','icar','phlf','nand']
    tps = ['5','18']
    temps = ['30','37']
    inducers = ['00','10','01','11']

    # create a dictionary where you specify strain, temp, and inducers as keys to grab the snapshot matrices
    snapshot_dict = {}
    for strain in strains: 
        snapshot_dict[strain] = {}
        for temp in temps: 
            snapshot_dict[strain][temp] = {}
            for inducer in inducers: 
                snapshot_dict[strain][temp][inducer] = {} # keys are to be Yf and Yp
                # get substring of colname that identifies the group (so everything except rep number)
                yp_colname = strain + '_' + inducer + temp + tps[0]
                # get list of indices that correspond to this group (in short, grabbing inds for all replicates)
                yp_col_inds = [ii for ii, this_col in enumerate(list(df.columns)) if yp_colname in this_col]
                snapshot_dict[strain][temp][inducer]['Yp'] = np.array(df.iloc[:,yp_col_inds])
                # do the same for the 18 hours timepoint i.e. Yf
                yf_colname = strain + '_' + inducer + temp + tps[1]
                yf_col_inds = [ii for ii, this_col in enumerate(list(df.columns)) if yf_colname in this_col]
                snapshot_dict[strain][temp][inducer]['Yf'] = np.array(df.iloc[:,yf_col_inds])
                
    return snapshot_dict

def get_unpaired_samples(df):
    # filter the samples that don't have a timepoint pair due to low sequencing depth
    unpaired_samples = []
    for sample in df.columns: 
        if '5' in sample:
            if sample.replace('5','18') not in df.columns:
                unpaired_samples.append(sample)
        elif '18' in sample: 
            if sample.replace('18','5') not in df.columns:
                unpaired_samples.append(sample)
    return unpaired_samples

def apply_normalizer(Yp,Yf):
    # normalize each datapoint to have unit norm
    transformer1 = Normalizer().fit(Yp.T)
    Yp_normed = transformer1.transform(Yp.T).T

    transformer2 = Normalizer().fit(Yf.T)
    Yf_normed = transformer2.transform(Yf.T).T
    return Yp_normed,Yf_normed

def calc_Koopman(Yf,Yp,flag=1,lambda_val=0.0,noise_scaler=1,verbose=True):
    ngenes = Yf.shape[0]
    ndatapts = Yf.shape[1]
    
    if flag == 1: # least-squares solution
        Yp_inv = np.linalg.pinv(Yp)
        K = np.dot(Yf,Yp_inv)
        print('The mean squared error is: ' + '{:0.3e}'.format(np.linalg.norm(K@Yp - Yf)**2 / ndatapts)) 
        # TO DO: Add SVD based DMD for modal decompostion (modes can be insightful)
        
    if flag == 2: # robust optimization approach
        solver_instance = cvxpy.SCS
        operator = Variable(shape=(ngenes,ngenes)) # Koopman matrix  K
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        print("[INFO]: Shape of Yf and Yp: " + repr(Yf.shape) + ', ' + repr(Yp.shape) )
        
        if type(lambda_val) == float:
            reg_term = lambda_val * cvxpy.norm(operator,p=1) 
        else:
            lambda_val = lambda_val[:,np.newaxis]
            Unoise = np.tile(lambda_val,ndatapts)
            reg_term = cvxpy.norm(cvxpy.matmul(operator + np.eye(ngenes),noise_scaler*Unoise),p='fro') # where exactly does this term come from?  

        norm2_fit_term = cvxpy.norm(Yf - cvxpy.matmul(operator,Yp),p=2)
        
        objective = Minimize(norm2_fit_term + reg_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=int(1e6))
        K = operator.value
        
        print("[INFO]: CVXPY problem status: " + prob.status)
        print('MSE =  ' + '{:0.3e}'.format(np.linalg.norm(K@Yp - Yf)**2 / ndatapts)) 
        print('\n','\n')

    return K

def calc_input_Koopman(LHS,Up,flag=1,lambda_val=0.0,noise_scaler=1,verbose=True):
    ''' From Yf=AYp+BUp or Yf-AYp=BUp (A known) or Yf-(AYp + B_1U1 + ... )=B_JUJ (B_1 to B_J-1 known), we see 
        that the optimization can always be written as ||loss_lhs - operator*Up||_2 + ||G(operator)||_F.
    '''
    ngenes = LHS.shape[0]
    ndatapts = LHS.shape[1]
    ninputs = Up.shape[0]

    if flag == 1: # least-squares solution
        Up_inv = np.linalg.pinv(Up)
        Ki = np.dot(LHS,Up_inv)
        print('The mean squared error is: ' + '{:0.3e}'.format(np.linalg.norm(LHS - Ki@Up)**2 / ndatapts)) 
        # TO DO: Add SVD based DMD for modal decompostion (modes can be insightful)
        
    if flag == 2: # robust optimization approach
        solver_instance = cvxpy.SCS
        operator = Variable(shape=(ngenes,ninputs)) # Koopman matrix  K
        
        print("[INFO]: CVXPY Koopman operator variable: " + repr(operator.shape))
        
        if type(lambda_val) == float:
            reg_term = lambda_val * cvxpy.norm(operator,p=1) 
        else:
            lambda_val = lambda_val[:,np.newaxis]
            Unoise = np.tile(lambda_val,ndatapts)
            reg_term = cvxpy.norm(cvxpy.matmul(operator,noise_scaler*Unoise),p='fro') # where exactly does this term come from? 

        norm2_fit_term = cvxpy.norm(LHS - cvxpy.matmul(operator,Up),p=2) 
        objective = Minimize(norm2_fit_term + reg_term)
        constraints = []
        prob = Problem(objective,constraints)
        result = prob.solve(verbose=verbose,solver=solver_instance,max_iters=int(1e6))
        Ki = operator.value

        print("[INFO]: CVXPY problem status: " + prob.status)
        print('MSE =  ' + '{:0.3e}'.format(np.linalg.norm(LHS - Ki@Up)**2 / ndatapts)) 
        print('\n','\n')

    return Ki

def get_ara_lac_genes(all_genes_filter):
    '''
    all_genes_filter: list of gene names in same order as df_tpm_filter
    '''
    lac_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if 'lac' in this_gene]
    ara_inds = [ii for ii,this_gene in enumerate(all_genes_filter) if 'ara' in this_gene]
    lac_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if 'lac' in this_gene]
    ara_genes = [this_gene for ii,this_gene in enumerate(all_genes_filter) if 'ara' in this_gene]
    my_genes = ara_genes + lac_genes
    my_inds = ara_inds + lac_inds
    return my_genes, my_inds

def get_DE_genes(res_dir_list,all_genes_filter,p_thresh=0.05,fc_thresh=2): 
    '''
    res_dir_list: list of directories for where to find DE results from DESeq2    

    '''
    res_list = []
    for res_dir in res_dir_list:  
        res_df = pd.read_csv(res_dir)
        # first filter by padj
        filter_p = res_df.loc[res_df.padj <  p_thresh]
        # next by FC
        res_filter_df  = filter_p.loc[np.abs(filter_p.log2FoldChange)>=\
                                                    np.log2(fc_thresh)]
        res_list.append(res_filter_df)

    # take the union of the genes remaining in each condition
    genes_DE = set()
    for res in res_list: 
        genes_DE = genes_DE.union(res.iloc[:,0]) # the first row contains gene names
    genes_DE = list(genes_DE)

    my_inds = [ii for ii,this_gene in enumerate(all_genes_filter) \
               for jj,DE_gene in enumerate(genes_DE) if  DE_gene == this_gene]
    my_genes = [all_genes_filter[ii] for ii in my_inds]

    return my_genes, my_inds


# Load TPM or GeTMM dataframe (results are identical, implying GeTMM normalization is unnecessary for our dataset)
df_tpm_filter = pd.read_csv('data/TPM_matrix_NAND.csv') # suffix is 'filter' because very low count genes have already been removed
# get list of samples that are to be removed because they don't have a timepoint pair, 
# i.e. a col in Xp does not have corresponding col in Xf or vice versa
unpaired_samples = get_unpaired_samples(df_tpm_filter)
# get gene names
all_genes_filter =  df_tpm_filter['gene'] 
# remove 'gene' col from df
df_tpm_filter = df_tpm_filter.iloc[:,1:]
# drop the unpaired samples
df_tpm_filter = df_tpm_filter.drop(columns=unpaired_samples).reset_index(drop=True)

# path to deseq results for gene downselection 
res_dir_list = ['data/condition_nand00375_vs_wt00375_results.csv',\
                'data/condition_nand003718_vs_wt003718_results.csv']

# get all snapshots keyed in the following way: 
# [strain][temperature][inducer][snapshots]
snapshot_dict = snapshots_from_df(df_tpm_filter)

# get circuit gene names and indices
circuit_inds = list(range(13)) # circuit genes are in first 13 rows of df
circuit_genes = all_genes_filter[:13]

# select genes for modeling
ara_lac = False # 13 ara and lac genes
de_only = True # only differentially expressed genes

# get host gene names and indices
if ara_lac:
    p_thresh, fc_thresh = 'n/a', 'n/a'
    my_genes,  my_inds  = get_ara_lac_genes(all_genes_filter) # only ara and lac genes, want to create more smaller networks like this:
    selected_genes = 'ara_lac'
# see: https://www.weizmann.ac.il/mcb/UriAlon/e-coli-transcription-network#:~:text=E.-,coli%20transcription%20network,in%20cells%20orchestrate%20gene%20expression.&text=Each%20network%20motif%20has%20a,responses%20to%20fluctuating%20external%20signals.
elif de_only: 
    p_thresh, fc_thresh = 0.05, 2.0
    my_genes, my_inds = get_DE_genes(res_dir_list,all_genes_filter,p_thresh=p_thresh,fc_thresh=fc_thresh)
    selected_genes = 'de_only'

# Structured learning sequence:
# 0) Learn host dynamics from wild type 
# 1) Learn single inducer impact on host (arabinose and iptg separate)
# 2) Learn control matrix that accounts for PhlF gate + arabinose dynamics
# 3) Learn control matrix that accounts for IcaR gate + IPTG dynamics
# 4) Learn control matrix that accounts for NAND circuit + arabinose + iptg + phlf + nand

TEMP = '30'
NOISE_SCALER = 100 # parameter that controls sparsity
VERBOSE = True
# for large number of genes, heatmap visualization is cumbersome
doVisualizeHeatmaps = False
# save state-space model
doSave = True
fn = 'x8_NAND' # name to save pickle file as, will also be used as id for run

# ensure unique filename/id is being used for the run
prev_ids = pd.read_csv('data/run_log.csv')['id']
if fn in list(prev_ids):
    sys.exit("The id (fn) being used already exists in run_log.csv. Provide a unique name and retry.")

################# Wild type dynamics ###################################
Yp = snapshot_dict['wt'][TEMP]['00']['Yp']
Yf = snapshot_dict['wt'][TEMP]['00']['Yf']
# normalize each snapshot to have unit norm, a dynamics preserving transformation
Yp_normed, Yf_normed = apply_normalizer(Yp,Yf)
Yp_normed, Yf_normed = Yp_normed[my_inds], Yf_normed[my_inds]
# compute the intrareplicate noise for regularizing K as motivated by Robust DMD
lambda_val_vec_p = np.std(Yp_normed,axis=1)[:,np.newaxis]
lambda_val_vec_f = np.std(Yf_normed,axis=1)[:,np.newaxis]
lambda_val_vec = np.std(np.hstack((lambda_val_vec_p,lambda_val_vec_f)),axis=1)
K = calc_Koopman(Yf_normed,Yp_normed,flag=2,lambda_val=lambda_val_vec,\
    noise_scaler=NOISE_SCALER,verbose=VERBOSE)

################# WT + arabinose dynamics ###################################
Yp = snapshot_dict['wt'][TEMP]['10']['Yp']
Yf = snapshot_dict['wt'][TEMP]['10']['Yf']
# normalize each snapshot to have unit norm, a dynamics preserving transformation
Yp_normed, Yf_normed = apply_normalizer(Yp,Yf)
Yp_normed, Yf_normed = Yp_normed[my_inds], Yf_normed[my_inds]
# treat arabinose as a step input to the system 
Uara = (Yp_normed.max() - Yf_normed.min())/2 * np.ones((1,Yp_normed.shape[1]))
# the scaling coefficient is to have mag of step be on same scale as the data
# compute the intrareplicate noise for regularizing K as motivated by Robust DMD
lambda_val_vec = np.std(Uara,axis=1)
# form the LHS of the optimization problem
LHS = Yf_normed  - K@Yp_normed
Kara = calc_input_Koopman(LHS,Uara,flag=2,lambda_val=lambda_val_vec,\
    noise_scaler=NOISE_SCALER,verbose=VERBOSE)

################# WT + IPTG dynamics ###################################
Yp = snapshot_dict['wt'][TEMP]['01']['Yp']
Yf = snapshot_dict['wt'][TEMP]['01']['Yf']
# normalize each snapshot to have unit norm, a dynamics preserving transformation
Yp_normed, Yf_normed = apply_normalizer(Yp,Yf)
Yp_normed, Yf_normed = Yp_normed[my_inds], Yf_normed[my_inds]
# treat arabinose as a step input to the system 
Uiptg = (Yp_normed.max() - Yf_normed.min())/2 * np.ones((1,Yp_normed.shape[1]))
# the scaling coefficient is to have mag of step be on same scale as the data
# compute the intrareplicate noise for regularizing K as motivated by Robust DMD
lambda_val_vec = np.std(Uiptg,axis=1)
# form the LHS of the optimization problem
LHS = Yf_normed  - K@Yp_normed
Kiptg = calc_input_Koopman(LHS,Uiptg,flag=2,lambda_val=lambda_val_vec,\
    noise_scaler=NOISE_SCALER,verbose=VERBOSE)

################# PhlF Gate + arabinose dynamics ###################################
Yp = snapshot_dict['phlf'][TEMP]['01']['Yp']
Yf = snapshot_dict['phlf'][TEMP]['01']['Yf']
# normalize each snapshot to have unit norm, a dynamics preserving transformation
Yp_normed, Yf_normed = apply_normalizer(Yp,Yf)
Uphlf_p, Uphlf_f = Yp_normed[circuit_inds], Yf_normed[circuit_inds]
Yp_normed, Yf_normed = Yp_normed[my_inds], Yf_normed[my_inds]
# compute the intrareplicate noise for regularizing K as motivated by Robust DMD
lambda_val_vec_p = np.std(Uphlf_p,axis=1)[:,np.newaxis]
lambda_val_vec_f = np.std(Uphlf_f,axis=1)[:,np.newaxis]
lambda_val_vec = np.std(np.hstack((lambda_val_vec_p,lambda_val_vec_f)),axis=1)
# if number of samples in Yp/Yf do not match the number of  samples in the 
# previously used step inputs, simply downselect the step input as we do here
# form the LHS of the optimization problem
LHS = Yf_normed  - K@Yp_normed - Kara@Uara[:,:-1]
Kphlf = calc_input_Koopman(LHS,Uphlf_p,flag=2,lambda_val=lambda_val_vec,\
    noise_scaler=NOISE_SCALER,verbose=VERBOSE)

################# IcaR Gate + IPTG dynamics ###################################
Yp = snapshot_dict['icar'][TEMP]['01']['Yp']
Yf = snapshot_dict['icar'][TEMP]['01']['Yf']
# normalize each snapshot to have unit norm, a dynamics preserving transformation
Yp_normed, Yf_normed = apply_normalizer(Yp,Yf)
Uicar_p, Uicar_f = Yp_normed[circuit_inds], Yf_normed[circuit_inds]
Yp_normed, Yf_normed = Yp_normed[my_inds], Yf_normed[my_inds]
# compute the intrareplicate noise for regularizing K as motivated by Robust DMD
lambda_val_vec_p = np.std(Uicar_p,axis=1)[:,np.newaxis]
lambda_val_vec_f = np.std(Uicar_f,axis=1)[:,np.newaxis]
lambda_val_vec = np.std(np.hstack((lambda_val_vec_p,lambda_val_vec_f)),axis=1)
# if number of samples in Yp/Yf do not match the number of  samples in the 
# previously used step inputs, simply downselect the step input as we do here
# this will only be an issue with the manually defined step inputs
# form the LHS of the optimization problem
if TEMP == '37':
    LHS = Yf_normed  - K@Yp_normed - Kiptg@Uiptg[:,:-1]
elif TEMP == '30':
    LHS = Yf_normed  - K@Yp_normed - Kiptg@np.hstack((Uiptg,Uiptg[:,0:1]))
Kicar = calc_input_Koopman(LHS,Uicar_p,flag=2,lambda_val=lambda_val_vec,\
    noise_scaler=NOISE_SCALER,verbose=VERBOSE)

########## NAND Circuit + arabinose + IPTG + Phlf Gate + Icar Gate dynamics ###################################
Yp = snapshot_dict['nand'][TEMP]['11']['Yp']
Yf = snapshot_dict['nand'][TEMP]['11']['Yf']
# normalize each snapshot to have unit norm, a dynamics preserving transformation
Yp_normed, Yf_normed = apply_normalizer(Yp,Yf)
Unand_p, Unand_f = Yp_normed[circuit_inds], Yf_normed[circuit_inds]
Yp_normed, Yf_normed = Yp_normed[my_inds], Yf_normed[my_inds]
# compute the intrareplicate noise for regularizing K as motivated by Robust DMD
lambda_val_vec_p = np.std(Unand_p,axis=1)[:,np.newaxis]
lambda_val_vec_f = np.std(Unand_f,axis=1)[:,np.newaxis]
lambda_val_vec = np.std(np.hstack((lambda_val_vec_p,lambda_val_vec_f)),axis=1)
# form the LHS of the optimization problem
if TEMP == '37':
    LHS = Yf_normed  - K@Yp_normed - Kiptg@Uiptg - Kara@Uara  - \
        Kphlf@Unand_p - Kicar@Unand_p
elif TEMP == '30':
    LHS = Yf_normed  - K@Yp_normed - Kiptg@Uiptg \
        - Kara@Uara[:,:-1]  - Kphlf@Unand_p - Kicar@Unand_p
Knand = calc_input_Koopman(LHS,Unand_p,flag=2,lambda_val=lambda_val_vec,\
    noise_scaler=NOISE_SCALER,verbose=VERBOSE)


if doVisualizeHeatmaps:
    # visualizing state-space model


    plt.figure(figsize=(22,22))
    sns.heatmap(K,center=0)

    # fig,axs = plt.subplots(2,3,figsize=(22,22))
    # LW = 0.2
    # sns.heatmap(K,square=True,center=0,cbar_kws={'shrink':0.8,'extend':'both'},\
    #     ax=axs[0,0])
    # sns.heatmap(Kara,square=True,center=0,cbar_kws={'shrink':0.8,'extend':'both'},\
    #     ax=axs[0,1])
    # sns.heatmap(Kiptg,square=True,center=0,cbar_kws={'shrink':0.8,'extend':'both'},\
    #     ax=axs[0,2])
    # sns.heatmap(Kphlf,square=True,center=0,cbar_kws={'shrink':0.8,'extend':'both'},\
    #     ax=axs[1,0])
    # sns.heatmap(Kicar,square=True,center=0,cbar_kws={'shrink':0.8,'extend':'both'},\
    #     ax=axs[1,1])
    # sns.heatmap(Knand,square=True,center=0,cbar_kws={'shrink':0.8,'extend':'both'},\
    #     ax=axs[1,2])
    # plt.tight_layout()
    plt.show()

if doSave: 
    pickle.dump([K,Kara,Kiptg,Kphlf,Kicar,Knand,my_genes,my_inds],open('data/'+fn+'.pkl','wb'))
    df = pd.DataFrame({'id':[fn],'n_genes':[len(my_genes)],'padj_thresh':[p_thresh],'fc_thresh':[fc_thresh],\
                        'selected_genes':selected_genes,'noise_scaler':NOISE_SCALER,'temp':TEMP})
    # before writing to run_log, need to start newline
    with open('data/run_log.csv', 'a') as f:
        f.write('\n')
    df.to_csv('data/run_log.csv', mode='a', index=False, header=False)





