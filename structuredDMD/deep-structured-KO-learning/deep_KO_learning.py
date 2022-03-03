import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
print("Using PyTorch Version %s" %torch.__version__)
import os

class Net(nn.Module):

    ''' 
    This is a feedforward network taking snapshots of a dynamical system
    as input and spews two outputs. Output 1: KPsiXp, the featurized snapshot 
    data propogated forward by the trainable Koopman operator. 
    Output 2: PsiXf, the image of the featurized snapshot data.
    When the network is trained, the two outputs should be close in any distance
    '''
    
    def __init__(self, input_dim, output_dim, hl_sizes):
        super(Net, self).__init__()
        current_dim = input_dim
        self.linears = nn.ModuleList()
        for hl_dim in hl_sizes:
            self.linears.append(nn.Linear(current_dim, hl_dim))
            current_dim = hl_dim
        self.linears.append(nn.Linear(output_dim, output_dim,bias=False))

    def forward(self, x):
        input_vecs = x
        for layer in self.linears[:-1]:
            x = F.relu(layer(x))
        y = torch.cat((torch.Tensor(np.ones((x.shape[0],1))),input_vecs,x),dim=1)
        x = self.linears[-1](y)
        return {'KPsiXp':x,'PsiXf':y} 
    

if __name__ == '__main__': 

    script_dir = os.path.dirname('deep_KO_learning.py') # getting relative path
    trained_models_path = os.path.join(script_dir, 'trained_models') # which relative path do you want to see
    data_path = os.path.join(script_dir,'data/')

    save_network = 0
    net_name = '/toggle_switch'

    save_trainLoss_fig = 1
    figs_path = os.path.join(script_dir,'figures')

    ### Datasets ###

    dataset = 0 # each dataset contains the global snapshot matrix as well as the number of snapshots per trajectory and the number of trajectories

    if dataset == 0: 
        file_dir = 'toggle_switch_data.p'
        
    if dataset == 1: 
        file_dir = 'toggle_switch_data_normed.p'
        
    if dataset == 2: 
        file_dir = 'stable_linsys.p'
        
    if dataset == 3:
        file_dir = 'slow_manifold_data.p'
        
    if dataset == 4: 
        file_dir = 'slow_manifold_data_normed.p'
        
    if dataset == 5:
        file_dir = 'malathion_polyculture_pfluorescens_TPMs.p'

    if dataset == 6:
        file_dir = 'mt_poly_bt_TPMs.p'


    def get_snapshot_matrices(X,nT,nTraj): 
        '''This function assumes the global snapshot matrix is constructed with trajectories 
            sequentially placed in the columns'''
        prevInds = [x for x in range(0,nT-1)]
        forInds = [x for x in range(1,nT)]
        for i in range(0,nTraj-1):
            if i == 0:
                more_prevInds = [x + nT for x in prevInds]
                more_forInds = [x + nT for x in forInds]
            else: 
                more_prevInds = [x + nT for x in more_prevInds]
                more_forInds = [x + nT for x in more_forInds]
            prevInds = prevInds + more_prevInds
            forInds = forInds + more_forInds
        Xp = X[:,prevInds]
        Xf = X[:,forInds]
        return Xp,Xf

    X,nT,nTraj = pickle.load(open(data_path+file_dir,'rb'))
    Xp,Xf = get_snapshot_matrices(X,nT,nTraj)
    trainXp = torch.Tensor(Xp.T)
    trainXf = torch.Tensor(Xf.T)
    testX = torch.Tensor(X.T)

    numDatapoints = nT*nTraj # number of total snapshots

    print('Dimension of the state: ' + str(trainXp.shape[1]));
    print('Number of trajectories: ' + str(nTraj));
    print('Number of total snapshots: ' + str(nT*nTraj));


    ### Neural network parameters ###

    NUM_INPUTS = trainXp.shape[1] # dimension of input
    NUM_HL = 8 # number of hidden layers (excludes the input and output layers)
    NODES_HL = 8 # number of nodes per hidden layer (number of learned observables)
    HL_SIZES = [NODES_HL for i in range(0,NUM_HL+1)] 
    NUM_OUTPUTS = NUM_INPUTS + HL_SIZES[-1] + 1 # output layer takes in dimension of input + 1 + dimension of hl's
    BATCH_SIZE = 2 #int(nT/10) 

    net = Net(NUM_INPUTS,NUM_OUTPUTS,HL_SIZES)
    print(net)

    ### Defining the loss function and the optimizer ###

    LEARNING_RATE = 0.025
    L2_REG = 0.0
    MOMENTUM = 0.00

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=L2_REG)


    ### Training the network ###
    print_less_often = 2
    epoch_to_save_net = 25
    eps = 1e-10
    train_loss = []
    maxEpochs =  4 # 20000
    prev_loss = 0
    curr_loss = 1e10
    epoch = 0
    net.train()
    while (epoch <= maxEpochs): 

        if epoch % print_less_often == 0:
            if np.abs(prev_loss - curr_loss) < eps:
                break
            prev_loss = curr_loss

        for i in range(0,trainXp.shape[0],BATCH_SIZE):
            
            Kpsixp = net(trainXp[i:i+BATCH_SIZE])['KPsiXp'] 
            psixf = net(trainXf[i:i+BATCH_SIZE])['PsiXf']
            loss = loss_func(psixf, Kpsixp)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_loss = loss.item()

        if epoch % print_less_often == 0:
            print('['+str(epoch)+']'+' loss = '+str(loss.item()))
        train_loss.append(loss.item()) 
        epoch+=1

        if epoch % epoch_to_save_net == 0:
            print('Saving network at epoch ' + str(epoch))
            if save_network:
                pickle.dump([NUM_INPUTS,NUM_OUTPUTS,HL_SIZES],open(trained_models_path+net_name+'_netsize.pickle','wb'))
                torch.save(net.state_dict(), trained_models_path+net_name+'_net.pt') # saving the model state in ordered dict
        
    print('['+str(epoch)+']'+' loss = '+ str(loss.item()))
    ### Done training ### 

    ### Saving network (hyper)parameters ###
    if save_network:
        pickle.dump([NUM_INPUTS,NUM_OUTPUTS,HL_SIZES],open(trained_models_path+net_name+'_netsize.pickle','wb'))
        torch.save(net.state_dict(), trained_models_path+net_name+'_net.pt') # saving the model state in ordered dict

    ### Plotting the training loss ###
    import matplotlib.pyplot as plt;
    plt.rcParams.update({'font.size':14});
    plt.rcParams.update({'figure.autolayout': True})
    plt.semilogy(train_loss,lw=4);
    plt.ylabel('MSE Loss');
    plt.xlabel('Epoch');
    if save_trainLoss_fig:
        plt.savefig(figs_path+net_name+'_Loss.pdf')



