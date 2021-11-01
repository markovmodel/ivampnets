import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Mean_std_layer(nn.Module):
    """ Custom Linear layer for substracting the mean and dividing by the std
    
    Parameters
    ----------
    intput_size: int
        The input size.
    mean: torch.Tensor
        The mean values of all training points of the input features. Should have the size (1,intput_size)
    std: torch.Tensor
        The std values of all training points of the input features. Should have the size (1,intput_size)
    """
    def __init__(self, intput_size, mean=None, std=None):
        super().__init__()
        self.input_size = intput_size
        if mean is None:
            mean = torch.zeros((1,input_size))
        self.weights_mean = nn.Parameter(mean, requires_grad=False)  # nn.Parameter is a Tensor that's a module parameter.
        if std is None:
            std = torch.ones((1,input_size))
        self.weights_std = nn.Parameter(std, requires_grad=False)

    def forward(self, x):
        y = (x-self.weights_mean)/self.weights_std
        return y  
    
    def set_both(self, mean, std):
        new_params = [mean, std]
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                new_param = new_params[i]
                param.copy_(torch.Tensor(new_param[None,:]))


class Mask(torch.nn.Module):
    ''' Mask, which acts directly on the input features.
    
    Parameters
    ----------
    input_size: int
        Feature input size.
    N: int
        Number of subsystems.
    factor_fake: float, default=1.
        Factor how strong the fake subsystem is to take over input space. Makes the mask sparser for the real subsystems.
    noise: float, default=0.
        Regularize the masked by adding noise to the input. Therefore, the downstream lobes cannot recover inputs with low important weights.
        The larger the noise the stronger the weigh assignment of the mask will become.
    cutoff: float, must be between 0 and 1
        Cutoff after which an attention weight is set to zero. A total uninformative weight would be one, which is how
        the mask is initialized. 
    mean: torch.Tensor
        The mean values of all training points of the input features. Should have the size (1,intput_size)
    std: torch.Tensor
        The std values of all training points of the input features. Should have the size (1,intput_size)
    '''
    def __init__(self, input_size, N, factor_fake=1., noise=0., cutoff=0.9, 
                 mean=None, std=None, device='cpu'):
        super(Mask, self).__init__()
        self.input_size = input_size
        self.normalizer = Mean_std_layer(input_size, mean, std)
        self.factor_fake=factor_fake
        self.N = N
        list_weights = []
        for n in range(self.N):
            alpha = torch.ones((1, input_size, 1))
            weight = torch.nn.Parameter(data=alpha, requires_grad=True)
            list_weights.append(weight)
        self.list_weights = nn.ParameterList(list_weights)
        self.noise=noise
        self.cutoff = cutoff
        self.device=device
    def forward(self, x):
        ''' Applies the attention weights to all inputs and adds the defined noise. Furthermore, it
            normalizes the input to be approximately Gaussian.
        '''
        weight_sf = self.get_softmax()        
        prod = self.N + 1
        # first remove mean and std
        x = self.normalizer(x)
        
        masked_x = x[:,:,None] * weight_sf * prod# include factor
        
        if self.noise>0.:
            max_attention_value = torch.max(weight_sf, dim=1, keepdim=True)[0].detach()
            shape = weight_sf.shape
#             shape = (x.shape[0], weight_sf.shape[1], weight_sf.shape[2])
            random_numbers = torch.randn(shape, device=self.device) * self.noise
            masked_x += (1 - weight_sf/max_attention_value) * random_numbers
        
        # split them for each subsystem
        masked_list = torch.split(masked_x, 1, dim=2)
        return masked_list
    
    def get_softmax(self):
        ''' Estimates the attention weight for each input and subsystem.
        '''
        weights_all = []
        for param in self.list_weights:
            # first make a softmax over the input feature dimension to make them all positive
            weights_all.append(F.softmax(param, dim=1)*self.input_size) # the factor makes them on average around 1
        weights_per_N = torch.cat(weights_all, dim=2) # dim: 1 x input_size x N
        # add a fake subsystem
        fake_axis = torch.ones_like(self.list_weights[0])*self.factor_fake
        weights_per_N_fake = torch.cat([weights_per_N, fake_axis], dim=2)
        
        # normalize them along the subsystem axis
        weights_per_N_fake = torch.relu(weights_per_N_fake-self.cutoff) # set all to zero smaller cutoff
        weights_per_N_fake = weights_per_N_fake / torch.sum(weights_per_N_fake, dim=2, keepdims=True) # norm them to 1
        # remove fake axis
        weight_sf = weights_per_N_fake[:,:,:self.N]
                
        return weight_sf
    
    
class Mask_proteins(torch.nn.Module):
    ''' Mask, which acts on protein residue distances.
    
    Parameters
    ----------
    input_size: int
        Feature input size.
    N: int
        Number of subsystems.
    skip_res: int
        How many residues at the ends of the amino acid chain are neglected for distance calculation.
    patch_size: int
        Size of the window which slides over the acid chain.
    skip: int
        How many residues are skipped in each step of the window. It results in the fact that skip many residues have the same
        attention weight.
    factor_fake: float, default=1.
        Factor how strong the fake subsystem is to take over input space. Makes the mask sparser for the real subsystems.
    noise: float, default=0.
        Regularize the masked by adding noise to the input. Therefore, the downstream lobes cannot recover inputs with low important weights.
        The larger the noise the stronger the weigh assignment of the mask will become.
    cutoff: float, must be between 0 and 1
        Cutoff after which an attention weight is set to zero. A total uninformative weight would be one, which is how
        the mask is initialized. 
    mean: torch.Tensor
        The mean values of all training points of the input features. Should have the size (1,intput_size)
    std: torch.Tensor
        The std values of all training points of the input features. Should have the size (1,intput_size)
    '''
    def __init__(self, input_size, N, skip_res, patchsize, skip, factor_fake=3., 
                 noise=0., cutoff=0.5, mean=None, std=None, device='cpu'):
        super(Mask_proteins, self).__init__()
        
        self.device = device
        self.normalizer = Mean_std_layer(input_size, mean, std)
        self.noise = noise
        self.patchsize = patchsize
        self.skip = skip
        self.factor_fake = factor_fake    
        self.N = N
        self.cutoff = cutoff
        self.skip_res = skip_res
        self.n_residues = int(-1/2 + np.sqrt(1/4+input_size*2) + self.skip_res)
        print('Number of residues is: {}'.format(self.n_residues))
        self.residues_1 = []
        self.residues_2 = []
        
        self.nb_per_res = int(np.ceil(self.patchsize/self.skip)) # number of windows for each residue
        self.bs_per_res = np.empty((self.n_residues, self.nb_per_res), dtype=int)
        
        self.balance = (self.n_residues%self.skip)//2 # how much move the whole windows to make it symmetric at the ends
        for i in range(self.n_residues):
            start = (i+self.balance)//self.skip #within skip the same values
            self.bs_per_res[i] = np.arange(start, start+self.nb_per_res)
        
        self.number_weights = self.bs_per_res[-1,-1]+1
        # get the indexes of the residues which are part of the distances in the input
        for n1 in range(self.n_residues-self.skip_res):
            for n2 in range(n1+self.skip_res, self.n_residues):
                self.residues_1.append(n1)
                self.residues_2.append(n2)
                
        # initialize the weights you need for the windows.
        list_weights = []
        for n in range(self.N):
            alpha = torch.ones((1, self.number_weights, 1))
            
            weight = torch.nn.Parameter(data=alpha, requires_grad=True)
            list_weights.append(weight)
        self.list_weights = nn.ParameterList(list_weights)

        
    def forward(self, x):
        ''' Applies the attention weights of each residue to all distances and adds the defined noise. Furthermore, it
            normalizes the input to be approximately Gaussian.
        '''
        # first remove mean and std
        x = self.normalizer(x)
        # get the weights for each residue
        weights_for_res = self.get_softmax()
        
        prod = self.N + 1 # plus one because of the fake subsystem
        # get the weights for each input feature, due to the distance two residue weights
        weight_1 = weights_for_res[self.residues_1] * prod
        weight_2 = weights_for_res[self.residues_2] * prod
        alpha = weight_1[None,:,:] * weight_2[None,:,:]
        
        
        masked_x = x[:,:,None] * alpha 
        if self.noise>0.:
            max_attention_value = torch.max(alpha, dim=1, keepdim=True)[0].detach()
            shape = (x.shape[0], alpha.shape[1], alpha.shape[2]) # You should check again which one!
            # shape = alpha.shape
            random_numbers = torch.randn(shape, device=self.device) * self.noise
            masked_x += (1 - alpha/max_attention_value) * random_numbers
        # split them for each subsystem
        masked_list = torch.split(masked_x, 1, dim=2)
        return masked_list
    
    def get_softmax(self):
        ''' Estimates the attention weights for each residue for all subsystems.
        '''
        weights_per_N = []
        weights_all = []
        for param in self.list_weights:
            weights_all.append(param)
        
        for param in self.list_weights:
            # make them positive
            param = F.softmax(param, dim=1)*self.number_weights # this way on average 1
            weights_for_res = []
            for i in range(self.nb_per_res): # get all weights b for each residue

                weights_for_res.append(param[:,self.bs_per_res[:,i],:])

            # take the product of all windows involved for the same residue
            weights_for_res = torch.prod(torch.cat(weights_for_res, dim=0), dim=0) # take the product of the b factors
           
            weights_per_N.append(weights_for_res)  
        # Add the fake subsystem
        fake_axis = torch.ones_like(weights_per_N[0])*self.factor_fake
        weights_per_N = torch.cat(weights_per_N, dim=1)
        weights_per_N_fake = torch.cat([weights_per_N, fake_axis], dim=1)
        # normalize them along the subsystem axis
        weights_per_N_fake = torch.relu(weights_per_N_fake-self.cutoff) # set all to zero smaller 0.5
        weights_per_N_fake = weights_per_N_fake / torch.sum(weights_per_N_fake, dim=1, keepdims=True) # norm them to 1
        # remove the fake system
        weights_for_res = weights_per_N_fake[:,:self.N]
        
        return weights_for_res