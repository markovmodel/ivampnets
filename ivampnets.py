from typing import Optional, Union, Callable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import chain


from deeptime.base import Model, Transformer
from deeptime.base_torch import DLEstimatorMixin
from deeptime.util.torch import map_data
from deeptime.markov.tools.analysis import pcca_memberships

CLIP_VALUE = 1.

def _inv(x, return_sqrt=False, epsilon=1e-6):
    '''Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.
    Parameters
    ----------
    x: numpy array with shape [m,m]
        matrix to be inverted

    ret_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    '''

    # Calculate eigvalues and eigvectors
    eigval_all, eigvec_all = torch.symeig(x, eigenvectors=True)
#     eigval_all, eigvec_all =  torch.linalg.eigh(x, UPLO='U')
    # Filter out eigvalues below threshold and corresponding eigvectors
#     eig_th = torch.Tensor(epsilon)
    index_eig = eigval_all > epsilon
#     print(index_eig)
    eigval = eigval_all[index_eig]
    eigvec = eigvec_all[:,index_eig]

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag = torch.diag(torch.sqrt(1/eigval))
    else:
        diag = torch.diag(1/eigval)
#     print(diag.shape, eigvec.shape)    
    # Rebuild the square root of the inverse matrix
    x_inv = torch.matmul(eigvec, torch.matmul(diag, eigvec.T))

    return x_inv


def symeig_reg(mat, epsilon: float = 1e-6, mode='regularize', eigenvectors=True) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r""" Solves a eigenvector/eigenvalue decomposition for a hermetian matrix also if it is rank deficient.

    Parameters
    ----------
    mat : torch.Tensor
        the hermetian matrix
    epsilon : float, default=1e-6
        Cutoff for eigenvalues.
    mode : str, default='regularize'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    eigenvectors : bool, default=True
        Whether to compute eigenvectors.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, Optional[torch.Tensor]]
        Eigenvalues and -vectors.
    """
    assert mode in sym_inverse.valid_modes, f"Invalid mode {mode}, supported are {sym_inverse.valid_modes}"

    if mode == 'regularize':
        identity = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        mat = mat + epsilon * identity

    # Calculate eigvalues and potentially eigvectors
    eigval, eigvec = torch.symeig(mat, eigenvectors=True)
#     eigval, eigvec = torch.linalg.eigh(mat, UPLO='U')

    if eigenvectors:
        eigvec = eigvec.transpose(0, 1)

    if mode == 'trunc':
        # Filter out Eigenvalues below threshold and corresponding Eigenvectors
        mask = eigval > epsilon
        eigval = eigval[mask]
        if eigenvectors:
            eigvec = eigvec[mask]
    elif mode == 'regularize':
        # Calculate eigvalues and eigvectors
        eigval = torch.abs(eigval)
    elif mode == 'clamp':
        eigval = torch.clamp_min(eigval, min=epsilon)

    else:
        raise RuntimeError("Invalid mode! Should have been caught by the assertion.")

    if eigenvectors:
        return eigval, eigvec
    else:
        return eigval, eigvec


def sym_inverse(mat, epsilon: float = 1e-6, return_sqrt=False, mode='regularize', return_both=False):
    """ Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    mat: numpy array with shape [m,m]
        Matrix to be inverted.
    epsilon : float
        Cutoff for eigenvalues.
    return_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead
    mode: str, default='trunc'
        Whether to truncate eigenvalues if they are too small or to regularize them by taking the absolute value
        and adding a small positive constant. :code:`trunc` leads to truncation, :code:`regularize` leads to epsilon
        being added to the eigenvalues after taking the absolute value
    return_both: bool, default=False
        Whether to return the sqrt and its inverse or simply the inverse
    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    """
    if mode=='old':
        return _inv(mat, epsilon=epsilon, return_sqrt=return_sqrt)
    eigval, eigvec = symeig_reg(mat, epsilon, mode)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    if return_sqrt:
        diag_inv = torch.diag(torch.sqrt(1. / eigval))
        if return_both:
            diag = torch.diag(torch.sqrt(eigval))
    else:
        diag_inv = torch.diag(1. / eigval)
        if return_both:
            diag = torch.diag(eigval)
    if not return_both:
        return eigvec.t() @ diag_inv @ eigvec
    else:
        return eigvec.t() @ diag_inv @ eigvec, eigvec.t() @ diag @ eigvec


sym_inverse.valid_modes = ('trunc', 'regularize', 'clamp', 'old')


def covariances(x: torch.Tensor, y: torch.Tensor, remove_mean: bool = True):
    """Computes instantaneous and time-lagged covariances matrices.

    Parameters
    ----------
    x : (T, n) torch.Tensor
        Instantaneous data.
    y : (T, n) torch.Tensor
        Time-lagged data.
    remove_mean: bool, default=True
        Whether to remove the mean of x and y.

    Returns
    -------
    cov_00 : (n, n) torch.Tensor
        Auto-covariance matrix of x.
    cov_0t : (n, n) torch.Tensor
        Cross-covariance matrix of x and y.
    cov_tt : (n, n) torch.Tensor
        Auto-covariance matrix of y.

    See Also
    --------
    deeptime.covariance.Covariance : Estimator yielding these kind of covariance matrices based on raw numpy arrays
                                     using an online estimation procedure.
    """

    assert x.shape == y.shape, "x and y must be of same shape"
    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    # Calculate the cross-covariance
    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    # Calculate the auto-correlations
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


valid_score_methods = ('VAMP1', 'VAMP2', 'VAMPE')

def VAMPE_score(chi_t, chi_tau, epsilon=1e-6, mode='regularize'):
    '''Calculates the VAMPE score for an individual VAMPnet model. Furthermore, it returns
    the singular functions and singular values to construct the global operator.
    Parameters
    ----------
    chi_t: (T, n) torch.Tensor
        Instantaneous data with shape batchsize x outputsize.
    chi_tau: (T, n) torch.Tensor
        Time-lagged data with shape batchsize x outputsize.

    Returns
    -------
    score: (1) torch.Tensor
        VAMPE score.
    S: (n, n) torch.Tensor
        Singular values on the diagonal of the matrix.
    u: (n, n) torch.Tensor
        Left singular functions.
    v: (n, n) torch.Tensor
        Right singular functions.
    trace_cov: (1) torch.Tensor
        Sum of eigenvalues of the covariance matrix.
    '''
    shape = chi_t.shape
        
    batch_size = shape[0]

    x, y = chi_t, chi_tau
    
    # Calculate the covariance matrices
    cov_00 = 1/(batch_size) * torch.matmul(x.T, x) 
    cov_11 = 1/(batch_size) * torch.matmul(y.T, y)
    cov_01 = 1/(batch_size) * torch.matmul(x.T, y)

    # Calculate the inverse of the self-covariance matrices
    cov_00_inv = sym_inverse(cov_00, return_sqrt = True, epsilon=epsilon, mode=mode)
    cov_11_inv = sym_inverse(cov_11, return_sqrt = True, epsilon=epsilon, mode=mode)

    # Estimate Vamp-matrix
    K = torch.matmul(cov_00_inv, torch.matmul(cov_01, cov_11_inv))
    # Estimate the singular value decomposition
    a, sing_values, b = torch.svd(K, compute_uv=True)
    # Estimate the singular functions
    u = cov_00_inv @ a
    v = cov_11_inv @ b
    S = torch.diag(sing_values)
    # Estimate the VAMPE score
    term1 = 2* S @ u.T @ cov_01 @ v
    term2 = S @ u.T @ cov_00 @ u @ S @ v.T @ cov_11 @ v
    
    score = torch.trace(term1 - term2)
    # expand zero dimension for summation
    score = torch.unsqueeze(score, dim=0)
    
    # estimate sum of eigenvalues of the covariance matrix to enforce harder assignment
    trace_cov = torch.unsqueeze(torch.trace(cov_00), dim=0)
    
    return score, S, u, v, trace_cov


def VAMPE_score_pair(chi1_t, chi1_tau, chi2_t, chi2_tau, S1, S2, u1, u2, v1, v2, device='cpu'):
    '''Calculates the VAMPE score for a pair of individual VAMPnet models. The operator is constructed
    by the two individual operators and evaluate on the outer space constructed by the individual
    feature functions.
    Parameters
    ----------
    chi1_t: (T, n) torch.Tensor
        Instantaneous data with shape batchsize x outputsize of VAMPnet 1.
    chi1_tau: (T, n) torch.Tensor
        Time-lagged data with shape batchsize x outputsize of VAMPnet 1.
    chi2_t: (T, m) torch.Tensor
        Instantaneous data with shape batchsize x outputsize of VAMPnet 2.
    chi2_tau: (T, m) torch.Tensor
        Time-lagged data with shape batchsize x outputsize of VAMPnet 2.      
    S1: (n, n) torch.Tensor
        Singular values of VAMPnet 1.      
    S2: (m, m) torch.Tensor
        Singular values of VAMPnet 2.   
    u1: (n, n) torch.Tensor
        Left singular functions of VAMPnet 1. 
    u2: (m, m) torch.Tensor
        Left singular functions of VAMPnet 2.     
    v1: (n, n) torch.Tensor
        Right singular functions of VAMPnet 1.    
    v2: (m, m) torch.Tensor
        Right singular functions of VAMPnet 2.

    Returns
    -------
    score: (1) torch.Tensor
        VAMPE score for the performance of the constructed operator on the global features.
    pen_C00: (1) torch.Tensor
        Error of the left singular functions.
    pen_C11: (1) torch.Tensor
        Error of the right singular functions.
    pen_C01: (1) torch.Tensor
        Error of the correlation of the two singular functions.
    '''
    
    
    shape1 = chi1_t.shape
    shape2 = chi2_t.shape  
    new_shape = shape1[1] * shape2[1]
    batch_size = shape1[0]

    # construct the singular functions for the global model from both individual subsystems
    U_train = torch.reshape(u1[:,None,:,None] * u2[None,:,None,:], (new_shape, new_shape))
    V_train = torch.reshape(v1[:,None,:,None] * v2[None,:,None,:], (new_shape, new_shape))
    K_train = torch.reshape(S1[:,None,:,None] * S2[None,:,None,:], (new_shape, new_shape))
    # construct the global feature space as the outer product of the individual spaces
    chi_t_outer = torch.reshape(chi1_t[:,:,None] * chi2_t[:,None,:], (batch_size,new_shape))
    chi_tau_outer = torch.reshape(chi1_tau[:,:,None] * chi2_tau[:,None,:], (batch_size,new_shape))

    x, y = chi_t_outer, chi_tau_outer
    # Calculate the covariance matrices
    cov_00 = 1/(batch_size) * torch.matmul(x.T, x) 
    cov_11 = 1/(batch_size) * torch.matmul(y.T, y)
    cov_01 = 1/(batch_size) * torch.matmul(x.T, y)

    # map the matrices on the singular functions
    
    C00_map = U_train.T @ cov_00 @ U_train
    C11_map = V_train.T @ cov_11 @ V_train
    C01_map = U_train.T @ cov_01 @ V_train
    # helper function to estimate errors from optimal solution
    unit_matrix = torch.eye(new_shape, device=device)
    # Estimate the deviation from the optimal behaviour if the two system would be truly independent
    pen_C00 = torch.unsqueeze(torch.sum(torch.abs(unit_matrix - C00_map)), dim=0) / (new_shape-1)**2
    pen_C11 = torch.unsqueeze(torch.sum(torch.abs(unit_matrix - C11_map)), dim=0) / (new_shape-1)**2
    pen_C01 = torch.unsqueeze(torch.sum(torch.abs(C01_map - K_train)), dim=0) / (new_shape-1)**2
    # Estimate the VAMPE score of how well the constructed operator predicts the dynamic of the global feature space
    term1 = 2 * K_train @ C01_map
    term2 = K_train @ C00_map @ K_train @ C11_map
    
    score = torch.trace(term1 - term2)
    # add zero dimension for summation
    score = torch.unsqueeze(score, dim=0)
    
    return score, pen_C00, pen_C11, pen_C01

def score_loss(score1, score2, score12):
    ''' Estimates the discrepancy of the global score and the two individual VAMPE scores.
    Parameters
    ----------
    score1: (1) torch.Tensor
        Score of VAMPnets 1.
    score2: (1) torch.Tensor
        Score of VAMPnets 2.
    score12: (1) torch.Tensor
        Score of the constructed global operator.
    
    Returns
    -------
    pen_scores: (1) torch.Tensor
        Error of the scores due to non independent behavior.
    
    '''
    prod_score = score1 * score2
    # Estimate normalizer to rescale them but not use them for gradient updates.
    norm1 = torch.abs(prod_score.detach())
    norm2 = torch.abs(score12.detach())
    diff = torch.abs(score12 - prod_score)
    score_diff = torch.unsqueeze(diff / norm1, dim=0)
    score_diff2 = torch.unsqueeze(diff / norm2, dim=0)
    pen_scores = (score_diff + score_diff2) / 2.
    
    return pen_scores

def score_all_systems(chi_t_list, chi_tau_list, epsilon=1e-6, mode='regularize'):
    ''' Estimates all scores and singular functions/values for all VAMPnets.
    Parameters
    ----------
    chi_t_list: list of length number subsystems
        List of the feature functions of all VAMPnets for the instantaneous data.
    chi_tau_list: list of length number subsystems
        List of the feature functions of all VAMPnets for the time-lagged data.
        
    Returns
    -------
    scores_single: list of length number subsystems
        List of the individual scores of all subsystems.
    S_single: list of length number subsystems
        List of the individual singular values of all subsystems.
    u_single: list of length number subsystems
        List of the individual left singular functions of all subsystems.
    v_single: list of length number subsystems
        List of the individual right singular functions of all subsystems.
    trace_single: list of length number subsystems
        List of the traces of covariance matrices of all subsystems.
    '''
    scores_single = []
    u_single = []
    v_single = []
    S_single = []
    trace_single = []
    N = len(chi_t_list)
    for i in range(N):
        
        chi_i_t = chi_t_list[i]
        chi_i_tau = chi_tau_list[i]

        score_i, S_i , u_i, v_i, trace_i = VAMPE_score(chi_i_t, chi_i_tau, epsilon=epsilon, mode=mode)
        scores_single.append(score_i)
        trace_single.append(trace_i)
        u_single.append(u_i)
        v_single.append(v_i)
        S_single.append(S_i)

    return scores_single, S_single, u_single, v_single, trace_single

def score_all_outer_systems(chi_t_list, chi_tau_list, S_list, u_list, v_list, device='cpu'):
    ''' Estimates the global scores and all penalties.
    Parameters
    ----------
    chi_t_list: list of length number subsystems
        List of the feature functions of all VAMPnets for the instantaneous data.
    chi_tau_list: list of length number subsystems
        List of the feature functions of all VAMPnets for the time-lagged data.
    S_list: list of length number subsystems
        List of the individual singular values of all subsystems.
    u_list: list of length number subsystems
        List of the individual left singular functions of all subsystems.
    v_list: list of length number subsystems
        List of the individual right singular functions of all subsystems.
    
    Returns
    -------
    scores_outer: list of length N*(N-1)/2, where N is the number of subsystems
        List of global scores of all pairs of VAMPnets.
    pen_C00_map: list of length number pairs.
        List of penalties of singular left functions.
    pen_C11_map: list of length number pairs.
        List of penalties of singular right functions.
    pen_C01_map: list of length number pairs.
        List of penalties of the correlation of the singular functions.
    '''
    scores_outer = []
    pen_C00_map = []
    pen_C11_map = []
    pen_C01_map = []
    # N_pair = len(scores_outer)
    # N = int(0.5+np.sqrt(0.25+2*N_pair))
    N = len(chi_t_list)
    for i in range(N):
        
        for j in range(i+1,N):
            
            score_ij, pen_C00_ij, pen_C11_ij, pen_C01_ij = VAMPE_score_pair(
                                        chi_t_list[i], chi_tau_list[i], 
                                        chi_t_list[j], chi_tau_list[j], 
                                        S_list[i], S_list[j],
                                        u_list[i], u_list[j],
                                        v_list[i], v_list[j],
                                        device=device)
            scores_outer.append(score_ij)
            pen_C00_map.append(pen_C00_ij)
            pen_C11_map.append(pen_C11_ij)
            pen_C01_map.append(pen_C01_ij)
            
    return scores_outer, pen_C00_map, pen_C11_map, pen_C01_map


def pen_all_scores(scores, score_pairs):
    ''' Estimate all penalties of the individual and global scores.
    Paramters
    ---------
    scores: list of length number subsystems
        List of the VAMPE scores of all VAMPnets.
    score_pairs: list of length number of pairs
        List of the pairwise global scores of all combinations of VAMPnets.
        
    Returns
    -------
    pen_scores: list of length number of pairs
        List of the pairwise penalties of all combindations of VAMPnets.
    '''
    pen_scores = []
    counter = 0
    N = len(scores)
    for i in range(N):
        
        for j in range(i+1, N):
            pen_score_ij = score_loss(scores[i], scores[j], score_pairs[counter])
            pen_scores.append(pen_score_ij)
            counter+=1
                
    return pen_scores

def estimate_transition_matrix(chi_t, chi_tau, mode='regularize', epsilon=1e-6):
    ''' Estimate the transition matrix given the feature vectors at time t and tau
    
    
    '''
    shape = chi_t.shape
        
    batch_size = shape[0]

    x, y = chi_t, chi_tau
    
    # Calculate the covariance matrices
    cov_00 = 1/(batch_size) * torch.matmul(x.T, x) 
    cov_11 = 1/(batch_size) * torch.matmul(y.T, y)
    cov_01 = 1/(batch_size) * torch.matmul(x.T, y)

    # Calculate the inverse of the self-covariance matrices
    cov_00_inv = sym_inverse(cov_00, return_sqrt = False, epsilon=epsilon, mode=mode)
    
    T = cov_00_inv @ cov_01
    
    return T.detach().to('cpu').numpy()

class iVAMPnetModel(Transformer, Model):
    r"""
    A iVAMPNet model which can be fit to data optimizing for one of the implemented VAMP scores.

    Parameters
    ----------
    lobes : list of torch.nn.Module
        List of the lobes of each VAMPNet. See also :class:`deeptime.util.torch.MLP`.
    mask : layer
        Layer, which masks the inputs to the different iVAMPnets. This makes it possible to interpret which
        part of the input is important for each lobe.
    dtype : data type, default=np.float32
        The data type for which operations should be performed. Leads to an appropriate cast within fit and
        transform methods.
    device : device, default=None
        The device for the lobes. Can be None which defaults to CPU.

    See Also
    --------
    iVAMPNet : The corresponding estimator.
    """

    def __init__(self, lobes: list, mask,
                 dtype=np.float32, device=None, epsilon=1e-6, mode='regularize'):
        super().__init__()
        self._lobes = lobes
        self._mask = mask
        self._N = len(lobes)
        self._dtype = dtype
        if self._dtype == np.float32:
            for n in range(self._N):
                self._lobes[n] = self._lobes[n].float()
        elif self._dtype == np.float64:
            for n in range(self._N):
                self._lobes[n] = self._lobes[n].double() 
        self._device = device
        self._epsilon = epsilon
        self._mode = mode
        
    def transform(self, data, numpy=True, batchsize=0, **kwargs):
        '''Transforms the supplied data with the model. It outputs the fuzzy state assignment for each
        subsystem.
        
        Parameters
        ----------
        data: nd.array or torch.Tensor
            Data which should be transformed to a fuzzy state assignment.
        numpy: bool, default=True
            If the output should be converted to a numpy array.
        batchsize: int, default=0
            The batchsize which should be used to predict one chunk of the data, which is useful, if 
            data does not fit into the memory. If batchsize<=0 the whole dataset will be simultaneously 
            transformed.
            
        Returns
        -------
        out: nd.array or torch.Tensor
            The transformed data. If numpy=True the output will be a nd.array otherwise a torch.Tensor.
        '''
        for lobe in self._lobes:
            lobe.eval()
            
        if batchsize>0:
            batches = data.shape[0]//batchsize + (data.shape[0]%batchsize>0)
            if isinstance(data, torch.Tensor):
                torch.split(data, batches)
            else:
                data = np.array_split(data, batches)
                
        out = [[] for _ in range(self._N)]
        with torch.no_grad():
            for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
                mask_data = self._mask(data_tensor)
                for n in range(self._N):
                    mask_n = torch.squeeze(mask_data[n], dim=2)
                    if numpy:
                        out[n].append(self._lobes[n](mask_n).cpu().numpy())
                    else:
                        out[n].append(self._lobes[n](mask_n))
        return out if len(out) > 1 else out[0]
    
    
    def get_transition_matrix(self, data_0, data_t, batchsize=0):
        ''' Estimates the transition matrix based on the two provided datasets, where each frame
        should be lagtime apart.
        
        Parameters
        ----------
        data_0: nd.array or torch.Tensor
            The instantaneous data.
        data_t: nd.array or torch.Tensor
            The time-lagged data.
        batchsize: int, default=0
            The batchsize which should be used to predict one chunk of the data, which is useful, if 
            data does not fit into the memory. If batchsize<=0 the whole dataset will be simultaneously 
            transformed.
            
        Returns
        -------
        T_list: list
            The list of the transition matrices of all subsystems.
        
        '''
        
        chi_t_list = self.transform(data_0, numpy=False, batchsize=batchsize)
        chi_tau_list = self.transform(data_t, numpy=False, batchsize=batchsize)
        T_list = []
        for n in range(self._N):
            chi_t, chi_tau = torch.cat(chi_t_list[n], dim=0), torch.cat(chi_tau_list[n], dim=0)
            K = estimate_transition_matrix(chi_t, chi_tau, mode=self._mode, epsilon=self._epsilon).astype('float64') 
            # Converting to double precision destroys the normalization
            T = K / K.sum(axis=1)[:, None]
            T_list.append(T)
        return T_list
    
    def timescales(self, data_0, data_t, tau, batchsize=0):
        ''' Estimates the timescales of the model given the provided data.
        
        Parameters
        ----------
        data_0: nd.array or torch.Tensor
            The instantaneous data.
        data_t: nd.array or torch.Tensor
            The time-lagged data.
        tau: int
            The time-lagged used for the data.
        batchsize: int, default=0
            The batchsize which should be used to predict one chunk of the data, which is useful, if 
            data does not fit into the memory. If batchsize<=0 the whole dataset will be simultaneously 
            transformed.
            
        Returns
        -------
        its: list
            The list of the implied timescales of all subsystems.
        
        '''
        
        T_list = self.get_transition_matrix(data_0, data_t, batchsize=batchsize)
        its = []
        for T in T_list:
            eigvals = np.linalg.eigvals(T)
            eigvals_sort = np.sort(eigvals)[:-1] # remove eigenvalue 1
            its.append( - tau/np.log(np.abs(eigvals_sort[::-1])))
        
        return its
    
    
class iVAMPnet(DLEstimatorMixin, Transformer):
    r""" Implementation of iVAMPNets :cite:`vnet-mardt2018vampnets` which try to find an optimal featurization of
    data based on a VAMPE score :cite:`vnet-wu2020variational` by using neural networks as featurizing transforms
    which are sought to be independent. This estimator is also a transformer
    and can be used to transform data into the optimized space. From there it can either be used to estimate
    Markov state models via making assignment probabilities crisp (in case of softmax output distributions) or
    to estimate the Koopman operator using the :class:`VAMP <deeptime.decomposition.VAMP>` estimator.

    Parameters
    ----------
    lobes : list of torch.nn.Module
        List of the lobes of each VAMPNet. See also :class:`deeptime.util.torch.MLP`.
    mask : torch.nn.module
        Module which masks the input features to assign them to a specific subsystem.
    device : torch device, default=None
        The device on which the torch modules are executed.
    optimizer : str or Callable, default='Adam'
        An optimizer which can either be provided in terms of a class reference (like `torch.optim.Adam`) or
        a string (like `'Adam'`). Defaults to Adam.
    learning_rate : float, default=5e-4
        The learning rate of the optimizer.
    score_mode : str, default='regularize'
        The mode under which inverses of positive semi-definite matrices are estimated. Per default, the matrices
        are perturbed by a small constant added to the diagonal. This makes sure that eigenvalues are not too
        small. For a complete list of modes, see :meth:`sym_inverse`.
    epsilon : float, default=1e-6
        The strength of the regularization under which matrices are inverted. Meaning depends on the score_mode,
        see :meth:`sym_inverse`.
    dtype : dtype, default=np.float32
        The data type of the modules and incoming data.
    shuffle : bool, default=True
        Whether to shuffle data during training after each epoch.

    See Also
    --------
    deeptime.decomposition.VAMP

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: vnet-
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, lobes: list, mask: nn.Module,
                 device=None, optimizer: Union[str, Callable] = 'Adam', learning_rate: float = 5e-4,
                 score_mode: str = 'regularize', epsilon: float = 1e-6,
                 dtype=np.float32, shuffle: bool = True):
        super().__init__()
        self.N = len(lobes)
        self.lobes = lobes
        self.mask = mask
        self.score_mode = score_mode
        self._step = 0
        self.shuffle = shuffle
        self._epsilon = epsilon
        self.device = device
        self.learning_rate = learning_rate
        self.dtype = dtype
        self.optimizer_lobes = [torch.optim.Adam(lobe.parameters(), lr=self.learning_rate) for lobe in self.lobes]
        self.optimizer_mask = torch.optim.Adam(self.mask.parameters(), lr=self.learning_rate)
        self._train_scores = []
        self._validation_scores = []
        self._train_vampe = []
        self._train_pen_C00 = []
        self._train_pen_C11 = []
        self._train_pen_C01 = []
        self._train_pen_scores = []
        self._train_trace = []
        self._validation_vampe = []
        self._validation_pen_C00 = []
        self._validation_pen_C11 = []
        self._validation_pen_C01 = []
        self._validation_pen_scores = []
        self._validation_trace = []

    @property
    def train_scores(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_scores)
    @property
    def train_vampe(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_vampe)
    @property
    def train_pen_C00(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_pen_C00)
    @property
    def train_pen_C11(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_pen_C11)
    @property
    def train_pen_C01(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_pen_C01)
    @property
    def train_pen_scores(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_pen_scores)
    @property
    def train_trace(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_trace)
    
    @property
    def validation_scores(self) -> np.ndarray:
        r""" The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_scores)
    @property
    def validation_vampe(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_vampe)
    @property
    def validation_pen_C00(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_pen_C00)
    @property
    def validation_pen_C11(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_pen_C11)
    @property
    def validation_pen_C01(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_pen_C01)
    @property
    def validation_pen_scores(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_pen_scores)
    @property
    def validation_trace(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._validation_trace)
    @property
    def epsilon(self) -> float:
        r""" Regularization parameter for matrix inverses.

        :getter: Gets the currently set parameter.
        :setter: Sets a new parameter. Must be non-negative.
        :type: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        assert value >= 0
        self._epsilon = value

    @property
    def score_method(self) -> str:
        r""" Property which steers the scoring behavior of this estimator.

        :getter: Gets the current score.
        :setter: Sets the score to use.
        :type: str
        """
        return self._score_method

    @score_method.setter
    def score_method(self, value: str):
        if value not in valid_score_methods:
            raise ValueError(f"Tried setting an unsupported scoring method '{value}', "
                             f"available are {valid_score_methods}.")
        self._score_method = value

#     @property
#     def lobes(self) -> nn.Module:
#         r""" The instantaneous lobe of the VAMPNet.

#         :getter: Gets the instantaneous lobe.
#         :setter: Sets a new lobe.
#         :type: torch.nn.Module
#         """
#         return self.lobes

#     @lobes.setter
#     def lobes(self, value: list):
#         assert len(value)==self.N, 'You must provide as many lobes as independent subsystems!'
#         for n in range(self.N):
#             self.lobes[n] = value[n]
#             if self.dtype == np.float32:
#                 self.lobes[n] = self.lobes[n].float()
#             else:
#                 self.lobes[n] = self.lobes[n].double()
#             self.lobes[n] = self.lobes[n].to(device=self.device)
    def forward(self, data):
        
        if data.get_device():
            data = data.to(device=self.device)
        masked_data = self.mask(data)
        chi_data_list = []
        for n in range(self.N):
            lobe = self.lobes[n]
            data_n = torch.squeeze(masked_data[n], dim=2)
            chi_data_list.append(lobe(data_n))
        return chi_data_list
    
    def reset_scores(self):
        self._train_scores = []
        self._validation_scores = []
        self._train_vampe = []
        self._train_pen_C00 = []
        self._train_pen_C11 = []
        self._train_pen_C01 = []
        self._train_pen_scores = []
        self._train_trace = []
        self._validation_vampe = []
        self._validation_pen_C00 = []
        self._validation_pen_C11 = []
        self._validation_pen_C01 = []
        self._validation_pen_scores = []
        self._validation_trace = []
        self._step = 0
                                              
    def partial_fit(self, data, lam_decomp: float = 1., mask: bool = False, lam_trace: float = 0., 
                    train_score_callback: Callable[[int, torch.Tensor], None] = None,
                   tb_writer=None, clip=False):
        r""" Performs a partial fit on data. This does not perform any batching.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.
        lam_decomp : float
            The weighting factor how much the dependency score should be weighted in the loss.
        mask : bool default False
            Whether the mask should be trained or not.
        lam_trace : float
            The weighting factor how much the trace should be weighted in the loss.
        train_score_callback : callable, optional, default=None
            An optional callback function which is evaluated after partial fit, containing the current step
            of the training (only meaningful during a :meth:`fit`) and the current score as torch Tensor.
        tb_writer : tensorboard writer
            If given, scores will be recorded in the tensorboard log file.
        clip : bool default=False
            If True the gradients of the weights will be clipped by norm before applying them for the update.
        Returns
        -------
        self : iVAMPNet
            Reference to self.
        """

        if self.dtype == np.float32:
            for n in range(self.N):
                self.lobes[n] = self.lobes[n].float()
        elif self.dtype == np.float64:
            for n in range(self.N):
                self.lobes[n] = self.lobes[n].double()
        for n in range(self.N):
            self.lobes[n].train()
        self.mask.train()
        
        assert isinstance(data, (list, tuple)) and len(data) == 2, \
            "Data must be a list or tuple of batches belonging to instantaneous " \
            "and respective time-lagged data."

        batch_0, batch_t = data[0], data[1]

        if isinstance(data[0], np.ndarray):
            batch_0 = torch.from_numpy(data[0].astype(self.dtype)).to(device=self.device)
        if isinstance(data[1], np.ndarray):
            batch_t = torch.from_numpy(data[1].astype(self.dtype)).to(device=self.device)
        for opt in self.optimizer_lobes:
            opt.zero_grad()
        if mask:
            self.optimizer_mask.zero_grad()
        chi_t_list = self.forward(batch_0) # returns list of feature vectors
        chi_tau_list = self.forward(batch_t)
        # Estimate all individual scores and singular functions
        scores_single, S_single, u_single, v_single, trace_single = score_all_systems(chi_t_list, chi_tau_list, 
                                                                      epsilon=self._epsilon, mode=self.score_mode)
        # Estimate all pairwise scores and independent penalties
        score_pairs, pen_C00_map, pen_C11_map, pen_C01_map = score_all_outer_systems(chi_t_list, chi_tau_list, S_single, 
                                                                           u_single, v_single, device=self.device)
        # Estimate the penalty of the scores
        pen_scores = pen_all_scores(scores_single, score_pairs)
        # Take the mean over all pairs
        pen_scores_all = torch.mean(torch.cat(pen_scores, dim=0))
        pen_C00_map_all = torch.mean(torch.cat(pen_C00_map, dim=0))
        pen_C11_map_all = torch.mean(torch.cat(pen_C11_map, dim=0))
        pen_C01_map_all = torch.mean(torch.cat(pen_C01_map, dim=0))
        trace_all = torch.mean(torch.cat(trace_single, dim=0))
        # Estimate the sum of scores, !!! Check if mean is correct
        vamp_sum_score = torch.mean(torch.cat(scores_single, dim=0))
        vamp_score_pairs = torch.mean(torch.cat(score_pairs, dim=0))
        
        loss_value = - vamp_score_pairs + lam_decomp * pen_scores_all - lam_trace * trace_all
        loss_value.backward()
        if clip:
            # clip the gradients
            for lobe in self.lobes:
                torch.nn.utils.clip_grad_norm_(lobe.parameters(), CLIP_VALUE)
        
        if mask:
            if clip:
                torch.nn.utils.clip_grad_norm_(self.mask.parameters(), CLIP_VALUE)
            self.optimizer_mask.step()
        for opt in self.optimizer_lobes:
            opt.step()
        if train_score_callback is not None:
            lval_detached = loss_value.detach()
            train_score_callback(self._step, -lval_detached)
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', {'train': loss_value.item()}, self._step)
            tb_writer.add_scalars('VAMPE', {'train': vamp_score_pairs.item()}, self._step)
            tb_writer.add_scalars('Pen_C00', {'train': pen_C00_map_all.item()}, self._step)
            tb_writer.add_scalars('Pen_C11', {'train': pen_C11_map_all.item()}, self._step)
            tb_writer.add_scalars('Pen_C01', {'train': pen_C01_map_all.item()}, self._step)
            tb_writer.add_scalars('Pen_scores', {'train': pen_scores_all.item()}, self._step)
            tb_writer.add_scalars('Trace_all', {'train': trace_all.item()}, self._step)
        self._train_scores.append((self._step, (-loss_value).item()))
        self._train_vampe.append((self._step, (vamp_score_pairs).item()))
        self._train_pen_C00.append((self._step, (pen_C00_map_all).item()))
        self._train_pen_C11.append((self._step, (pen_C11_map_all).item()))
        self._train_pen_C01.append((self._step, (pen_C01_map_all).item()))
        self._train_pen_scores.append((self._step, (pen_scores_all).item()))
        self._train_trace.append((self._step, (trace_all).item()))
        self._step += 1

        return self
                                              
    def validate(self, validation_data: Tuple[torch.Tensor], lam_decomp: float = 1., lam_trace: float = 0.) -> torch.Tensor:
        r""" Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        validation_data : Tuple of torch Tensor containing instantaneous and timelagged data
            The validation data.
        lam_decomp : float
            The weighting factor how much the dependency score should be weighted in the loss.
        lam_trace : float
            The weighting factor how much the trace should be weighted in the loss.
            
        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """
        for lobe in self.lobes:
            lobe.eval()
        self.mask.eval()
        
        with torch.no_grad():
            chi_t_list = self.forward(validation_data[0])
            chi_tau_list = self.forward(validation_data[1])
            
            # Estimate all individual scores and singular functions
            scores_single, S_single, u_single, v_single, trace_single = score_all_systems(chi_t_list, chi_tau_list, 
                                                                       epsilon=self._epsilon, mode=self.score_mode)
            # Estimate all pairwise scores and independent penalties
            score_pairs, pen_C00_map, pen_C11_map, pen_C01_map = score_all_outer_systems(chi_t_list, chi_tau_list, 
                                                                  S_single, u_single, v_single, device=self.device)
            # Estimate the penalty of the scores
            pen_scores = pen_all_scores(scores_single, score_pairs)
            # Take the mean over all pairs
            pen_scores_all = torch.mean(torch.cat(pen_scores, dim=0))
            pen_C00_map_all = torch.mean(torch.cat(pen_C00_map, dim=0))
            pen_C11_map_all = torch.mean(torch.cat(pen_C11_map, dim=0))
            pen_C01_map_all = torch.mean(torch.cat(pen_C01_map, dim=0))
            trace_all = torch.mean(torch.cat(trace_single, dim=0))
            # Estimate the sum of scores, !!! Check if mean is correct
            vamp_sum_score = torch.mean(torch.cat(scores_single, dim=0))
            vamp_score_pairs = torch.mean(torch.cat(score_pairs, dim=0))

            loss_value = - vamp_score_pairs + lam_decomp * pen_scores_all - lam_trace * trace_all
            
            return loss_value, vamp_score_pairs, pen_scores_all, pen_C00_map_all, pen_C11_map_all, pen_C01_map_all, trace_all

                                              
    def fit(self, data_loader: torch.utils.data.DataLoader, n_epochs=1, validation_loader=None,
            mask=False, lam_decomp: float = 1., lam_trace: float = 0.,
            start_mask: int = 0, end_trace: int = 0, 
            train_score_callback: Callable[[int, torch.Tensor], None] = None,
            validation_score_callback: Callable[[int, torch.Tensor], None] = None,
            tb_writer=None, reset_step=False, clip=False, save_criteria=None, **kwargs):
        r""" Fits iVAMPnet on data.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data to use for training. Should yield a tuple of batches representing
            instantaneous and time-lagged samples.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
            Validation data, should also be yielded as a two-element tuple.
        mask : bool, default=False
            Bool to decide if the mask should be trained or not
        lam_decomp : float
            The weighting factor how much the dependency score should be weighted in the loss.
        lam_trace : float
            The weighting factor how much the trace should be weighted in the loss.
        start_mask : int, default=0
            The epoch after which the mask should be trained.
        end_trace : int, default=0
            The epoch from which on the trace should not be included in the loss anymore.
        train_score_callback : callable, optional, default=None
            Callback function which is invoked after each batch and gets as arguments the current training step
            as well as the score (as torch Tensor).
        validation_score_callback : callable, optional, default=None
            Callback function for validation data. Is invoked after each epoch if validation data is given
            and the callback function is not None. Same as the train callback, this gets the 'step' as well as
            the score.
        tb_writer : tensorboard writer
            If given, scores will be recorded in the tensorboard log file.
        clip : bool default=False
            If True the gradients of the weights will be clipped by norm before applying them for the update.
        save_criteria : float
            If the validation value of pen_C01 is lower than save_criteria the weights are saved.
            At the end of the training loop the weights will be set to the last saved weights.
        **kwargs
            Optional keyword arguments for scikit-learn compatibility

        Returns
        -------
        self : iVAMPNet
            Reference to self.
        """
        if reset_step: # if all statistics should be recollected from scratch
            self.reset_scores()    
        if save_criteria is not None:
            weights_temp = self.state_dict()
        # and train
        train_mask=False
        for epoch in range(n_epochs):
            if (epoch >= start_mask) and mask:
                train_mask=True
            if epoch >= end_trace:
                lam_trace = 0.
            for batch_0, batch_t in data_loader:
                self.partial_fit((batch_0, batch_t), lam_decomp=lam_decomp, mask=train_mask,
                                 lam_trace=lam_trace, 
                                 train_score_callback=train_score_callback, tb_writer=tb_writer,
                                 clip=clip)

            if validation_loader is not None:
                with torch.no_grad():
                    val_scores = []
                    val_vamp_scores = []
                    val_pen_scores = []
                    val_pen_C00 = []
                    val_pen_C11 = []
                    val_pen_C01 = []
                    val_trace = []
                    for val_batch in validation_loader:
                        ret = self.validate((val_batch[0], val_batch[1]), lam_decomp=lam_decomp, lam_trace=lam_trace)
                        loss_value, vamp_score_pairs, pen_scores_all, pen_C00_map_all, pen_C11_map_all, pen_C01_map_all, trace_all = ret
                        val_scores.append(-loss_value)
                        val_vamp_scores.append(vamp_score_pairs)
                        val_pen_scores.append(pen_scores_all)
                        val_pen_C00.append(pen_C00_map_all)
                        val_pen_C11.append(pen_C11_map_all)
                        val_pen_C01.append(pen_C01_map_all)
                        val_trace.append(trace_all)
                        
                    mean_score = torch.mean(torch.stack(val_scores))
                    mean_vamp_score = torch.mean(torch.stack(val_vamp_scores))
                    mean_pen_score = torch.mean(torch.stack(val_pen_scores))
                    mean_pen_C00 = torch.mean(torch.stack(val_pen_C00))
                    mean_pen_C11 = torch.mean(torch.stack(val_pen_C11))
                    mean_pen_C01 = torch.mean(torch.stack(val_pen_C01))
                    mean_trace = torch.mean(torch.stack(val_trace))
                    
                    if validation_score_callback is not None:
                        validation_score_callback(self._step, mean_score.detach()) 
                    if tb_writer is not None:
                        tb_writer.add_scalars('Loss', {'valid': -mean_score.item()}, self._step)
                        tb_writer.add_scalars('VAMPE', {'valid': mean_vamp_score.item()}, self._step)
                        tb_writer.add_scalars('Pen_C00', {'valid': mean_pen_C00.item()}, self._step)
                        tb_writer.add_scalars('Pen_C11', {'valid': mean_pen_C11.item()}, self._step)
                        tb_writer.add_scalars('Pen_C01', {'valid': mean_pen_C01.item()}, self._step)
                        tb_writer.add_scalars('Pen_scores', {'valid': mean_pen_score.item()}, self._step)
                        tb_writer.add_scalars('Trace_all', {'valid': mean_trace.item()}, self._step)
                    self._validation_scores.append((self._step, (mean_score).item()))
                    self._validation_vampe.append((self._step, (mean_vamp_score).item()))
                    self._validation_pen_C00.append((self._step, (mean_pen_C00).item()))
                    self._validation_pen_C11.append((self._step, (mean_pen_C11).item()))
                    self._validation_pen_C01.append((self._step, (mean_pen_C01).item()))
                    self._validation_pen_scores.append((self._step, (mean_pen_score).item()))
                    self._validation_trace.append((self._step, (mean_trace).item()))
                    
                    if save_criteria is not None:
                        if mean_pen_C01 < save_criteria:
                            # if the criteria is met, save the weights
                            weights_temp = self.state_dict()
                            
        if save_criteria is not None:
            # End the end of the loop load the weights which last fulfilled the save constraint
            self.load_state_dict(weights_temp)
            
        return self
    
    def transform(self, data, instantaneous: bool = True, **kwargs):
        r""" Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy array or torch tensor
            The data to transform.
        instantaneous : bool, default=True
            Whether to use the instantaneous lobe or the time-shifted lobe for transformation.
        **kwargs
            Ignored kwargs for api compatibility.

        Returns
        -------
        transform : array_like
            List of numpy array or numpy array containing transformed data.
        """
        model = self.fetch_model()
        return model.transform(data, **kwargs)

    def fetch_model(self) -> iVAMPnetModel:
        r""" Yields the current model. """
        return iVAMPnetModel(self.lobes, self.mask, dtype=self.dtype, device=self.device, epsilon=self.epsilon,
                            mode=self.score_mode)
        
    def state_dict(self):
        ''' Returns the state_dict of all lobes and the mask.
        
        Returns
        -------
        ret: list of state_dicts
        '''
        dicts_lobe = []
        for lobe in self.lobes:
            dicts_lobe.append(lobe.state_dict())
        mask_dict = self.mask.state_dict()
        ret = [dicts_lobe, mask_dict]
        return ret
    
    
    def load_state_dict(self, state_dict):
        ''' Loads the provided state_dict into the estimator. Useful to load a saved training instance.
        
        Parameters
        ----------
        state_dict: list
            Should be of the form given by the function self.state_dict. Its a list of a list of all state_dict lobes and the state dict of the mask.
        '''
        
        dict_lobes, mask_dict = state_dict
        for n in range(self.N):
            self.lobes[n].load_state_dict(dict_lobes[n])
        self.mask.load_state_dict(mask_dict)
        return
        
    def save_params(self, path: str):
        ''' Saves the state_dicts at the specified path.
        
        Parameters
        ----------
        path: str
            The path where the state_dict should be saved.
        '''
        dicts_lobe, mask_dict = self.state_dict()
        savez_dict = dict()
        for n in range(self.N):
            savez_dict['lobe_'+str(n)] = dicts_lobe[n] 
        savez_dict['mask_dict'] = mask_dict
        
        np.savez(path, **savez_dict)
        
        return print('Saved parameters at: '+path)
    
    def load_params(self, path: str):
        ''' Loads the state_dicts from the specified path.
        
        Parameters
        ----------
        path: str
            The path where the state_dict should be loaded from.
        '''
        dicts = np.load(path, allow_pickle=True)
        
        dict_lobes = []
        mask_dict = dicts['mask_dict'].item()
        for n in range(self.N):
            dict_lobes.append(dicts['lobe_'+str(n)].item())
        state_dict = [dict_lobes, mask_dict]
        self.load_state_dict(state_dict)
        
        return