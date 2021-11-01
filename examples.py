import numpy as np
import deeptime
import matplotlib.pyplot as plt

class Toymodel_2Systems():
    ''' Class for generating the data for the Toymodel with two subsystems sampled from a hidden Markov Chain
    '''
    def __init__(self, eps_list, mean=None, cov=None):
        super().__init__()
        self.eps_list = eps_list
        
        self.T, self.T1, self.T2 = self.generate_hidden_matrix()
        self.msm = msm = deeptime.markov.msm.MarkovStateModel(self.T)
        if mean is None:
            mean_per_state = np.array([[2, 2],
                              [2, -2],
                              [0,  2],
                              [0, -2],
                              [-2, 2],
                              [-2, -2]])
        if cov is None:
            cov = .1 * np.eye(2)
        self.mean = mean_per_state
        self.cov = cov
        
    def generate_traj(self, steps):
        ''' Generate a trajectory with the defined hidden Markov Chain
        
        Parameters
        ----------
        steps: int
            Number of timesteps
            
        Returns
        -------
        hidden_state_traj: np.array
            The hidden Markov Chain.
        observable_traj: np.array
            The observable trajectory.
        
        '''
        hidden_state_traj = self.msm.simulate(steps)
        observable_traj = np.zeros((hidden_state_traj.shape[0], 2)) - 1
        n_hidden = self.T.shape[0]
        for state in range(n_hidden):
            ix = np.where(hidden_state_traj == state)[0]
            observable_traj[ix] = np.random.multivariate_normal(self.mean[state], self.cov, size=ix.shape[0])
        
        return hidden_state_traj, observable_traj
        
        
    def generate_hidden_matrix(self):
        """
        Generates hidden transition matrix.
        """
        eps0, eps1, eps2, eps3 = self.eps_list
        X1 = np.array([[1-eps0-eps0, eps0, eps0],
                       [eps1, 1-eps1-eps1, eps1],
                       [eps0, eps1, 1-eps0-eps1]])
        X1 = X1/np.sum(X1, keepdims=True)
        pi = np.sum(X1,1, keepdims=True)

        T1 = X1 / pi
    #     X2 = np.array([[1-eps2-eps2, eps2, eps2],
    #                    [eps3, 1-eps3-eps3, eps3],
    #                    [eps2, eps3, 1-eps2-eps3]])
        X2 = np.array([[1-eps2, eps2],
                       [eps2,1-eps2]])
        X2 = X2/np.sum(X2, keepdims=True)
        pi = np.sum(X2,1, keepdims=True)

        T2 = X2 / pi

        T = np.kron(T1, T2)

        assert deeptime.markov.tools.analysis.is_transition_matrix(T)
        return T, T1, T2
    
    def plot_toymodel(self, hidden_state_traj, observable_traj):
        ''' Plots the toymodel given a hidden trajectory and the corresponding observable coordinates.
        
        Parameters
        ---------
        hidden_state_traj: nd.array
            The hidden trajectory of size (T,), where T is the number of frames.
        observable_traj: nd.array
            The observable array of size (T, n), where n is the size of the observable space.
        '''
        plt.scatter(*observable_traj.T, c=hidden_state_traj, alpha=.5)
        plt.show()
        
    def plot_eigfunc(self, hidden_state_traj, observable_traj, save=None):
        ''' Plots the true eigenfunctions.
        
        Parameters
        ----------
        hidden_state_traj: nd.array
            The hidden trajectory of size (T,), where T is the number of frames.
        observable_traj: nd.array
            The observable array of size (T, n), where n is the size of the observable space.
        save: default=None
            If save is not None, the figure will be saved.
        '''
        eigv, eigvec = np.linalg.eig(self.T)
        ind_sort = np.argsort(eigv)[::-1]
        eigv = eigv[ind_sort]
        eigvec = eigvec[:,ind_sort]

        x_size = 3
        y_size = 2
        factor=2
        factor_x=1.5
        factor_y=2
        fig, ax = plt.subplots(x_size, y_size, sharex=True, sharey=True, figsize=(6*factor_x,4*factor_y))
        i_state = 0
        skip=1
        ax[0,0].text(0.8,6,'Global eigenfunctions', fontsize=10*factor)
        for i in range(self.T.shape[0]):
        #     print(output_i, system_i)
            output_i = i//y_size
            system_i = i%y_size
        #     print(output_i, system_i)
            eigv_i = eigvec[:,i]
            if i ==0:
                c=np.ones_like(eigv_i[hidden_state_traj[::skip]])
            else:
                c=eigv_i[hidden_state_traj[::skip]]
            ax[output_i, system_i].scatter(
                *observable_traj[::skip].T, c=c,
            )
            ax[output_i, system_i].set_title(r'$\lambda_{}={:.3}$'.format(i,eigv[i]), fontsize=10*factor)
            if output_i==(x_size-1):
                ax[output_i, system_i].set_xlabel('x', fontsize=10*factor)
                ax[output_i, system_i].set_xticks([-2,0,2])
                ax[output_i, system_i].set_xticklabels([-2,0,2], fontsize=8*factor)
            if system_i ==0:
                ax[output_i, system_i].set_ylabel('y', fontsize=10*factor)
                ax[output_i, system_i].set_yticks([-2,0,2])
                ax[output_i, system_i].set_yticklabels([-2,0,2], fontsize=8*factor)
        if save is not None:
            fig.savefig('./3x2_mix_T_hidden_eigvec.png', bbox_inches='tight', dpi=900)
        fig.show()
        
def plot_mask(mask, vmax=1., save=False, skip=1):
    ''' Plots the mask of the toymodels.
    
    Parameters
    ----------
    mask: masks.Mask
        The mask defined in masks.py
    vmax: float
        The maximal value of the scale which will be used.
    save: bool
        If True, the figure will be saved.
    skip: int
        Number of input features which will be skipped for the yticks.
    
    '''
    attention = mask.get_softmax()
    attention_np = np.squeeze(attention.detach().to('cpu').numpy())
    plt.imshow(attention_np, vmin=0, vmax=vmax, cmap=plt.cm.binary, aspect='auto')
    plt.xlabel('Subsystem', fontsize=18)
    plt.ylabel('Input', fontsize=18)
    input_size, number_subsystems = attention_np.shape
    plt.xticks(np.arange(number_subsystems),['{}'.format(i) for i in range(number_subsystems)], fontsize=16)
    plt.yticks(np.arange(0,input_size,skip),['x{}'.format(i) for i in range(0,input_size,skip)], fontsize=16)
    plt.show()
    if save:
        plt.savefig('./Mask.pdf', bbox_inches='tight')
        
def plot_states(model, data, save=False):
    ''' Plots the state probability vector of all subsystems.
    
    Parameters
    ----------
    model: ivampnets.iVAMPnetModel
        The model which transforms the input data.
    data: torch.Tensor or nd.array
        Input data which should be plotted. Has to be transformabel by the model.
    save: bool
        If True, the figure will be saved.
    '''
    pred_list = model.transform(data)
    number_subsystems = len(pred_list)
    transformed_data = []
    output_sizes = []
    for n in range(model._N):
        transformed_data.append(np.concatenate(pred_list[n], axis=0)) 
        output_sizes.append(transformed_data[-1].shape[-1])
    transformed_data = np.concatenate(transformed_data, axis=1)
    subsysteme = ['I', 'II']
        
    max_output_size = max(output_sizes)
    x_size = output_sizes[0]
    y_size = output_sizes[1]
    factor=2
    factor_x=1.5
    factor_y=2
    fig, ax = plt.subplots(x_size, y_size, sharex=True, sharey=True, figsize=(6*factor_x,4*factor_y))
    ax[0,0].text(1.,6,'State assignment', fontsize=10*factor)
    state_real = 0
    for i_state in range(number_subsystems * max_output_size):
        output_i = i_state%max_output_size
        system_i = i_state//max_output_size
        if output_i < output_sizes[system_i]:
            z = transformed_data[:,state_real]
        #     print(z.shape)
            ax[output_i, system_i].scatter(
                x=data[:, 0], y=data[:, 1], c=z,
            )
            if output_i ==0:
                ax[output_i, system_i].set_title(f"Subsystem {subsysteme[system_i]}", fontsize=10*factor)
            if system_i ==0:
                if output_i==(y_size):
                    ax[output_i, system_i].set_xlabel('x', fontsize=10*factor)
                    ax[output_i, system_i].set_xticks([-2,0,2])
                    ax[output_i, system_i].set_xticklabels([-2,0,2], fontsize=8*factor)
            else:
                if output_i==(y_size-1):
                    ax[output_i, system_i].set_xlabel('x', fontsize=10*factor)
                    ax[output_i, system_i].set_xticks([-2,0,2])
                    ax[output_i, system_i].set_xticklabels([-2,0,2], fontsize=8*factor)
            if system_i ==0:
                ax[output_i, system_i].set_ylabel('y', fontsize=10*factor)
                ax[output_i, system_i].set_yticks([-2,0,2])
                ax[output_i, system_i].set_yticklabels([-2,0,2], fontsize=8*factor)
            state_real+=1
        else:
            ax[output_i, system_i].axis('off')
    if save:
        fig.savefig('3x2_mix_state_assignment.png', bbox_inches='tight', dpi=900)
    plt.show()
    
def plot_eigfuncs(model, dataset):
    ''' Plots the eigenfunctions of the approximation of the model given the dataset.
    
    Parameters
    ----------
    model: ivampnets.iVAMPnetModel
        The model which transforms the input data.
    dataset: TrajectoryDataset
        Dataset with data and data_lagged.
    '''
    T_list = model.get_transition_matrix(dataset.data, dataset.data_lagged)
    pred_list = model.transform(dataset.data)
    number_subsystems = len(pred_list)
    transformed_data = []
    output_sizes = []
    for n in range(model._N):
        transformed_data.append(np.concatenate(pred_list[n], axis=0)) 
        output_sizes.append(transformed_data[-1].shape[-1])
    
    x_size = output_sizes[0]
    y_size = 2
    factor=2
    factor_x=1.5
    factor_y=2
    fig, ax = plt.subplots(x_size, y_size, sharex=True, sharey=True, figsize=(6*factor_x,4*factor_y))
    ax[0,0].text(-2,6,'Subsystem I', fontsize=10*factor)
    ax[0,1].text(-2,6,'Subsystem II', fontsize=10*factor)
    i_state = 0
    for n in range(number_subsystems):
        K=T_list[n]
        eigv, eigvec = np.linalg.eig(K)
        ind_sort = np.argsort(eigv)[::-1]
        eigv = eigv[ind_sort]
        eigvec = eigvec[:,ind_sort]
        for i in range(K.shape[0]):
            output_i=i
            system_i=n
            eigv_i = eigvec[:,i]
            if i ==0:
                c=np.ones_like(transformed_data[n]@eigv_i)
            else:
                c=transformed_data[n]@eigv_i
            if output_i < output_sizes[system_i]:

                ax[output_i, system_i].scatter(
                    *dataset.data.T, c=c,
                )
                ax[output_i, system_i].set_title(r'$\lambda_{}={:.3}$'.format(output_i,eigv[i]), fontsize=10*factor)
                if output_i==(output_sizes[n]-1):
                    ax[output_i, system_i].set_xlabel('x', fontsize=10*factor)
                    ax[output_i, system_i].set_xticks([-2,0,2])
                    ax[output_i, system_i].set_xticklabels([-2,0,2], fontsize=8*factor)
                if system_i ==0:
                    ax[output_i, system_i].set_ylabel('y', fontsize=10*factor)
                    ax[output_i, system_i].set_yticks([-2,0,2])
                    ax[output_i, system_i].set_yticklabels([-2,0,2], fontsize=8*factor)

    ax[2, 1].axis('off')
    # fig.savefig('./Figs/3x2_mix_T_hidden_eigvec_estimated.png', bbox_inches='tight', dpi=900)
    plt.show()
    
    
class HyperCube():
    ''' Class for generating the data for the Hyper Cube sampled from a hidden Markov Chain.
    
    Parameters
    -----------
    eps_list: list.
        List of the probability for each independent subsystem to stay in the same state
    lam: float.
        Coupling of the subsystems. If zero no coupling is active. 
    mean: np.array
        Defines the mean values of the multivariant Gaussians, when generating a trajectory in the observable space.
        If None, predefined values are taken
    std: np.array
        Defines the std values of the same multivariant Gaussian.
    '''
    def __init__(self, eps_list, lam=0.0, mean=None, cov=None):
        super().__init__()
        self.eps_list = eps_list
        self.lam = lam
        self.T_total, self.T_list, self.T_coupled_list = self.generate_hidden_matrix()
        self.msm = msm = deeptime.markov.msm.MarkovStateModel(self.T_total)
        self.N = len(eps_list)
        output_size = [2 for _ in range(self.N)]
        if mean is None:
            indices_fullsys = np.arange(2**self.N)
            indices_subsystems = np.unravel_index(indices_fullsys, output_size)
            indices_fullsys, indices_subsystems
            mean_per_state = []
            for i in range(len(indices_fullsys)):
                list_ind = [indices_subsystems[n][i] for n in range(self.N)]
                mean_per_state.append(list_ind)
            mean_per_state = 2*np.array(mean_per_state)
        if cov is None:
            cov = .1 * np.eye(self.N)
        self.mean = mean_per_state
        self.cov = cov
        
        self.eigvals_list = []
        self.eigvals_list_coupled = []
        for i in range(self.N):
            Ti = self.T_list[i]
            eigv, eigvec = np.linalg.eig(Ti)
            ind_sort = np.argsort(eigv)[::-1]
            eigv = eigv[ind_sort]
            self.eigvals_list.append(eigv[1:])
            if i<(self.N//2):
                Ti = self.T_coupled_list[i]
                eigv, eigvec = np.linalg.eig(Ti)
                ind_sort = np.argsort(eigv)[::-1]
                eigv = eigv[ind_sort]
                self.eigvals_list_coupled.append(eigv[1:-1])
        
    def generate_traj(self, steps, angles=None, dim_noise=0):
        ''' Generate a trajectory with the defined hidden Markov Chain
        
        Parameters
        ----------
        steps: int
            Number of timesteps
        angles: np.array
            Rotate the observable space by specified angles.
        dim_noise: int
            Number of noise dimensions
        Returns
        -------
        hidden_state_traj: np.array
            The hidden Markov Chain.
        observable_traj: np.array
            The observable trajectory.
        
        '''
            
        hidden_state_traj = self.msm.simulate(steps)
        observable_traj = np.zeros((hidden_state_traj.shape[0], self.N)) - 1
        n_hidden = self.T_total.shape[0]
        for state in range(n_hidden):
            ix = np.where(hidden_state_traj == state)[0]
            observable_traj[ix] = np.random.multivariate_normal(self.mean[state], self.cov, size=ix.shape[0])
        
        if angles is not None:
            rot_matrix = self._get_rotation_matrix(angles)
            observable_traj = observable_traj @ rot_matrix
        if dim_noise>0:
            observable_traj = np.concatenate((observable_traj, np.random.randn(steps, dim_noise)), axis=1)
            
        return hidden_state_traj, observable_traj
        
        
    def generate_hidden_matrix(self):
        """
        Generates hidden transition matrix.
        """
        T_list = []
        T_coupled_list = []
        lam = self.lam
        for i in range(len(self.eps_list)):
            epsi = self.eps_list[i]
            Ti = np.array([[1-epsi, epsi],
              [epsi, 1-epsi]])
            T_list.append(Ti)
            if (i%2)==0:
                eps1 = self.eps_list[i]
                eps2 = self.eps_list[i+1]
                Tij = np.array([[(1 - eps2) * (1 - eps1) - lam, eps2 * (1 - eps1) - lam, (1 - eps2) * eps1+lam, eps2 * eps1+lam],
                     [eps2 * (1 - eps1) - lam, (1 - eps2) * (1 - eps1) - lam,  eps2  * eps1+lam,      (1 - eps2) *  eps1+lam],
                     [(1 - eps2) * eps1 + lam,  eps2 * eps1 + lam,       (1 - eps2) * (1 - eps1) - lam, eps2 * (1 - eps1) - lam],
                     [eps2  * eps1 + lam, (1 - eps2) *  eps1 + lam, eps2 * (1 - eps1) - lam, (1 - eps2) * (1 - eps1) - lam]])
                T_coupled_list.append(Tij)

        T_total = np.array([[1]])
        for Ti in T_coupled_list:
            T_total = np.kron(T_total, Ti)
        return T_total, T_list, T_coupled_list
    
    def _get_rotation_matrix(self, angles=None):
        '''Goal is to create a rotation matrix which just rotates within a coupled 2D system, 
        so each subsystem just needs information from two input features'''
    
        if type(angles)==type(None):
            angles = 2 * np.pi * np.random.random(self.N//2)

        rot_total = np.eye(self.N)
        for i in range(self.N//2):
            rot_temp = np.eye(self.N)
            start = i*2
            end = start+2
            rot = np.array([[ np.cos(angles[i]),  np.sin(angles[i])],
                            [-np.sin(angles[i]),  np.cos(angles[i])]])
            rot_temp[start:end, start:end] = rot
    #         print(rot_temp)
            rot_total = rot_total @ rot_temp

        return rot_total
    
    
def plot_its(its, lag, ylog=False, multiple_runs = False):
    '''Plots the provided implied timescales.'

    Parameters
    ----------
    its: numpy array
        the its array returned by the function get_its
    lag: numpy array
        lag times array used to estimate the implied timescales
    ylog: Boolean, optional, default = False
        if true, the plot will be a logarithmic plot, otherwise it
        will be a semilogy plot
    multiple_runs: bool
        If True the provided its are expected to have a first dimension with number of runs which should be used to
        estimate a mean and an error estimate.

    '''
    fig, ax = plt.subplots()

    func = ax.loglog if ylog else ax.semilogy
    if not multiple_runs:
        its = np.sort(its, axis=0)
        for i in range(np.shape(its)[0]):
            j=i+1
            if i==0:
                label='Model'
            else:
                label=''
            func(lag, its[-j] ,'o',lw=2, ms=7, label=label)
    else:
        its_mean = np.mean(its, 0)[::-1]
        its_std = np.std(its, 0)[::-1]
        for index_its, m, s in zip(range(len(its)), its_mean, its_std):
            func(lag, m, color = 'C{}'.format(index_its))
            ax.fill_between(lag, m+s, m-s, color = 'C{}'.format(index_its), alpha = 0.2)

    func(lag,lag, 'k')
    ax.fill_between(lag, lag, 0.99, alpha=0.2, color='k');
    return ax, fig

def plot_hypercube_its(its, msmlags, its_true, ylog=False, save=None):
    '''Plots the provided implied timescales of the hypercube toy example.'

    Parameters
    ----------
    its: numpy array
        the its array returned by the function get_its
    lag: numpy array
        lag times array used to estimate the implied timescales
    ylog: Boolean, optional, default = False
        if true, the plot will be a logarithmic plot, otherwise it
        will be a semilogy plot
    multiple_runs: bool
        If True the provided its are expected to have a first dimension with number of runs which should be used to
        estimate a mean and an error estimate.

    '''
    ax, fig = plot_its(its, msmlags, ylog=ylog)

    for i, _its in enumerate(its_true.T):
        if i==0:
            label='True'
        else:
            label=''
        ax.hlines(_its, 1,msmlags.max(),  color='C{}'.format(i), label=label)
    #     ax.plot(msmlags, _its, 'x',ms=8, c='C{}'.format(i), label=label)
    ax.set_xlabel('Lagtime [a.u.]', fontsize=16)
    ax.set_ylabel('Implied Timescales [a.u.]', fontsize=16)
    ax.legend(fontsize=14, loc='lower right')
    ax.set_xticks([1,3,6,9])
    ax.set_xlim(0.95,8.5)
    ax.set_ylim(1,60)
    ax.set_xticklabels([1,3,6,9], fontsize=14)
    ax.set_xticklabels([], fontsize=14, minor=True)
    # ax.set_xticks([2,4,5,7,8])
    # ax.set_xticklabels(['','','','',''], fontsize=14)

    ax.set_yticks([1,10,50])
    ax.set_yticklabels([1,10,50], fontsize=14)
    ax.tick_params(which='major', direction='out', length=6, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)
    if save is not None:
        fig.savefig('./Hypercube_10_ITS.pdf', bbox_inches='tight')
    plt.show()
    
def plot_protein_mask(mask, skip_start=4, save=None):
    ''' Helper function to plot the mask of a protein. 
    Parameters
    ----------
    mask: masks.mask_proteins
        A mask_proteins object from the mask.py file.
    skip_start: int
        How many residues where skipped from the beginning of the chain before including them in the distance 
        calculation.
    save: bool
        If true the plot will be saved.
    '''
    import matplotlib.lines as mlines
    attention = mask.get_softmax()
    values = np.squeeze(attention.detach().to('cpu').numpy())
    plt.plot(np.arange(skip_start,attention.shape[0]+skip_start), values, linewidth=2)
    plt.xticks(fontsize=14)
    plt.xlabel('Residue', fontsize=16)
    plt.yticks(fontsize=14)
    plt.ylabel('Importance weight', fontsize=16)
    patch1 = mlines.Line2D([], [], color='C0',linewidth=3,
                             label='Subsystem I')
    patch2 = mlines.Line2D([], [], color='C1',linewidth=3,
                               label='Subsystem II')
    plt.legend(handles=[patch1, patch2], fontsize=14)
    if save:
        plt.savefig('./Syt_attention.pdf', bbox_inches='tight')
    plt.show()
    
def plot_protein_its(its, lag, ylog=False, multiple_runs = False, percent=0.68):
    '''Plots the implied timescales calculated by the function
    'get_its'

    Parameters
    ----------
    its: numpy array
        the its array returned by the function get_its
    lag: numpy array
        lag times array used to estimate the implied timescales
    ylog: Boolean, optional, default = False
        if true, the plot will be a logarithmic plot, otherwise it
        will be a semilogy plot

    '''
    fig, ax = plt.subplots(1,2, sharey=True, figsize=(12,4))
    number_subsystems = len(its)
    
    labels=['Subsystem I', 'Subsystem II', 'Subsystem III', 'Subsystem IV', 'Subsystem V']
    style = '-o'
    if not multiple_runs:
        for n, its_s in enumerate(its):
            func = ax[n].loglog if ylog else ax[n].semilogy
            
            for i in range(np.shape(its_s)[1]):
                
                if i==0:
                    label=labels[n]
                else:
                    label=''
                func(lag, its_s[:,-(i+1)], style, lw=2, ms=7,label=label)
    else:
        for n in range(number_subsystems):
            func = ax[n].loglog if ylog else ax[n].semilogy
            its_n = its[n]
            for index_its in range(its_n.shape[-1]):
                if index_its==0:
                    label=labels[n]
                else:
                    label=''
                its_all = its_n[:,:,index_its]
                sort_its = np.sort(its_all,axis=0)
                runs=its_all.shape[0]
                ind_upper_lower = int(runs/2- percent * runs/2)+1
                lower = sort_its[ind_upper_lower]
                upper = sort_its[-ind_upper_lower]
                m = its_all.mean(0)
                func(lag, m,  style, lw=2, ms=7,label=label, color = 'C{}'.format(index_its))
                ax[n].fill_between(lag, upper, lower, color = 'C{}'.format(index_its), alpha = 0.2)

    return ax, fig