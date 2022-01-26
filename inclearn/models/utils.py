import os
import torch
import numpy as np
import torch.nn as nn
import random
import string
from pathlib import Path
import torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def log_comet_metric(exp, name, val, step):
    exp.log_metric(name=name, value=val, step=step)


def save_model(model, path):
    torch.save(model.cpu(), path)

def load_model(path):
    model = torch.load(path)
    return model

def save_task_model_by_policy(model, task, policy, exp_dir):
    # task = 0 is the initilization
    if task == 0 or policy == 'init':
        save_model(model, '{}/init.pth'.format(exp_dir))

    # the first task model is the same for all 
    if task == 1:
        save_model(model, '{}/t_{}_seq.pth'.format(exp_dir, task))
        save_model(model, '{}/t_{}_lmc.pth'.format(exp_dir, task))
        save_model(model, '{}/t_{}_mtl.pth'.format(exp_dir, task))
    else:
        save_model(model, '{}/t_{}_{}.pth'.format(exp_dir, task, policy))


def load_task_model_by_policy(task, policy, exp_dir):
    if task == 0 or policy == 'init':
        return load_model('{}/init.pth'.format(exp_dir))
    return load_model('{}/t_{}_{}.pth'.format(exp_dir, task, policy))


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m

def assign_grads(m, grads):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    for param in state_dict.keys():
        if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
        param_count = state_dict[param].numel()
        param_shape = state_dict[param].shape
        state_dict[param].grad =  grads[index:index+param_count].view(param_shape).clone()
        index += param_count
    m.load_state_dict(state_dict)
    return m

def get_norm_distance(m1, m2):
    m1 = flatten_params(m1, numpy_output=False)
    m2 = flatten_params(m2, numpy_output=False)
    return torch.norm(m1-m2, 2).item()


def get_cosine_similarity(m1, m2):
    m1 = flatten_params(m1)
    m2 = flatten_params(m2)
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cosine(m1, m2)


def save_np_arrays(data, path):
    np.savez(file=path, **data)


class ContinualMeter:
    def __init__(self, name, n_tasks):
        self.name = name
        self.data = np.zeros((n_tasks, n_tasks))

    def update(self, current_task, target_task, metric):
        self.data[current_task-1][target_task-1] = round(metric, 3)

    def save(self, config):
        path = '{}/{}.csv'.format(config['exp_dir'], self.name)
        np.savetxt(path, self.data, delimiter=",")


def get_latex_str_for_minima(policy, task):
    if policy == 'seq':
        return r"$\hat{{w}}_{{{}}}".format(task)
    elif policy == 'lmc':
        return r"$\bar{{w}}_{{{}}}".format(task)
    elif policy == 'mtl':
        return r"$w^*_{{{}}}".format(task)
    else:
        raise Exception("unknown policy")

def get_latex_str_for_path(p1, t1, p2, t2):
    start = get_latex_str_for_minima(p1, t1)
    end = get_latex_str_for_minima(p2, t2)
    return start + r" \rightarrow " + end


def get_model_grads(model, loader):
    grads = []
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    count = 0
    test_loss = 0
    for data, target, task_id in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            count += len(target)
            output = model(data, task_id)
            curr_loss = criterion(output, target)
            curr_loss.backward()
    for param in model.parameters():
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    # print("Norm grad >> ", torch.norm(grads))
    return grads


from torch.autograd import Variable, Function
import scipy as sp

# Inherit from Function
class OLELoss(Function):
    @staticmethod
    def forward(self, X, y):
        X = X.cpu().numpy()
        y = y.cpu().numpy()

        classes = np.unique(y)
        C = classes.size
        
        N, D = X.shape

        DELTA = 1.
        

        # gradients initialization
        Obj_c = 0
        dX_c = np.zeros((N, D))
        Obj_all = 0;
        dX_all = np.zeros((N,D))

        eigThd = 1e-6 # threshold small eigenvalues for a better subgradient


        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            A = X[y==c,:]

            # SVD
            U, S, V = sp.linalg.svd(A, full_matrices = False, lapack_driver='gesvd')
                
            V = V.T
            nuclear = np.sum(S);

            ## L_c = max(DELTA, ||TY_c||_*)-DELTA
            
            if nuclear>DELTA:
              Obj_c += nuclear;
            
              # discard small singular values
              r = np.sum(S<eigThd)

            #   s = S[0:S.shape[0]-r]/S.max()*0.1 + 0.9

            #   uprod = (U[:,0:U.shape[1]-r].dot(np.diag(s))).dot(V[:,0:V.shape[1]-r].T)
              uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)
            
              dX_c[y==c,:] += uprod
            else:
              Obj_c+= DELTA
            
        # compute objective and gradient for secon term ||TX||*
                                 
        U, S, V = sp.linalg.svd(X, full_matrices = False, lapack_driver='gesvd')  # all classes

        V = V.T

        Obj_all = np.sum(S);

        r = np.sum(S<eigThd)

        # s = 1-S[0:S.shape[0]-r]/S.max()*0.1

        uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)
        # uprod = (U[:,0:U.shape[1]-r].dot(np.diag(s))).dot(V[:,0:V.shape[1]-r].T)

        dX_all = uprod

        
        obj = (Obj_c  - Obj_all)/N*0.325


        # dX = ( - dX_all)/N*np.float(0.25) 
        dX = (dX_c  - dX_all)/N*np.float(0.325)
        # assert obj > 0

        self.dX = torch.FloatTensor(dX)
        return torch.FloatTensor([float(obj)]).cuda()
    
    @staticmethod
    def backward(self, grad_output):
        # print self.dX
        return self.dX.cuda(), None