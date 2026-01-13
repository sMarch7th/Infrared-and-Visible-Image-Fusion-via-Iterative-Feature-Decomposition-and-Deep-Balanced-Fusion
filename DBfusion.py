import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from solver import  broyden, weight_tie 
from cnn_judge_train import SiameseNetwork


def list2vec(z1_list):
    bsz = z1_list[0].size(0)
    return torch.cat([elem.reshape(bsz, -1 , 1 ) for elem in z1_list], dim=1)

def vec2list(z1, cutoffs):
    """Convert a vector back to a list, via the cutoffs specified"""
    bsz = z1.shape[0]
    z1_list = []
    start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1]
    for i in range(len(cutoffs)):
        z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
        if i < len(cutoffs)-1:
            start_idx = end_idx
            end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1]
    return z1_list


class SimpleResidualBlock(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        super(SimpleResidualBlock, self).__init__()

        self.out_dim = out_dim
        self.conv1 = torch.nn.Conv1d(self.out_dim, self.out_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.out_dim, self.out_dim, 1)
        self.gn2 = torch.nn.BatchNorm1d(self.out_dim)  
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, injection_feature):
        out = self.conv1(x) + injection_feature
        out = self.relu2(self.conv2(self.gn2(out)))
        return out


class DBFusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        super(DBFusionBlock, self).__init__()

        self.out_dim = num_out_dims[-1]

        self.gate = torch.nn.Conv1d(num_out_dims[0], self.out_dim, 1)
        self.fuse = torch.nn.Conv1d(self.out_dim, self.out_dim, 1)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.gn3 = nn.GroupNorm(1, self.out_dim, affine=True)
            
    def forward(self, x, injection_features, residual_feature):
        extracted_feats = []
        for i, inj_feat in enumerate(injection_features):
            extracted_feats.append(torch.mul(x, self.dropout1(self.gate(inj_feat + x))))

        out = self.dropout2(self.fuse(torch.stack(extracted_feats, dim=0).sum(dim=0))) + residual_feature
        out = self.gn3(F.relu(out))
        
        return out

    
class DBFusionLayer(nn.Module):
    def __init__(self, num_out_dims):
        super(DBFusionLayer, self).__init__()
        self.num_branches = len(num_out_dims)
        self.block = SimpleResidualBlock
        self.fusion_block = DBFusionBlock
        self.branches = self._make_branches(self.num_branches, num_out_dims)

    def _make_one_branch(self, branch_index, num_out_dims):
        out_dim = num_out_dims[branch_index]
        return self.block(out_dim, deq_expand=1, num_groups=1, dropout=0)
    
    def _make_fusion_branch(self, branch_index, num_out_dims):
        return self.fusion_block(num_out_dims, deq_expand=2, dropout=0)

    def _make_branches(self, num_branch, num_out_dims):
        branch_layers = [self._make_one_branch(i, num_out_dims) for i in range(num_branch - 1)]
        branch_layers.append(self._make_fusion_branch(num_branch - 1, num_out_dims))
        return nn.ModuleList(branch_layers)

    def forward(self, x, injection):
        inject_features = [injection[0], injection[1], injection[2]]
        x_block_out = []
        for i in range(self.num_branches - 1):
            out = self.branches[i](x[i], inject_features[i])
            x_block_out.append(out)
        x_block_out.append(self.branches[self.num_branches - 1](x[self.num_branches - 1], x_block_out, injection[-1]))
        return x_block_out




class DBFusion(nn.Module):
    def __init__(self, channel_dim, num_modals, f_thres=50, b_thres=51, stop_mode="abs", deq=True, num_layers=1, solver='anderson'):
        super(DBFusion, self).__init__()
        self.f_thres = f_thres
        self.b_thres = b_thres
        self.stop_mode = stop_mode
        self.func_ = DBFusionLayer([channel_dim for _ in range(num_modals + 1)])
        self.f_solver = anderson if solver == 'anderson' else broyden
        self.b_solver = anderson if solver == 'anderson' else broyden
        self.deq = deq
        self.num_layers = num_layers

        self.weights = nn.ParameterList()
        for i in range(num_modals):
            self.weights.append(nn.Parameter(torch.FloatTensor(1), requires_grad=True))
            self.weights[i].data.fill_(1)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias.data.fill_(0)
        
    def featureFusion(self, features, fusion_feature, compute_jac_loss=True,shape=[]):
        batch_size = features[0].shape[0]
        feature_dim = features[0].shape[1]
        x_list = [f for f in features] + [fusion_feature]
        out_dim_list = [f.shape[1] for f in features] + [fusion_feature.shape[1]]
        z_list = [torch.zeros(batch_size, dim_size , fusion_feature.shape[2]).cuda() for dim_size in out_dim_list]
        cutoffs = [(elem.size(1) , elem.size(2)) for elem in z_list]
        z1 = list2vec(z_list)

        func = lambda z: list2vec(self.func_(vec2list(z, cutoffs), x_list))
        deq = self.deq
        jac_loss = torch.tensor(0.0).to(fusion_feature)
        if not deq:
            result = {'rel_trace':[]}
            for layer_ind in range(self.num_layers): 
                z1 = func(z1)
                print(z1[0,:,0], z1.min(),z1.max())
            new_z1 = z1
        else:
            with torch.no_grad():
                result = self.f_solver(func, z1, threshold=self.f_thres, stop_mode=self.stop_mode,cutoffs = cutoffs,shape=shape)
                z1 = result['result']
            new_z1 = z1
            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, \
                                                torch.zeros_like(grad), threshold=self.b_thres,cutoffs=cutoffs,shape=shape)['result']#
                    return new_grad
                self.hook = new_z1.register_hook(backward_hook)
        net = vec2list(new_z1, cutoffs)
        return net[-1], jac_loss.view(1,-1), result
    
    def forward(self, features):
        _,_,c,d = features[0].shape
        features_ = []
        for feature in features :
            feature_ = feature.view(feature.size(0),feature.size(1), -1)
            features_.append(feature_)
        fusion_feature = torch.stack([self.weights[i] * f for i, f in enumerate(features_)], dim=0).sum(dim=0) + self.bias
        fused_feat_hat, jacobian_loss, trace = self.featureFusion(features_, fusion_feature,shape=[c,d])
        fused_feat = fused_feat_hat.view(feature.size(0),feature.size(1),c,d)
        return fused_feat, jacobian_loss, trace




def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=0.05, stop_mode='rel', cutoffs=[],beta=1.0,shape=[],**kwargs, ):
    cut = cutoffs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {'abs': [(F[:, 0] - X[:, 0]).view_as(x0).norm().item(), (F[:, 1] - X[:, 1]).view_as(x0).norm().item()],
                  'rel': [(F[:, 0] - X[:, 0]).view_as(x0).norm().item() / (1e-5 + F[:, 0].norm().item()),
                          (F[:, 1] - X[:, 1]).view_as(x0).norm().item() / (1e-5 + F[:, 1].norm().item())]}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    success_count = 0
    lowest_xest = F[:,1].view_as(x0).clone().detach()
    fuse_maps_list = []
    for k in range(2,threshold ):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        F_list = vec2list(F[:, k % m].reshape_as(x0),cut)
        d_list = []
        for elem in F_list:
            elem_4d = elem.reshape(bsz,elem.shape[1],shape[0],shape[1])
            d_list.append(elem_4d)
        stop_eps = stop(d_list[2],d_list[1])  
        feature_fuse = d_list[2]
        fuse_maps_list.append(feature_fuse.clone().detach())

        if stop_eps < eps :
            lowest_xest = X[:, k % m].view_as(x0).clone().detach()
            success_count += 1
        else :
            success_count = 0
        if success_count == 3 :
            break


    out = {"result": lowest_xest,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "threshold": threshold,
           "fuse_maps": fuse_maps_list}
    X = F = None
    return out

def stop(imageA, imageB):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imageA ,imageB= imageA.to(device) , imageB.to(device)
    to_reduce_dim = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False).to(device)
    imageA_1,imageB_1 = to_reduce_dim(imageA), to_reduce_dim(imageB)
    converted_model = SiameseNetwork().to(device)
    ckpt_path = r'models/Discrimination_network.pth'
    converted_model.load_state_dict(torch.load(ckpt_path))
    with torch.no_grad():
        output = converted_model(imageA_1, imageB_1)
        score_map = F.softmax(output, dim=1)[:, 1]
        stop_eps = 1 - score_map
        stop_eps = stop_eps.mean().item()
        
    return stop_eps

def jac_loss_estimate(f0, z0, vecs=2, create_graph=True):
    vecs = vecs
    result = 0
    for i in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.norm()**2
    return result / vecs / np.prod(z0.shape)
