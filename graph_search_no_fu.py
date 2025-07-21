import torch

import torch.nn as nn
import torch.nn.functional as F
from .gumbel_rao import gumbel_rao,conditional_gumbel_qmc
from .operations import GraphMixedOp
from torch.autograd import Variable
# from torch.nn import Parameter



class GraphCell(nn.Module):
    def __init__(self,device,args):
        super().__init__()
        
        self.args = args
        
        self.node_steps = 1
        # self.node_multiplier = node_multiplier
        
        self.node_ops = nn.ModuleList()
        
        self.C = args.C
        # self.L = args.L
        self.O=args.O
        self.device=device
        #self.device=args.device
        node_op = GraphMixedOp(self.C, self.O, device=self.device, args=self.args)
        self.node_ops.append(node_op)

        # if self.node_multiplier != 1:
        #     self.out_conv = nn.Conv1d(self.C * self.node_multiplier, self.C, 1, 1)
        #     self.bn = nn.BatchNorm1d(self.C)
        #     self.out_dropout = nn.Dropout(args.drpt)

        # skip v3 and v4
        # self.ln = nn.LayerNorm([self.C, self.L])
        # self.dropout = nn.Dropout(args.drpt)

    def forward(self, input_features,node_weights):
        # init_state = self.node_ops[0](x, y, node_weights[0])
        # states.append(init_state)
        offset = 0
        
        # step_input_feature = sum(self.edge_ops[offset+j](h, edge_weights[offset+j]) for j, h in enumerate(states))
        s = self.node_ops[0](input_features, node_weights)
        # offset += len(states)
        # states.append(s)

        # out = torch.cat(states[-self.node_multiplier:], dim=1)
        # if self.node_multiplier != 1:
        #     out = self.out_conv(out)
        #     out = self.bn(out)
        #     out = F.relu(out)
        #     out = self.out_dropout(out)
        
        # # skip v4
        # out += x
        # out = self.ln(out)
        
        return s
    # def arch_parameters(self):
    #     pass


class GraphNetwork(nn.Module):
    def __init__(self, steps, multiplier, num_input_nodes, num_keep_edges, args, device,
                criterion=None, logger=None,):
        super().__init__()
        
        self.logger = logger
        self._steps = steps
        self._multiplier = multiplier
        self._criterion = criterion
        self.device=device

        # input node number in a cell
        self._num_input_nodes = num_input_nodes
        self._num_keep_edges = num_keep_edges
        
        self.k=args.k 
        self.temperature=args.temperature
        

        # self.cells = nn.ModuleList()
        self.cell = GraphCell(device,args).to(self.device)
        # self.cell_arch_parameters = self.cell.arch_parameters()
        # self.cells += [cell]
        self._initialize_zetas()
        self._arch_parameters = [self.zetas] 

    def compute_arch_entropy(self, dim=-1):
        alpha = self.arch_parameters()[0]
        #print(alpha.shape)
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy
    
    def forward(self, input_features):
        print("Graph Network forward -------------------------------------------------------------------------------")
        print(self._num_input_nodes,input_features.shape)
        assert self._num_input_nodes == input_features.shape[1]
        # changed to gumble rao gradient estimator 
        
        #weights = gumbel_rao(self.alphas_edges, k=100, temp=1500.0, adaptive_sampling=True, importance_sampling=True, learnable_temp=True)
 
        weights = gumbel_rao(self.zetas, k = self.k, temp = self.temperature)  
        weights=weights[0] # To make it a single list dukh mat pooch bhai
   
        #weights = F.softmax(self.alphas_edges, dim=-1)
        # print("Normal weights:",weights.shape)
        #weights=self.gumbel_softmax_sample(self.alphas_edges, num_sample = 20,temperature=10.0)
        out = self.cell(input_features, weights)
        out=out.to(self.device)
        return out


    def _initialize_zetas(self):
        k = 1
        num_ops = 2
        self.zetas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)  
        
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):  
        return self._arch_parameters

    