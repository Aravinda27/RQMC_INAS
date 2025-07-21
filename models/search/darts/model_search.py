import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .operations import *
from .genotypes import PRIMITIVES, Genotype
from .node_search import FusionNode
from .gumbel_rao import gumbel_rao


from IPython import embed

class FusionCell(nn.Module):
    def __init__(self, steps, multiplier, args,device):
        super(FusionCell, self).__init__()

        self._steps = steps
        self._multiplier = multiplier
        self._device=device
        self._ops = nn.ModuleList().to(self._device)
        self.args = args

        self._step_nodes = nn.ModuleList()
        self.num_input_nodes = args.num_input_nodes
        self.C = args.C
        self.L = args.L
        self.device=device
        self.ln = nn.LayerNorm([self.C * self._multiplier, self.L])

        # input features is a joint list of visual_features and skel_features
        for i in range(self._steps):
            for j in range(self.num_input_nodes+i):
                op = FusionMixedOp(self.C, self.L, self.args,self._device)
                self._ops.append(op)
        
        self._initialize_step_nodes(args)

    def _initialize_step_nodes(self, args):
        for i in range(self._steps):
            num_input = self.num_input_nodes + i
            # step_node = AttentionSumNode(args, num_input)
            step_node = FusionNode(args.node_steps, args.node_multiplier, self._device,args)
            self._step_nodes.append(step_node)

    def arch_parameters(self):
        self._arch_parameters = []
        for i in range(self._steps):
            self._arch_parameters += self._step_nodes[i].arch_parameters()
        return self._arch_parameters
    

    def forward(self, input_features, weights):
        # input_features = input_features.to(self.device)
        # weights = weights.to(self.device)
        states = []
        for input_feature in input_features:
            states.append(input_feature)
        print("KONA",len(states))

        offset = 0
        for i in range(self._steps):
            step_input_features = []
            for j, h in enumerate(states):
                h = h.to(self.device)  # Ensure input tensor h is on the correct device
                weights[offset + j] = weights[offset + j].to(self.device) 
                op_output = self._ops[offset + j](h, weights[offset + j])
                print(f"Step {i}, operation {j} output shape: {op_output.shape}")
                step_input_features.append(op_output)
            # Ensure all step_input_features have the same shape before summing
            if len(set([s.shape for s in step_input_features])) > 1:
                raise ValueError(f"Shape mismatch in step {i}! Got shapes {[s.shape for s in step_input_features]}")
            
            step_input_feature = sum(step_input_features)
            print(f"Shape of step_input_feature at step {i}: {step_input_feature.shape}")
            # Ensure all step_input_features have the same shape before summing
           
            
            
            s = self._step_nodes[i](step_input_feature, step_input_feature)
            offset += len(states)
            states.append(s)
    
        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.ln(out)
        out = F.relu(out)
    
        out = out.view(out.size(0), -1)
        return out

class FusionNetwork(nn.Module):

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
        self._num_input_nodes=4
        print("Self_nodes---------------",self._num_input_nodes)
        self._num_keep_edges = num_keep_edges

        # self.cells = nn.ModuleList()
        self.cell = FusionCell(steps, multiplier, args,device).to(self.device)
        self.cell_arch_parameters = self.cell.arch_parameters()
        # self.cells += [cell]

        self._initialize_alphas()
        self._arch_parameters = [self.alphas_edges] + self.cell_arch_parameters

    def compute_arch_entropy(self, dim=-1):
        alpha = self.arch_parameters()[0]
        #print(alpha.shape)
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy
    
    def forward(self, input_features):
        assert self._num_input_nodes == len(input_features)
        # changed to gumble rao gradient estimator 
        
        weights = gumbel_rao(self.alphas_edges, k=1000, temp=5.0, adaptive_sampling=True, importance_sampling=True, learnable_temp=True)

        #weights = gumbel_rao(self.alphas_edges, k = 10, temp = 0.1)
        #weights = F.softmax(self.alphas_edges, dim=-1)
        # print("Normal weights:",weights.shape)
        #weights=self.gumbel_softmax_sample(self.alphas_edges, num_sample = 20,temperature=10.0)
        out = self.cell(input_features, weights)
        out=out.to(self.device)
        return out


    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(self._num_input_nodes+i))
        num_ops = len(PRIMITIVES)
        self.alphas_edges = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):  
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            # n = 2
            n = self._num_input_nodes
            start = 0

            # force non_repeat node pairs
            selected_edges = []
            selected_nodes = []

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # alpha edges, only link two most important nodes
                # edges = sorted(range(i + self._num_input_nodes), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:self._num_keep_edges]
                # from_list = list(range(i + self._num_input_nodes))
                
                # sample strategy v3
                from_list = list(range(self._num_input_nodes))

                node_pairs = []
                for j_index, j in enumerate(from_list):
                    for k in from_list[j_index+1:]:
                        # if [j, k] not in selected_edges:
                        if (j not in selected_nodes) or (k not in selected_nodes):

                            W_j_max = max(W[j][t] for t in range(len(W[j])) if t != PRIMITIVES.index('none'))
                            W_k_max = max(W[k][t] for t in range(len(W[k])) if t != PRIMITIVES.index('none'))

                            node_pairs.append([j, k, W_j_max * W_k_max])

                selected_node_pair = sorted(node_pairs, key=lambda x: -x[2])[:1][0]
                edges = selected_node_pair[0:2]
                selected_edges.append(edges)
                selected_nodes += edges
                selected_nodes = list(set(selected_nodes))
                #  choose the most important operation
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                    # gene.append((PRIMITIVES[k_second_best], j))
                start = end
                n += 1

            return gene

        def _parse_step_nodes():
            gene_steps = []
            for i in range(self._steps):
                step_node_genotype = self.cell._step_nodes[i].node_genotype()
                gene_steps.append(step_node_genotype)
            return gene_steps
        
        # beta edges
        gene_edges = _parse(F.softmax(self.alphas_edges, dim=-1).data.cpu().numpy())
        gene_steps = _parse_step_nodes()
        
        gene_concat = range(self._num_input_nodes+self._steps-self._multiplier, self._steps+self._num_input_nodes)
        gene_concat = list(gene_concat)

        genotype = Genotype(
            edges=gene_edges, 
            concat=gene_concat,
            steps=gene_steps
        )

        return genotype