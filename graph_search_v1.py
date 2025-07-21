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

class GraphNodeCell(nn.Module):
    def __init__(self, node_steps, node_multiplier,device,args):
        super().__init__()
        
        self.args = args
        
        self.node_steps = 2 # Changed on 20/11 initial 2
        self.node_multiplier = node_multiplier
        
        self.edge_ops = nn.ModuleList()
        self.node_ops = nn.ModuleList()
        
        self.C = args.C
        self.O=args.O
        self.device=device
        #self.device=args.device
        
        self.num_input_nodes = 2
        # self.num_keep_edges = 2

        for i in range(self.node_steps):
            for j in range(self.num_input_nodes+i):
                print(i,j,"test1234")
                # C, L, args,device
                edge_op = FusionMixedOp(self.C, self.O,self.args, device=self.device)
                # print(edge_op,"test123----------------------------")
                self.edge_ops.append(edge_op)
        
        print('Nikal gaye hurray')        
        # for i in range(self.node_steps):  # Commented out on 20 NOV
        # C, O, device,args
        node_op = GraphMixedOp(self.C, self.O, device=self.device, args=self.args)
        self.node_ops.append(node_op)

        # if self.node_multiplier != 1:
        #     self.out_conv = nn.Conv1d(self.C * self.node_multiplier, self.C, 1, 1)
        #     self.bn = nn.BatchNorm1d(self.C)
        #     self.out_dropout = nn.Dropout(args.drpt)

        # skip v3 and v4
        # self.ln = nn.LayerNorm([self.C, self.L])
        # self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y, edge_weights, node_weights):
        states = [x, y]
        # init_state = self.node_ops[0](x, y, node_weights[0])
        # states.append(init_state)
        offset = 0
        for i in range(self.node_steps):
            step_input_feature = sum(self.edge_ops[offset+j](h, edge_weights[offset+j]) for j, h in enumerate(states))
            s = self.node_ops[i](step_input_feature, step_input_feature, node_weights[i])
            offset += len(states)
            states.append(s)

        out = torch.cat(states[-self.node_multiplier:], dim=1)
        
        # if self.node_multiplier != 1:
        #     out = self.out_conv(out)
        #     out = self.bn(out)
        #     out = F.relu(out)
        #     out = self.out_dropout(out)
        
        # skip v4
        out += x
        # out = self.ln(out)
        
        return out

class GraphNode(nn.Module):
    
    def __init__(self, node_steps, node_multiplier,device, args):
        super().__init__()
        # self.logger = logger
        self.node_steps = node_steps
        self.node_multiplier = node_multiplier
        self.node_cell = GraphNodeCell(node_steps, node_multiplier,device, args)

        self.num_input_nodes = 2 
        self.num_keep_edges = 2
        
        self._initialize_betas()
        self._initialize_gammas()

        self._arch_parameters = [self.betas, self.gammas]
        
    def _initialize_betas(self):
        k = sum(1 for i in range(self.node_steps) for n in range(self.num_input_nodes+i))
        num_ops = len(STEP_EDGE_PRIMITIVES)
        # beta controls node cell arch
        self.betas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    
    def _initialize_gammas(self):
        k = sum(1 for i in range(self.node_steps))
        num_ops = len(STEP_STEP_PRIMITIVES)
        # gamma controls node_step_nodes arch
        self.gammas = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    
    def forward(self, x, y):
        # Applied Gradient rao estimator
       
        
        # edge_weights= gumbel_rao(self.betas, k=1000, temp=5.0, adaptive_sampling=True, importance_sampling=True, learnable_temp=True)
        # node_weights= gumbel_rao(self.gammas, k=1000, temp=5.0, adaptive_sampling=True, importance_sampling=True, learnable_temp=True)
        edge_weights = gumbel_rao(self.betas, k = 5, temp=0.1)  #K=5 from 10
        node_weights = gumbel_rao(self.gammas, k = 2, temp = 0.1)
        # edge_weights = F.softmax(self.betas, dim=-1)
        # node_weights = F.softmax(self.gammas, dim=-1)
        out = self.node_cell(x, y, edge_weights, node_weights)        
        return out

    def compute_arch_entropy_gamma(self, dim=-1):
        alpha = self.arch_parameters()[0]
        #print(alpha.shape)
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy

    def arch_parameters(self):  
        return self._arch_parameters

    def node_genotype(self):
        def _parse(edge_weights, node_weights):
            edge_gene = []
            node_gene = []

            n = 2
            start = 0
            for i in range(self.node_steps):
                end = start + n
                
                W = edge_weights[start:end]
                edges = sorted(range(i + self.num_input_nodes), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:self.num_keep_edges]
                
                # print("edges:", edges)
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != STEP_EDGE_PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    edge_gene.append((STEP_EDGE_PRIMITIVES[k_best], j))
                    # gene.append((PRIMITIVES[k_second_best], j))

                start = end
                n += 1
                
            for i in range(self.node_steps):
                W = node_weights[i]
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k

                node_gene.append((STEP_STEP_PRIMITIVES[k_best]))

            return edge_gene, node_gene

        concat_gene = range(self.num_input_nodes+self.node_steps-self.node_multiplier, self.node_steps+self.num_input_nodes)
        concat_gene = list(concat_gene)

        edge_weights = F.softmax(self.betas, dim=-1)
        node_weights = F.softmax(self.gammas, dim=-1)
        
        edge_gene, node_gene = _parse(edge_weights, node_weights)

        fusion_gene = StepGenotype(
            inner_edges = edge_gene,
            inner_steps = node_gene,
            inner_concat = concat_gene,
        )
        # print(concat_gene)
        # print(edge_gene)
        # print(node_gene)
        return fusion_gene

class GraphCell(nn.Module):
    def __init__(self, steps, multiplier, args,device):
        super().__init__()

        self._steps = steps
        self._multiplier = multiplier
        self._device=device
        self._ops = nn.ModuleList().to(self._device)
        self._ops_graph = nn.ModuleList().to(self._device)
        self.args = args

        self._step_nodes = nn.ModuleList()
        self.num_input_nodes = args.num_input_nodes
        self.C = args.C
        self.L = args.L
        self.device=device
        self.ln = nn.LayerNorm([self.C * self._multiplier, self.L])

        # # input features is a joint list of visual_features and skel_features
        # For GraphMixedOp, we need to pass the concatenated visual_features and skel_features from the backbone
        for i in range(self._steps):
            for j in range(self.num_input_nodes+i):
                op = GraphMixedOp(self.C * 8, self.C,self._device,args)
                self._ops_graph.append(op)
        
        self._initialize_step_nodes(args)

    def _initialize_step_nodes(self, args):
        # for i in range(self._steps): Changed on 20/11
        num_input = self.num_input_nodes #+ i
        # step_node = AttentionSumNode(args, num_input)
        step_node = GraphNode(args.node_steps, args.node_multiplier, self._device,args)  #Changed on 20/11
        self._step_nodes.append(step_node)

    def arch_parameters(self):
        self._arch_parameters = []
        # for i in range(self._steps): # Changed on 20/11
        self._arch_parameters+=self._step_nodes[0].arch_parameters() # Changed on 20/11 earlier +=self._step_nodes.arch_parameters()
        return self._arch_parameters
    

    def forward(self, input_features, weights):
        # input_features = input_features.to(self.device)
        weights = weights.to(self.device)
        states = []
        for input_feature in input_features:
            states.append(input_feature)
        # print("KONA",len(states))

        offset = 0
        for i in range(self._steps):
            step_input_features = []
       
            for j, h in enumerate(states):
                h = h.to(self.device)
                weights[offset + j] = weights[offset + j].to(self.device) 
                op_output = self._ops_graph[offset + j](h, weights[offset + j])
                step_input_features.append(op_output)
                
            # Ensure all step_input_features have the same shape before summing
            if len(set([str(s.shape) for s in step_input_features])) > 1:
                raise ValueError(f"Shape mismatch in step {i}! Got shapes {[s.shape for s in step_input_features]}")
            
            step_input_feature = sum(step_input_features)
            # Ensure all step_input_features have the same shape before summing
           
            
            
            s = self._step_nodes[i](step_input_feature, step_input_feature)
            offset += len(states)
            states.append(s)
    
        out = torch.cat(states[-self._multiplier:], dim=1)
        out = self.ln(out)
        out = F.relu(out)
    
        out = out.view(out.size(0), -1)
        return out
    
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

        # self.cells = nn.ModuleList()
        self.cell = GraphCell(steps, multiplier, args,device).to(self.device)
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
        print("Graph Network forward -------------------------------------------------------------------------------")
        print(self._num_input_nodes,input_features.shape)
        assert self._num_input_nodes == input_features.shape[1]
        # changed to gumble rao gradient estimator 
        
        #weights = gumbel_rao(self.alphas_edges, k=100, temp=1500.0, adaptive_sampling=True, importance_sampling=True, learnable_temp=True)
 
        weights = gumbel_rao(self.alphas_edges, k = 100, temp = 0.1)  
   
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

