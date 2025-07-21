#NTU_darts_searchable.py
import torch
import torch.nn as nn
import torch.optim as op
import os

import models.auxiliary.scheduler as sc
import models.auxiliary.aux_models as aux
import models.auxiliary.gcn as gcn
import models.central.ntu as ntu
import models.search.train_searchable.ntu as tr

from IPython import embed
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# class GCNContextualExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNContextualExtractor, self).__init__()
#         self.conv1 = GCNConv(in_channels, 128)
#         self.conv2 = GCNConv(128, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x


from .darts.model_search import FusionNetwork
from .darts.model import Found_FusionNetwork
from models.search.plot_genotype import Plotter
from .darts.architect import Architect
import torch.nn.functional as F

from .darts.node_operations import *

def train_darts_model(dataloaders, args, device, logger):
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'dev', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batchsize
    criterion = torch.nn.CrossEntropyLoss()

    # model to train
    model = Searchable_Skeleton_Image_Net(args, criterion, logger,device)
    
    if args.resume_from_checkpoint:
        #print("Loading weights...")
        model_dir = "./BM-NAS-master_experimental/train_model_final/gumble_rao/k10/"
        pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
        max_epoch_file = max(pt_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
        #print(os.path.join(model_dir, max_epoch_file))
        state_dict = torch.load(os.path.join(model_dir, max_epoch_file))
        model.load_state_dict(state_dict)
            
    #print("Loading the train_darts_model completed !!!!!!")
    # loading pretrained weights

    # skemodel_filename = os.path.join(args.checkpointdir, args.ske_cp)
    # rgbmodel_filename = os.path.join(args.checkpointdir, args.rgb_cp)

    # model.skenet.load_state_dict(torch.load(skemodel_filename))
    # model.rgbnet.load_state_dict(torch.load(rgbmodel_filename))
    # parameters to update during training

    params = model.central_params()

    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=args.weight_decay)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    print(model.arch_parameters()[0].shape, "hehehe......................................................")

    arch_optimizer = op.Adam(model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    
    ## Commented as args.parallel not found
    model.to(device)  # Changed by us !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 😭
    # hardware tuning
    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model,device_ids=[0,1,2])
        model.to('cuda')

        # model = torch.nn.parallel.DistributedDataParallel(model)
        model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer)

    plotter = Plotter(args)
    best_genotype = tr.train_ntu_track_acc(model, architect,
                                            criterion, optimizer, scheduler, dataloaders,
                                            dataset_sizes,
                                            device=device,
                                            num_epochs=args.epochs,
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args)


    return best_genotype



# class GCNContextualExtractor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNContextualExtractor, self).__init__()
#         self.conv1 = GCNConv(in_channels, 128)
#         self.conv2 = GCNConv(128, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x


class   Searchable_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, logger,device):
        super(Searchable_Skeleton_Image_Net, self).__init__()

        self.args = args
        self.criterion = criterion
        self.logger = logger
        self.device=device

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Spectrum(args)
        
        # Initialize the GCN-based contextual embedding extractor
        #self.gcn_extractor = gcn.GCNContextualExtractor(in_channels=8, out_channels=8)

        # self.gp_v, self.gp_s = self._create_global_poolings()
        # self.gp_v, self.gp_s = self._create_global_poolings(args)
        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        # self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self.fusion_net = FusionNetwork( steps=self.steps, multiplier=self.multiplier,
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,device=self.device,
                                         criterion=self.criterion,
                                         logger=self.logger).to(self.device)

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [128, 256, 512, 512, 128, 256, 512, 512]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            #print("input feature in reshape_input_feature:",input_feature.shape)
            reshaped_feature = self.reshape_layers[i](input_feature)
            #print("Reshaped Feature shape:",reshaped_feature.shape)
            ret.append(reshaped_feature)
        return ret
    
    def prepare_feature_matrix(self, skeleton_features, spectrum_features):
    # Concatenate skeletal and spectrum features along the feature dimension
        all_features = list(skeleton_features) + list(spectrum_features)
        #print("All_features",len(all_features))
        #print("All_features",len(all_features))
        feature_matrix = torch.cat(all_features, dim=1)  # Shape: (num_nodes, num_features)
        return feature_matrix
    
    def create_fully_connected_edge_index(self,num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index).t().contiguous()  # Shape: (2, num_edges)
        return edge_index
    
    def _gcn_contextual_extractor(self,x, edge_index, in_channels, out_channels):
        #print("SELF_device",self.device)
        try:
            pass
            #in_channels = int(in_channels)
            #out_channels = int(out_channels)
        except ValueError:
            raise ValueError(f"in_channels or out_channels cannot be converted to integers. Got types: {type(in_channels)}, {type(out_channels)}")
        x=x.to(self.device)
        edge_index=edge_index.to(self.device)
    
        
        conv1 = GCNConv(in_channels, 128).to(self.device)
        conv2 = GCNConv(128, out_channels).to(self.device)
        
        x = conv1(x, edge_index)
        x = F.relu(x)
        x = conv2(x, edge_index)
        
        return x


    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]
        
        skel_features = self.skenet(skeleton)
        # visual_classifier = visual_features[-1]
        skel_features = skel_features[-5:-1]
        input_features = list(visual_features) + list(skel_features)

        
        input_features = self.reshape_input_features(input_features)
        feature_matrix=torch.cat(input_features,dim=1)
        
        

        # # Create the edge index (here using a fully connected graph)
        edge_index = self.create_fully_connected_edge_index(num_nodes=8)
        # #print("Edge_index",edge_index)

        # # Extract contextual embeddings using the GCN
        # # Move the model to the device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.gcn_extractor = self.gcn_extractor.to(device)
        in_channels=1024
        out_channels=128
        feature_matrix = feature_matrix.transpose(-2, -1)
        contextual_embeddings = self._gcn_contextual_extractor(feature_matrix, edge_index, in_channels, out_channels)
    
        contextual_embeddings = contextual_embeddings.transpose(-2, -1)
        
        # Combine contextual embeddings with original input features
        
        # combined_tensor = torch.cat((tensor1, tensor2), dim=dim)
        # Added the contextual embedding by broadcasting it to every channel
    
        input_features_new=[test.unsqueeze(0) for test in input_features]
        # input_features_new=torch.cat(dim=-1)
        features_new=torch.cat(input_features_new,dim=0)

        
        contextual_embeddings=contextual_embeddings.unsqueeze(0)
        contextual_tensor_expanded = contextual_embeddings.expand(8, -1, -1, -1)  # Expand to [8, 16, 128, 8]
        result_tensor = features_new + contextual_tensor_expanded
        combined_features = list(torch.unbind(result_tensor, dim=0))
    
        # combined_features = input_features + [contextual_embeddings]  # Previous Combined Features
        
        

        combined_features = [x.to(self.device) for x in combined_features] # Ensure combined_features is on the correct device

        out = self.fusion_net(input_features)
       
        # print("Wanted Feature:",out.shape)
        # print("---------------------------------------")
        out = self.central_classifier(out)

        return out
    

    def genotype(self):
        return self.fusion_net.genotype()

    def central_params(self):
        central_parameters = [
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

    def arch_parameters(self):
        return self.fusion_net.arch_parameters()

    # def _create_global_poolings(self, args):
    #     gp_list_v = [aux.GlobalPadPooling2D(args) for i in range(4)]
    #     gp_list_s = [aux.GlobalPadPooling2D(args) for i in range(4)]
    #     return nn.ModuleList(gp_list_v), nn.ModuleList(gp_list_s)

    def _create_global_poolings(self, args):
        gp_list_v = [aux.Global_Pool_FC() for i in range(4)]
        gp_list_s = [aux.Global_Pool_FC() for i in range(4)]
        return nn.ModuleList(gp_list_v), nn.ModuleList(gp_list_s)

class Found_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Spectrum(args)
        self._genotype = genotype

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        # self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion
        self.flag = 0
        self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion,
                                         genotype=self._genotype)

        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args):
        C_ins = [128, 256, 512, 512, 128, 256, 512, 512]
        reshape_layers = nn.ModuleList()

        input_nodes = []
        for edge in self._genotype.edges:
            input_nodes.append(edge[1])
        input_nodes = list(set(input_nodes))

        for i in range(len(C_ins)):
            if i in input_nodes:
                reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
            else:
                reshape_layers.append(nn.ReLU())

        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features = self.skenet(skeleton)
        skel_features = skel_features[-5:-1]

        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)
        
        out = self.fusion_net(input_features)

        # print("out shape:", out.shape)
        # embed()
        if self.flag == 0:
            #print("---------------------------------------")
            #print("Input feature shape count:",len(input_features))
            # for i in range(len(input_features)):
            #     print(input_features[i].shape)
            # print("Out feature shape count:",len(out))
            # for i in range(len(out)):
            #     print(out[i].shape)
            # print("---------------------------------------")
            self.flag =1
        out = self.central_classifier(out)
        if self.flag == 1:
            # print("---------------------------------------")
            # for i in range(len(out)):
            #     #print(out[i].shape)
            #     print(out[i])
            # print("---------------------------------------")
            self.flag =2
        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Simple_Concat_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Spectrum(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        # self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 2, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, _ = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        # v2 = input_features[2]
        v3 = input_features[3]
        s3 = input_features[7]

        out = torch.cat([v3, s3], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Ensemble_Concat_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Spectrum(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        # self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        # self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
        #                                     args.num_outputs)
        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 5, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512, 60, 60]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        # visual_features = visual_features[-5:-1]
        v_logits = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, s_logits = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features) + [v_logits, s_logits]
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        v2 = input_features[2]
        v3 = input_features[3]
        s3 = input_features[7]

        v_logits = input_features[8]
        s_logits = input_features[9]

        # embed()
        # exit(0)

        out = torch.cat([v2, v3, s3, v_logits, s_logits], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)

        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Ensemble_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Spectrum(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        # self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        # self.central_classifier = nn.Linear(self.args.C * self.args.L * 2,
        #                                     args.num_outputs)

        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 2, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512, 60, 60]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        # visual_features = visual_features[-5:-1]
        v_logits = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, s_logits = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features) + [v_logits, s_logits]
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        # v2 = input_features[2]
        # v3 = input_features[3]
        # s3 = input_features[7]

        v_logits = input_features[8]
        s_logits = input_features[9]

        # embed()
        # exit(0)

        out = torch.cat([v_logits, s_logits], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)

        out = self.central_classifier(out)

        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

class Found_Simple_Concat_Attn_Skeleton_Image_Net(nn.Module):
    def __init__(self, args, criterion, genotype):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.rgbnet = ntu.Visual(args)
        self.skenet = ntu.Spectrum(args)

        self.reshape_layers = self.create_reshape_layers(args)
        # self.fusion_layers = self._create_fc_layers()
        self.multiplier = args.multiplier
        self.steps = args.steps
        # self.parallel = args.parallel

        self.num_input_nodes = 8
        self.num_keep_edges = 2
        self._criterion = criterion

        self._genotype = None
        # self.fusion_net = Found_FusionNetwork( steps=self.steps, multiplier=self.multiplier,
        #                                  num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
        #                                  args=self.args,
        #                                  criterion=self.criterion,
        #                                  genotype=self._genotype)

        self.fusion_net = nn.ReLU()

        self.attn1 = ScaledDotAttn()
        self.attn2 = ScaledDotAttn()

        self.central_classifier = nn.Sequential(
            nn.Linear(self.args.C * self.args.L * 2, self.args.C),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.C),
            nn.Linear(self.args.C, args.num_outputs)
        )

    def create_reshape_layers(self, args):
        C_ins = [512, 1024, 2048, 2048, 128, 256, 1024, 512]
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer(C_ins[i], args.C, args.L))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):

        skeleton, image = tensor_tuple[1], tensor_tuple[0]

        # apply net on input image
        visual_features = self.rgbnet(image)
        # visual_classifier = visual_features[-1]
        visual_features = visual_features[-5:-1]

        # apply net on input skeleton
        skel_features, _ = self.skenet(skeleton)
        skel_features = skel_features[-4:]

        input_features = list(visual_features) + list(skel_features)
        input_features = self.reshape_input_features(input_features)

        # concat most important features v2, v3, s3, aka: 2, 3, 7
        # v2 = input_features[2]
        v3 = input_features[3]
        s3 = input_features[7]

        out1 = self.attn1(v3, s3)
        out2 = self.attn2(s3, v3)

        out = torch.cat([out1, out2], dim=1)
        # out = self.fusion_net(input_features)
        # print("out shape:", out.shape)
        # embed()
        # exit()
        out = out.view(out.size(0), -1)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self._genotype

    def central_params(self):
        central_parameters = [
            # {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)
