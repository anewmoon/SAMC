# Implementation based on https://github.com/PetarV-/DGI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from layers import IGATConv, ClusterLayer


class SAMC(nn.Module):

    def __init__(self, n_nb, n_in, n_h, activation, num_clusters, thresh_beta):
        
        super(SAMC, self).__init__()

        self.num_clusters = num_clusters
        self.act = activation
        self.thresh_beta = thresh_beta
        
        self.gene_encoder_l1 = IGATConv(n_in, 256, heads=1, concat=False, dropout=0.6)
        self.gene_encoder_l2 = IGATConv(256, n_h, heads=1, concat=False, dropout=0.6)
        self.gene_decoder_l1 = IGATConv(n_h, 256, heads=1, concat=False, dropout=0.6)
        self.gene_decoder_l2 = IGATConv(256, n_in, heads=1, concat=False, dropout=0.6)

        self.vision_encoder_l1 = IGATConv(n_in, 256, heads=1, concat=False, dropout=0.6)
        self.vision_encoder_l2 = IGATConv(256, n_h, heads=1, concat=False, dropout=0.6)
        self.vision_decoder_l1 = IGATConv(n_h, 256, heads=1, concat=False, dropout=0.6)
        self.vision_decoder_l2 = IGATConv(256, n_in, heads=1, concat=False, dropout=0.6)

        self.cl = ClusterLayer(n_h, num_clusters)
        self.dc = InnerProductDecoder(act=lambda x: x)
        
        self.gw = np.ones(n_nb) * 0.5
        self.vw = np.ones(n_nb) * 0.5

    def forward(self, seq1, vis_seq1, adj, sparse):
        
        h_1 = self.act(self.gene_encoder_l1(seq1[-1], adj))
        h_1 = self.gene_encoder_l2(h_1, adj, attention=True)

        vis_h_1 = self.act(self.vision_encoder_l1(vis_seq1[-1], adj))
        vis_h_1 = self.vision_encoder_l2(vis_h_1, adj, attention=True)


        w_gene = torch.from_numpy(self.gw).float().to(h_1.device).unsqueeze(1)
        w_vis = torch.from_numpy(self.vw).float().to(vis_h_1.device).unsqueeze(1)
        z_1 = w_gene * h_1 + w_vis * vis_h_1

        q = self.cl(z_1)  
        self.cl.labels = q
        
        with torch.no_grad():
            p = q**2 / torch.sum(q, 0)
            p = torch.t(torch.t(p) / torch.sum(p, 1))
        
        kl_loss_func = nn.KLDivLoss(reduction='batchmean')
        loss_p = kl_loss_func(q.log(), p)

        current_labels = q.argmax(1).detach().cpu().numpy()
        with torch.no_grad():
            try:
                if len(np.unique(current_labels)) > 1:
                    s_h = silhouette_samples(h_1.detach().cpu().numpy(), current_labels)
                    s_vis = silhouette_samples(vis_h_1.detach().cpu().numpy(), current_labels)
                    
                    s_h_norm = s_h + 1  
                    s_vis_norm = s_vis + 1
                    
                    total = s_h_norm + s_vis_norm
                    total[total == 0] = 1e-6  # Avoid division by zero
                    
                    self.gw = s_h_norm / total
                    self.vw = s_vis_norm / total
            except Exception:
                pass

        with torch.no_grad():
            hard_labels = F.one_hot(torch.argmax(q, 1), self.num_clusters)
            masked_q = hard_labels * q
            self.cl.centers = torch.mm(torch.t(masked_q / torch.sum(masked_q, 0)), z_1).detach()

        
        self.gene_decoder_l1.lin_src.data = self.gene_encoder_l2.lin_src.transpose(0, 1)
        self.gene_decoder_l2.lin_src.data = self.gene_encoder_l1.lin_src.transpose(0, 1)
        self.vision_decoder_l1.lin_src.data = self.vision_encoder_l2.lin_src.transpose(0, 1)
        self.vision_decoder_l2.lin_src.data = self.vision_encoder_l1.lin_src.transpose(0, 1)

        h_2 = self.act(self.gene_decoder_l1(h_1, adj, attention=True, tied_attention=self.gene_encoder_l2.attentions))
        h_2 = self.gene_decoder_l2(h_2, adj, attention=True, tied_attention=self.gene_encoder_l1.attentions)
        
        vis_h_2 = self.act(self.vision_decoder_l1(vis_h_1, adj, attention=True, tied_attention=self.vision_encoder_l2.attentions))
        vis_h_2 = self.vision_decoder_l2(vis_h_2, adj, attention=True, tied_attention=self.vision_encoder_l1.attentions)

        mse_loss = nn.MSELoss()
        loss_r = mse_loss(h_2, seq1[-1])
        vis_loss_r = mse_loss(vis_h_2, vis_seq1[-1])

        
        high_confidence, _ = thresholding(q.cpu(), beta_2=self.thresh_beta)
        imax = q.cpu().detach().numpy().argmax(1)
        imax[~np.isin(np.arange(len(imax)), high_confidence)] = -1 # Mark low-confidence as -1
        
        length = vis_seq1[-1].shape[0]
        adj_tar = np.zeros((length, length))
        for i in range(length):
            if imax[i] != -1:
                adj_tar[i][imax == imax[i]] = 1
        np.fill_diagonal(adj_tar, 0)
        cross_tar = np.copy(adj_tar)
        np.fill_diagonal(cross_tar, 1)
        
        tar_adj = torch.Tensor(adj_tar).to(h_1.device)
        tar_cross = torch.Tensor(cross_tar).to(h_1.device)
        
        norm_h_1 = F.normalize(h_1, p=2, dim=1)
        norm_vis_h_1 = F.normalize(vis_h_1, p=2, dim=1)
        
        comp = self.dc(norm_h_1)
        vis_comp = self.dc(norm_vis_h_1)
        cross_comp = self.dc(norm_h_1, norm_vis_h_1)
        
        loss_f = F.binary_cross_entropy_with_logits(comp, tar_adj)
        vis_loss_f = F.binary_cross_entropy_with_logits(vis_comp, tar_adj)
        cross_loss_f = F.binary_cross_entropy_with_logits(cross_comp, tar_cross)

        try:
            if len(np.unique(current_labels)) < 2:
                scs = -1.0
            else:
                scs = silhouette_score(z_1.detach().cpu().numpy(), current_labels)
        except Exception:
            scs = -1.0
            
        return loss_p, loss_r, vis_loss_r, loss_f, vis_loss_f, cross_loss_f, z_1, current_labels, scs

    def embed(self, seq, vis_seq, adj, sparse, return_att=False):
        
        h_1 = self.act(self.gene_encoder_l1(seq[-1], adj))
        vis_h_1 = self.act(self.vision_encoder_l1(vis_seq[-1], adj))

        if return_att:
            h_1, gene_att = self.gene_encoder_l2(h_1, adj, return_attention_weights=True)
            vis_h_1, vis_att = self.vision_encoder_l2(vis_h_1, adj, return_attention_weights=True)
            return h_1.detach(), vis_h_1.detach(), gene_att, vis_att
        else:
            h_1 = self.gene_encoder_l2(h_1, adj, attention=True)
            vis_h_1 = self.vision_encoder_l2(vis_h_1, adj, attention=True)
            return h_1.detach(), vis_h_1.detach()

    def pretrain(self, seq1, vis_seq1, adj, sparse):
        
        h_1 = self.act(self.gene_encoder_l1(seq1[-1], adj))
        h_1 = self.gene_encoder_l2(h_1, adj, attention=True)
        vis_h_1 = self.act(self.vision_encoder_l1(vis_seq1[-1], adj))
        vis_h_1 = self.vision_encoder_l2(vis_h_1, adj, attention=True)

        self.gene_decoder_l1.lin_src.data = self.gene_encoder_l2.lin_src.transpose(0, 1)
        self.gene_decoder_l2.lin_src.data = self.gene_encoder_l1.lin_src.transpose(0, 1)
        self.vision_decoder_l1.lin_src.data = self.vision_encoder_l2.lin_src.transpose(0, 1)
        self.vision_decoder_l2.lin_src.data = self.vision_encoder_l1.lin_src.transpose(0, 1)
        
        h_2 = self.act(self.gene_decoder_l1(h_1, adj, attention=True, tied_attention=self.gene_encoder_l2.attentions))
        h_2 = self.gene_decoder_l2(h_2, adj, attention=True, tied_attention=self.gene_encoder_l1.attentions)
        vis_h_2 = self.act(self.vision_decoder_l1(vis_h_1, adj, attention=True, tied_attention=self.vision_encoder_l2.attentions))
        vis_h_2 = self.vision_decoder_l2(vis_h_2, adj, attention=True, tied_attention=self.vision_encoder_l1.attentions)
        
        return h_2, vis_h_2


class InnerProductDecoder(nn.Module):

    def __init__(self, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.act = act

    def forward(self, z, zt=None):
        
        if zt is None:
            adj = self.act(torch.mm(z, z.t()))
        else:
            adj = self.act(torch.mm(z, zt.t()))
        return adj


def thresholding(p, beta_1=0, beta_2=0.7):
    
    p_np = p.detach().numpy()
    
    confidence = p_np.max(1)
    
    high_conf_indices = np.where(confidence > beta_2)[0]
    low_conf_indices = np.where(confidence <= beta_2)[0]
    
    return high_conf_indices, low_conf_indices