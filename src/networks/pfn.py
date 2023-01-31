from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from config import DEVICE
import numpy as np
from config import * 
import random
import os
import math
import copy
from utils.utils import *

def cumsoftmax(x):
    return torch.cumsum(F.softmax(x,-1),dim=-1)


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.bool
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)



class pfn_unit(nn.Module):
    def __init__(self, input_size, drop=0.1):
        super(pfn_unit, self).__init__()

        self.hidden_transform = LinearDropConnect(300, 5 * 300, bias=True, dropout= drop)
        self.input_transform = nn.Linear(input_size, 5 * 300, bias=True)

        self.transform = nn.Linear(300*3, 300)
        self.drop_weight_modules = [self.hidden_transform]
            
        torch.nn.init.orthogonal_(self.input_transform.weight, gain=1)
        torch.nn.init.orthogonal_(self.transform.weight, gain=1)


    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


    def forward(self, x, hidden):
        h_in, c_in = hidden

        gates = self.input_transform(x) + self.hidden_transform(h_in)
        c, eg_cin, rg_cin, eg_c, rg_c = gates[:, :].chunk(5, 1)

        eg_cin = 1 - cumsoftmax(eg_cin)
        rg_cin = cumsoftmax(rg_cin)

        eg_c = 1 - cumsoftmax(eg_c)
        rg_c = cumsoftmax(rg_c)

        c = torch.tanh(c)

        overlap_c = rg_c * eg_c
        upper_c = rg_c - overlap_c
        downer_c = eg_c - overlap_c

        overlap_cin =rg_cin * eg_cin
        upper_cin = rg_cin - overlap_cin
        downer_cin = eg_cin - overlap_cin

        share = overlap_cin * c_in + overlap_c * c

        c_cau = upper_cin * c_in + upper_c * c 
        c_emo = downer_cin * c_in + downer_c * c 
        c_share = share

        h_cau = torch.tanh(c_cau)
        h_emo = torch.tanh(c_emo)
        h_share = torch.tanh(c_share)

        
        c_out = torch.cat((c_cau, c_emo, c_share), dim=-1)
        c_out_2 = self.transform(c_out)
        h_out = torch.tanh(c_out_2)


        return (h_out, c_out_2), (h_emo, h_cau, h_share), c_out_2

class encoder(nn.Module):
    def __init__(self, input_size, drop=0.1):
        super(encoder, self).__init__()
        self.unit = pfn_unit(input_size, drop)

    def hidden_init(self, batch_size):
        h0 = torch.zeros(batch_size,300).requires_grad_(False).to(DEVICE)
        c0 = torch.zeros(batch_size, 300).requires_grad_(False).to(DEVICE)
        return (h0, c0)

    def forward(self, x, layers=1):
        seq_len = x.size(0)
        batch_size = x.size(1)
        h_emo, h_cau, h_share = [], [], []
        if self.training:
            self.unit.sample_masks()
            
        for layer in range(layers):
            hidden = self.hidden_init(batch_size)
            output = []
            for t in range(seq_len):
                hidden, h_task, output_layer = self.unit(x[t, :, :], hidden)
                output.append(output_layer)
                if layer == layers -1:
                    h_emo.append(h_task[0])
                    h_cau.append(h_task[1])
                    h_share.append(h_task[2])
                
            x = torch.stack(output, dim=0)
            

        h_emo = torch.stack(h_emo, dim=0)
        h_cau = torch.stack(h_cau, dim=0)
        h_share = torch.stack(h_share, dim=0)

        return h_emo.transpose(0, 1), h_cau.transpose(0, 1), h_share.transpose(0, 1) #[batch, seq_len, hidden]



class re_unit(nn.Module):
    def __init__(self, k=5, drop=0.1):
        super(re_unit, self).__init__()
        self.hidden_size = 300
        self.relation_size = 1

        self.hid2hid = nn.Linear(self.hidden_size * 2 + 50, self.hidden_size)
        self.hid2rel = nn.Linear(self.hidden_size, 1)
        self.elu = nn.ELU()

        self.r = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(drop)
        
        self.K = k
        self.pos_layer = nn.Embedding(2* self.K + 1, 50)
        nn.init.xavier_uniform_(self.pos_layer.weight)

        torch.nn.init.orthogonal_(self.hid2hid.weight, gain=1)
        torch.nn.init.orthogonal_(self.hid2rel.weight, gain=1)
        torch.nn.init.orthogonal_(self.r.weight, gain=1)
        


    def forward(self, h_e, h_c, h_share, mask):
        batch_size, length, hidden = h_c.size()

       
        h_e = h_e + h_share
        h_c = h_c + h_share
        couples, rel_pos, emo_cau_pos = self.couple_generator(h_e, h_c, self.K)
          
        rel_pos = rel_pos + self.K
        rel_pos_emb = self.pos_layer(rel_pos)
        kernel = self.kernel_generator(rel_pos)
        kernel = kernel.unsqueeze(0).expand(batch_size, -1, -1)
        rel_pos_emb = torch.matmul(kernel, rel_pos_emb)

        pair = torch.cat((couples, rel_pos_emb), dim=-1)
        
        pair = self.ln(self.hid2hid(pair))
        pair = self.elu(self.dropout(pair))
        pair = self.hid2rel(pair)
 
        pair = pair.squeeze(-1)
        
        return pair , emo_cau_pos#[batch,seq,seq]
    
    def couple_generator(self, H_e, H_c,  k):
        batch, seq_len, feat_dim = H_e.size()
        P_left = torch.cat([H_e] * seq_len, dim=2)
        P_left = P_left.reshape(-1, seq_len * seq_len, feat_dim)
        P_right = torch.cat([H_c] * seq_len, dim=1)
        P = torch.cat([P_left, P_right], dim=2)
        
        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)
        

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
                      
            rel_mask = rel_mask.unsqueeze(1).expand(-1, 2 * feat_dim)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
                   
            P = P.masked_select(rel_mask)
            P = P.reshape(batch, -1, 2 * feat_dim)
        
        assert rel_pos.size(0) == P.size(1)
        rel_pos = rel_pos.unsqueeze(0).expand(batch, -1)

        emo_cau_pos = []
        for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
            emo_cau_pos.append([emo, cau])
        return P, rel_pos, emo_cau_pos
    
    def kernel_generator(self, rel_pos):
        n_couple = rel_pos.size(1)
        rel_pos_ = rel_pos[0].type(torch.FloatTensor).to(DEVICE)
        kernel_left = torch.cat([rel_pos_.reshape(-1, 1)] * n_couple, dim=1)
        kernel = kernel_left - kernel_left.transpose(0, 1)
        return torch.exp(-(torch.pow(kernel, 2)))



class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.line_e = nn.Linear(300 * 2, 300)
        self.out_e_subwork = nn.Linear(300, 1)
        self.line_c = nn.Linear(300 * 2, 300)
        self.out_c_subwork = nn.Linear(300, 1)
        

        
        self.leakyrule = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        torch.nn.init.orthogonal_(self.line_e.weight, gain=1)
        torch.nn.init.orthogonal_(self.out_e_subwork.weight, gain=1)
        torch.nn.init.orthogonal_( self.line_c.weight, gain=1)
        torch.nn.init.orthogonal_(self.out_c_subwork.weight, gain=1)


    def forward(self, sent_e, sent_c, share):
        sent_e_subwork = self.leakyrule(self.line_e(torch.cat((sent_e, share), -1))) 
        pred_e_subwork = self.out_e_subwork(sent_e_subwork)
        sent_c_subwork = self.leakyrule(self.line_c(torch.cat((sent_c, share), -1))) 
        pred_c_subwork = self.out_c_subwork(sent_c_subwork)
        
        pred_e_subwork = pred_e_subwork.squeeze(2)
        pred_c_subwork = pred_c_subwork.squeeze(2)
                
        return pred_e_subwork, pred_c_subwork, sent_e_subwork, sent_c_subwork

class A_2_Net(nn.Module):
    def __init__(self, configs, input_size=768, k=5, drop=0.1):
        super(A_2_Net, self).__init__()
        
        self.feature_extractor = encoder(input_size, drop)
        self.k = k
        self.drop = drop
        
        self.re = re_unit(self.k, self.drop)
        self.pred = Pre_Predictions(configs)
        self.dropout = nn.Dropout(self.drop)
     
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.KL_loss = KL_div()

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, y_mask_b, mask_table, doc_couples, y_mask):
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))
        doc_sents_h_ = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
           
        doc_sents_h_ = doc_sents_h_.transpose(0, 1)

        if self.training:
            doc_sents_h_ = self.dropout(doc_sents_h_)

        h_emo, h_cau, h_share = self.feature_extractor(doc_sents_h_, layers=1)
        
        
        pred_e, pred_c, sent_e_subwork, sent_c_subwork = self.pred(h_emo, h_cau, h_share)
        pair_score, emo_cau_pos = self.re(h_emo, h_cau, h_share, y_mask_b)
        
        
        kl_loss = self.KL_loss(pred_e, pred_c, sent_e_subwork, sent_c_subwork, mask_table,  pair_score, emo_cau_pos, doc_couples, y_mask, self.k)
        return pair_score, emo_cau_pos, pred_e, pred_c, kl_loss
    
    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h
    
    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        couples_true, couples_mask, doc_couples_pred = self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask, test)

        couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        couples_true = couples_true.masked_select(couples_mask)
        couples_pred = couples_pred.masked_select(couples_mask)
        loss_couple = criterion(couples_pred, couples_true)
       

        return loss_couple, doc_couples_pred

    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        pred_e_id = torch.as_tensor(pred_e)
        pred_c_id = torch.as_tensor(pred_c)
        y_mask_id = ~torch.as_tensor(y_mask)
        
        pred_e_id.masked_fill_(y_mask_id, -1e15)
       
        pred_c_id.masked_fill_(y_mask_id, -1e15)
        
        pred_e = pred_e.masked_select(y_mask)
        true_e = y_emotions.masked_select(y_mask)
        loss_e = criterion(pred_e, true_e)

        pred_c = pred_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)
        loss_c = criterion(pred_c, true_c)
        
        pred_e_list, pred_c_list = [], []
        for batch_e, batch_c in zip(pred_e_id, pred_c_id):
            
            batch_e = (logistic(batch_e.detach().cpu().numpy()) > 0.5).astype(int)
            batch_c = (logistic(batch_c.detach().cpu().numpy()) > 0.5).astype(int)
            
            
            batch_e_id = np.argwhere(batch_e == 1).flatten() + 1
            batch_c_id = np.argwhere(batch_c == 1).flatten() + 1
            pred_e_list.append(batch_e_id)
            pred_c_list.append(batch_c_id)          
        
        return loss_e, loss_c, pred_e_list, pred_c_list 
    
    def loss_score(self, pred_score, score_label, y_mask):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        

        criterion = nn.MSELoss()
        pred_score = pred_score.masked_select(y_mask)
        score_label = score_label.masked_select(y_mask)
        loss_score = criterion(pred_score, score_label)

        return loss_score
    
    
class KL_div(nn.Module):
    def __init__(self, hidden_size=300):
        super(KL_div, self).__init__()   
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.KLDivLoss()
        
        self.criterion_BCE = nn.BCELoss(reduction='mean')
        
    
    
    def forward(self, pred_e, pred_c, subwork_sent_e, subwork_sent_c, mask_table,  couples_pred, emo_cau_pos, doc_couples, y_mask, k=5):
        
        batch, seq_len = pred_e.size()
        _, _, hidden_size = subwork_sent_c.size()
        pred_e = self.sigmoid(pred_e.unsqueeze(2)) #[batch, seq, 1]
        pred_c = self.sigmoid(pred_c.unsqueeze(1)) #[batch, 1, seq]
        subwork_table = torch.bmm(pred_e, pred_c) #batch, seq, seq
        subwork_table = torch.pow(subwork_table, 1/2)
        

        constraint_table = torch.bmm(subwork_sent_e, subwork_sent_c.transpose(-2, -1)) / math.sqrt(hidden_size) #[batch, seq, seq]
        
        mask_table = 1 - mask_table
        mask_table = torch.BoolTensor(mask_table).to(DEVICE)
        constraint_table.masked_fill_(mask_table, -1e9)
        constraint_table = F.softmax(constraint_table, dim = -1)
        
        final_constraint_table = torch.mul(subwork_table, constraint_table) #[batch, seq, seq]
        final_constraint_table = final_constraint_table.reshape(batch, seq_len * seq_len, 1)
        
        
        
        base_idx = np.arange(1, seq_len + 1)
        emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
        cau_pos = np.concatenate([base_idx] * seq_len, axis=0)

        rel_pos = cau_pos - emo_pos
        rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
        emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
        cau_pos = torch.LongTensor(cau_pos).to(DEVICE)
        

        if seq_len > k + 1:
            rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
            rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
            rel_pos = rel_pos.masked_select(rel_mask)
            emo_pos = emo_pos.masked_select(rel_mask)
            cau_pos = cau_pos.masked_select(rel_mask)
                      
            rel_mask = rel_mask.unsqueeze(1)
            rel_mask = rel_mask.unsqueeze(0).expand(batch, -1, -1)
                   
            final_constraint_table = final_constraint_table.masked_select(rel_mask)
            final_constraint_table = final_constraint_table.reshape(batch, -1, 1)
        
        final_constraint_table = final_constraint_table.squeeze(-1)
        
        couples_true, couples_mask, _ = self.output_util(couples_pred, emo_cau_pos, doc_couples, y_mask)
        couples_mask = torch.BoolTensor(couples_mask).to(DEVICE)
        
        final_constraint_table = final_constraint_table.masked_select(couples_mask)
        
        

        couples_pred = self.sigmoid(couples_pred)
        couples_pred = couples_pred.masked_select(couples_mask)
        
        final_constraint_table_p = F.log_softmax(final_constraint_table, -1)
        couples_pred_q = F.softmax(couples_pred, -1)
        loss_score_1 = self.criterion(final_constraint_table_p, couples_pred_q)
        
        
        couples_pred_p = F.log_softmax(couples_pred, -1)
        final_constraint_table_q = F.softmax(final_constraint_table, -1)
        loss_score_2 = self.criterion(couples_pred_p, final_constraint_table_q)

        return  1/2 * (loss_score_2 + loss_score_1)
        
    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred
