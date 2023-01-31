import sys, os, warnings, time

sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import random
import torch
from config import *
from data_loader import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from networks.pfn import *
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '-l', type=float, default='2e-5', help='learning rate')
parser.add_argument('--seed', '-s', type=int, default='129', help='random seed')
parser.add_argument('--batch', '-b', type=int, default='2', help='batch size')
parser.add_argument('--pos_len', '-p', type=int, default='5', help='relative position length')
parser.add_argument('--drop', '-d', type=float, default='0.1', help='dropout rate')
args = parser.parse_args()
TORCH_SEED = args.seed

def main(configs, fold_id):
    print('TORCH_SEED', TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    configs.batch_size = args.batch
    configs.lr = args.lr
    pos_k = args.pos_len
    drop_out = args.drop
    print('args.lr', args.lr, 'configs.lr', configs.lr)
    print('args.batch', args.batch, 'configs.batch_size', configs.batch_size)
    print('args.pos_len', args.pos_len)
    print('drop_out', args.drop)

    train_loader = build_train_data(configs, fold_id=fold_id)
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = A_2_Net(configs, k=pos_k, drop=drop_out).to(DEVICE)

    params = model.parameters()
    optimizer = AdamW(params, lr=configs.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
   
    model.zero_grad()
    metric_ec, metric_e, metric_c = (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)
    p_recall_e, p_recall_c = None, None
    early_stop_flag = None
    
    avg_cost = np.zeros([configs.epochs, 4], dtype=np.float32)
    
    
    for epoch in range(1, configs.epochs+1):
        total_ec_loss, total_e_loss, total_c_loss, total_score_loss, total_kl_loss = 0, 0, 0, 0, 0     

        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, mask = batch

            couples_pred, emo_cau_pos, pred_e, pred_c, loss_kl = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, mask, adj_b, doc_couples_b, y_mask_b)
            
            
            loss_e, loss_c, _ , _ = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
            loss_couple, _ = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b)           
            
            
            loss = loss_couple  +  0.4 *(loss_e + loss_c) +  0.4 *loss_kl 
            loss = loss / configs.gradient_accumulation_steps
            
            total_ec_loss += loss_couple.item()
            total_e_loss += loss_e.item()
            total_c_loss += loss_c.item()
            total_kl_loss += loss_kl.item()

            loss.backward()
          
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                scheduler.step()  
                
          
        total_ec_loss = total_ec_loss / len(train_loader)
        total_e_loss = total_e_loss / len(train_loader)
        total_c_loss = total_c_loss / len(train_loader)
        total_score_loss = total_score_loss / len(train_loader)
        total_kl_loss = total_kl_loss / len(train_loader)
        avg_cost[epoch - 1, :] = total_ec_loss, total_e_loss, total_c_loss, total_kl_loss #,total_score_loss
        print('epoch:', epoch, 'total_ec_loss:', total_ec_loss, 'total_e_loss:', total_e_loss, 'total_c_loss:', total_c_loss, 'total_kl_loss:', total_kl_loss)

        with torch.no_grad():
            model.eval()

            if configs.split == 'split10':
                test_ec, test_e, test_c, _, _, _, recall_e, recall_c = inference_one_epoch(configs, test_loader, model)
                print("epoch:", epoch, 'f_ec:', test_ec[2], 'f_e:', test_e[2], 'f_c:', test_c[2])
                print('recall_e', recall_e[1], 'recall_c', recall_c[1])
                if test_e[2] > metric_e[2]:
                    metric_e = test_e
                if test_c[2] > metric_c[2]:
                    metric_c = test_c
                if test_ec[2] > metric_ec[2]:
                    early_stop_flag = 1
                    metric_ec = test_ec
                    p_recall_e, p_recall_c = recall_e, recall_c
                else:
                    early_stop_flag += 1
                    
       
        if epoch > configs.epochs * 0.7 and early_stop_flag >= 5:
            break
    return metric_ec, metric_e, metric_c, p_recall_e, p_recall_c


def inference_one_batch(configs, batch, model):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, mask = batch

    couples_pred, emo_cau_pos, pred_e, pred_c, kl_loss = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, mask, adj_b,  doc_couples_b, y_mask_b)

    loss_e, loss_c, pred_e_id, pred_c_id = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
    loss_couple, doc_couples_pred_b = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, y_mask_b, test=True)
    

    return to_np(loss_couple), to_np(loss_e), to_np(loss_c), \
           doc_couples_b, doc_couples_pred_b, doc_id_b, pred_e_id, pred_c_id


def inference_one_epoch(configs, batches, model):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    emo_all, cau_all = [], []
    loss = 0
    for batch in batches:
        loss_ec, _, _, doc_couples, doc_couples_pred, doc_id_b, pred_e_id, pred_c_id = inference_one_batch(configs, batch, model)
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)
        emo_all.extend(pred_e_id)
        cau_all.extend(pred_c_id)
        loss += loss_ec
    print('loss_ec:', loss/len(batches))

    doc_couples_pred_all = lexicon_based_extraction(doc_id_all, doc_couples_pred_all)
    metric_ec, metric_e, metric_c = eval_func(doc_couples_all, doc_couples_pred_all)
    recall_e, recall_c = eval_consistency(emo_all, cau_all, doc_couples_pred_all)
    return metric_ec, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all, recall_e, recall_c


def lexicon_based_extraction(doc_ids, couples_pred):
    emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i) in enumerate(zip(doc_ids, couples_pred)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]

        emotional_clauses_i = emotional_clauses[doc_id]
        for couple in couples_pred_i[1:]:
            if couple[0][0] in emotional_clauses_i and logistic(couple[1]) > 0.5:
                couples_pred_i_filtered.append(couple[0])

        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered


if __name__ == '__main__':
    configs = Config()
    
    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 50
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    con_e, con_c = [], []
    for fold_id in range(1, 11):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c, p_recall_e, p_recall_c = main(configs, fold_id)
        print('F_ecp: {}'.format(float_n(metric_ec[2])))
        print('F_e: {}'.format(float_n(metric_e[2])))
        print('F_c: {}'.format(float_n(metric_c[2])))

        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)
        con_e.append(p_recall_e)
        con_c.append(p_recall_c)
        

    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()
    p_recall_e = np.mean(np.array(con_e), axis=0).tolist()
    p_recall_c = np.mean(np.array(con_c), axis=0).tolist()
    
    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
    print('-------------------- consistency----------------------')
    print('F_e: {}, P_e: {}, R_e: {}'.format(float_n(p_recall_e[2]), float_n(p_recall_e[0]), float_n(p_recall_e[1])))
    print('F_c: {}, P_c: {}, R_c: {}'.format(float_n(p_recall_c[2]), float_n(p_recall_c[0]), float_n(p_recall_c[1])))

