import torch
import torch.nn as nn
import torch.nn.functional as F

layer_dict = {2:0,6:1,15:2}     # 


sparse_token_list_192 = [300,200,110]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303,110,36]
sparse_token_list_64 = [66,30,17]  
   
sparse_token_list_72 = [68,30,18] 
sparse_token_list_180 = [268,170,100] 
sparse_token_list_547 = [700,570,471]
sparse_token_list_365 = [550,450,220]       # 2*576  4*300 10*200  16*110
sparse_token_list_147 = [182,144,130]
sparse_token_list_98 = [152,104,60] 
sparse_token_list_19 = [16,9,4]  
sparse_token_list_49 = [90,40,20]   

sparse_token_dict = {
    547: sparse_token_list_547,
    365: sparse_token_list_365,
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64 : sparse_token_list_64,
    49 : sparse_token_list_49,
    19 : sparse_token_list_19,
    72 : sparse_token_list_72,
    180: sparse_token_list_180,
    98: sparse_token_list_98,
    147: sparse_token_list_147,
}

def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx,retained_tokens):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    self_attn_weights = self_attn_weights.mean(1) # B, L[Q], L[K]

    t_token_idx = t_token_idx[1] + text_token_start
    # print('self_attn_weights:', self_attn_weights.shape)
    # print('t_token_idx:',t_token_idx.shape)
    # print('v_token_start:',v_token_start.shape)
    # relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num] # B, L2, L1
    self_attn_weights = self_attn_weights[:, t_token_idx , :]
    relation_vis_text = self_attn_weights[:, : , v_token_start]

    relation_vis_text = relation_vis_text.mean(1) # B, L1

    relation_vis = relation_vis_text
    # print('relation_vis:', relation_vis.shape)
    s_flag = True       # s_flag controls whether token merge is needed.

    sparse_token_list = sparse_token_dict[retained_tokens]

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        # print('k:',  min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1))
        _, indices = torch.topk(relation_vis, min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text

if __name__ == "__main__":

    self_attn_weights, v_token_start, v_token_num, text_token_start = torch.rand(4, 16, 1084, 1084), 36, 576, 700
    mask = attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start)
    print(mask.shape)