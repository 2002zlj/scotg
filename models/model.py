from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from continual_clip.utils import get_class_ids_per_task, get_class_names, batch, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation
import copy

from continual_clip.cc import conceptual_captions

from continual_clip import utils
import os
import random

from continual_clip.dynamic_dataset import DynamicDataset
from dataclasses import dataclass, field
import transformers
from typing import Dict, Optional, Sequence, List
from transformers import CLIPModel,CLIPProcessor,CLIPTokenizer
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
import json
import logging


def MMEA_collate_fn(batches):
    output = dict()
    output['inputs'] = []
    targets=[]
    task_ids=[]
    for batch in batches:
        input,target,task_id = batch
        output['inputs'].append(input)
        targets.append(torch.tensor(target).unsqueeze(0))
        task_ids.append(torch.tensor(task_id).unsqueeze(0))
    output['targets'] = torch.cat(targets,dim=0)
    output['task_ids'] = torch.cat(task_ids,dim=0)
    return output

class MMEA_CLIP(nn.Module):
    def __init__(self, CLIP_model):
        self.model = CLIP_model
    def forward(self, images, text):
        logits_per_images =[]
        for image in images:
            logits_per_image, _ = self.model(image, text)
            logits_per_images.append(logits_per_image)
        logits_per_image =torch.cat( [torch.mean(x,dim=0).unsqueeze(0) for x in logits_per_images],dim=0)
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text



def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


class CLIPTextModelForPromptTuning(nn.Module):
    def __init__(self, model: object, deep_g: int, deep_replace_method: str = "replace"):
        '''
        CLIP Text Encoder for PE
        model: CLIP Text Encoder
        deep_g: number of layers to append prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        '''
        super().__init__()
        self.model = model
        self.d_model = 512
        self.deep_g = deep_g
        self.deep_replace_method = deep_replace_method

    def forward(self, 
            text_tokens: torch.Tensor, 
            attn_mask: torch.Tensor, 
            g_prompt: torch.Tensor
        ):
        '''
        text_tokens: [batch_size, n_tokens]
        attn_mask: [batch_size, n_tokens]
        g_prompt: [text_tokens, n_deep, n_prompts, d_model]
        '''
        # test = self.model(text_tokens,attn_mask)
        bs = g_prompt.size(0)

        g_prompt = g_prompt.to(text_tokens.device)
        
        g = g_prompt[:, 0]
        L_g = g.size(1)


        x = self.model.embeddings.token_embedding(text_tokens)

        x = torch.cat([x[:,0:1,:], g, x[:,1:,:]], dim=1)
        x = x + self.model.embeddings.position_embedding(torch.arange(x.size(1), device=attn_mask.device).unsqueeze(0))

        for i,l in enumerate(self.model.encoder.layers):
            
            if i > 0:
                if i < self.deep_g:
                    if self.deep_replace_method == "replace":
                        g = g_prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:,1:(L_g+1),:]
                        g = torch.cat([previous_g_out, g_prompt[:, i]], dim=1)
                    elif self.deep_replace_method == "accumulate_same":
                        g = torch.cat([g, g_prompt[:, i]], dim=1)
                    x = torch.cat([x[:,0:1,:], g, x[:,(L_g+1):,:]], dim=1)
                    L_g = g.size(1)
                    
            res = x
            x = l.layer_norm1(x)

            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _create_4d_causal_attention_mask(input_shape=[x.size(0), x.size(1)], dtype=x.dtype,device=x.device)

            attn_mask_ = torch.cat([torch.ones(bs, L_g, device=attn_mask.device), attn_mask], dim=-1)
            # expand attention_mask
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attn_mask_, x.dtype)

            #  CLIPSdpaAttention
            if attention_mask is not None and causal_attention_mask is not None:
                attn_mask_ = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attn_mask_ = causal_attention_mask
            else:
                attn_mask_ = attention_mask            
            
            bsz, tgt_len, embed_dim =  x.size()

            # q = l.self_attn.q_proj(x) * 0.125   # 源代码
            q = l.self_attn.q_proj(x) 
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            q = q.view(bsz, -1, l.self_attn.num_heads, l.self_attn.head_dim).transpose(1, 2)
            k = k.view(bsz, -1, l.self_attn.num_heads, l.self_attn.head_dim).transpose(1, 2)
            v = v.view(bsz, -1, l.self_attn.num_heads, l.self_attn.head_dim).transpose(1, 2)

            assert q.size(2)==x.size(1)
            # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask_,
                dropout_p=l.self_attn.dropout if l.self_attn.training else 0.0,
                scale=l.self_attn.scale,
            )
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
            x = l.self_attn.out_proj(attn_output)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)
            
            x = res + x

        x = self.model.final_layer_norm(x)

        index = text_tokens.argmax(dim=-1) + L_g
        return x[torch.arange(x.size(0)), index]
    
###############################################
### CLIP Vision Encoder Parameter-Efficient ###
###############################################
    
class CLIPVisionModelForPromptTuning(nn.Module):
    def __init__(self, 
            model: object, 
            deep_g: int, 
            deep_replace_method: str = "replace",
            visual:int=0,
            first:int =None,
            second:int =None,
            third:int =None,            
        ):
        '''
        CLIP Vision Encoder for PE
        model: CLIP Vision Encoder
        deep_g: number of layers to append prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        '''
        super().__init__()
        self.model = model
        self.d_model = 768
        self.deep_g = deep_g
        self.deep_replace_method = deep_replace_method
        self.visual = visual
        self.first = first
        self.second = second
        self.third = third

    def forward(self, 
            image: torch.Tensor, 
            g_prompt: torch.Tensor
        ):
        '''
        image: [batch_size, 3, 224, 224]
        g_prompt: [batch_size, n_deep, n_prompts, d_model]
        '''
        # test = self.model(image)
        if self.visual==0:
            x = image
        else:
            x = self.model.embeddings(image)

        g = g_prompt[:, 0]

        x = torch.cat([x, g], dim=1)
        x = self.model.pre_layrnorm(x)
        L_g = g.size(1)
        
        for i,l in enumerate(self.model.encoder.layers):

            if i > 0:
                if i < self.deep_g:
                    if self.deep_replace_method == "replace":
                        g = g_prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:,-L_g:,:]
                        g = torch.cat([previous_g_out, g_prompt[:, i]], dim=1)
                    elif self.deep_replace_method == "accumulate_same":
                        g = torch.cat([g, g_prompt[:, i]], dim=1)
                    x = torch.cat([x[:, :-L_g, :], g], dim=1)
                    L_g = g.size(1)

            res = x
            x = l.layer_norm1(x)

            bsz, tgt_len, embed_dim =  x.size()

            q = l.self_attn.q_proj(x) 
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            q = q.view(bsz, -1, l.self_attn.num_heads, l.self_attn.head_dim).transpose(1, 2)
            k = k.view(bsz, -1, l.self_attn.num_heads, l.self_attn.head_dim).transpose(1, 2)
            v = v.view(bsz, -1, l.self_attn.num_heads, l.self_attn.head_dim).transpose(1, 2)

            assert q.size(2)==x.size(1)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=l.self_attn.dropout if l.self_attn.training else 0.0,
                scale=l.self_attn.scale,
            )
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
            x = l.self_attn.out_proj(attn_output)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)
            
            x = res + x
            if i==self.first:
                first = x
            if i==self.second:
                second = x
            if i==self.third:
                third = x

        return [self.model.post_layernorm(x[:,0,:]),self.model.post_layernorm(first[:,0,:]),self.model.post_layernorm(second[:,0,:]),self.model.post_layernorm(third[:,0,:])]
    
################################
### CLIP Parameter-Efficient ###
################################

class SCOTG_classroom(nn.Module):
    def __init__(self, 
            L_g: int = 2, 
            deep_g: int = 3, 
            text_deep_replace_method: str = "replace",
            vision_deep_replace_method: str = "replace",
            model_path: str = None,
            visual:int =None,
            first:int =None,
            second:int =None,
            third:int =None,
            a1:float =None,
            a2:float =None,
            a3:float =None,
        ):
        '''
        CLIP Parameter-Efficient
        L_g: number of g-prompts
        deep_g: number of layers to attach g-prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        '''
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_path)

        ### Text Encoder ###
        self.text_model = CLIPTextModelForPromptTuning(
            model = self.clip_model.text_model, 
            deep_g = deep_g, 
            deep_replace_method = text_deep_replace_method
        )

        ### Vision Encoder ###
        self.vision_model = CLIPVisionModelForPromptTuning(
            model = self.clip_model.vision_model, 
            deep_g = deep_g, 
            deep_replace_method = vision_deep_replace_method,
            visual = visual,
            first = first,
            second = second,
            third = third
        )

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

        self.image_proj = self.clip_model.visual_projection
        self.text_proj = self.clip_model.text_projection

        for p in self.parameters():
            p.requires_grad = False
        
        self.prompt_proj = nn.Linear(self.text_model.d_model, self.vision_model.d_model)
        self.g_values = nn.Parameter(torch.zeros(deep_g, L_g, self.text_model.d_model))
        
        nn.init.xavier_uniform_(self.g_values.data)

        self.L_g = L_g
        self.deep_g = deep_g
        
    def forward(
            self, 
            image: torch.Tensor, 
            text_tokens: torch.Tensor, 
            attn_mask: torch.Tensor,
            device = "cuda"
        ):
        '''
        image: [batch_size, 3, 224, 224]
        text_tokens: [n_classes, max_length]
        attn_mask: [n_classes, max_length]
        '''
        batch_size = image.shape[0]

        text_g_prompt = self.g_values.repeat(text_tokens[0].size(0), 1, 1, 1).to(device)
        vision_g_prompt = self.prompt_proj(self.g_values.repeat(batch_size, 1, 1, 1))

        L_g = text_g_prompt[:,0].size(1)
        max_length = 77
        
        text_tokens_temp = []
        attn_mask_temp = []

        for x in text_tokens:
            if x.shape[1]+L_g>max_length:
                text_tokens_temp.append(x[:,0:max_length-L_g])
            else:
                text_tokens_temp.append(x)

        for x in attn_mask:
            if x.shape[1]+L_g>max_length:
                attn_mask_temp.append(x[:,0:max_length-L_g])
            else:
                attn_mask_temp.append(x)

        text_tokens =text_tokens_temp
        attn_mask = attn_mask_temp

        text_out = [self.text_model(x, y, text_g_prompt) for  x,y in zip(text_tokens,attn_mask)]
        img_out = self.vision_model(image, vision_g_prompt) 
                 
        # Project to common dimensional space
        text_projs = [self.text_proj(x) for x in text_out]
        img_projs = [self.image_proj(x) for x in img_out]
        # Normalize
        text_embed = [x / _get_vector_norm(x) for x in text_projs]
        img_embed = [x / _get_vector_norm(x) for x in img_projs]

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embed[0], img_embed[0].t().to(text_embed[0].device)) * logit_scale.to(text_embed[0].device)+\
        self.a1*torch.matmul(text_embed[1], img_embed[1].t().to(text_embed[0].device)) * logit_scale.to(text_embed[0].device)+\
        self.a2*torch.matmul(text_embed[2], img_embed[2].t().to(text_embed[0].device)) * logit_scale.to(text_embed[0].device)+\
        self.a3*torch.matmul(text_embed[3], img_embed[3].t().to(text_embed[0].device)) * logit_scale.to(text_embed[0].device)
     
        logits_per_image = logits_per_text.t()        

        return logits_per_image,logits_per_text


        
    def encode_image(
            self, 
            image: torch.Tensor, 
            device = "cuda"
        ):
        '''
        image: [batch_size, 3, 224, 224]
        text_tokens: [n_classes, max_length]
        attn_mask: [n_classes, max_length]
        '''
        batch_size = image.shape[0]

        vision_g_prompt = self.prompt_proj(self.g_values.repeat(batch_size, 1, 1, 1))

        img_out = self.vision_model(image, vision_g_prompt) 
                 
        # Project to common dimensional space
        img_projs = [self.image_proj(x) for x in img_out]
        # Normalize
        img_embed = [x / _get_vector_norm(x) for x in img_projs]

        return img_embed




class FSCIL_classrroom_SCOTG(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model = SCOTG_classroom(L_g=cfg.L_g, deep_g=cfg.deep_g, 
                                                      text_deep_replace_method=cfg.text_deep_replace_method,
                                                      vision_deep_replace_method=cfg.vision_deep_replace_method,
                                                      model_path=cfg.model_name,
                                                      visual=cfg.visual,
                                                      first = cfg.first,
                                                      second =cfg.second,
                                                      third = cfg.third,
                                                      a1 = cfg.a1,
                                                      a2 = cfg.a2,
                                                      a3 = cfg.a3,)
        self.text_preprocess = CLIPProcessor.from_pretrained(cfg.model_name).tokenizer
        self.regularization_method = cfg.regularization_method
        self.lr = cfg.lr
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.num_classes_per_exp = []
        self.text_tokens = None
        with open(cfg.workdir+'/re_classroom_4_turbo.json','r') as f:
            self.relations = json.load(f)

    
    def forward(self, image):
        with torch.no_grad():
            logits_per_image,logits_per_text = self.model(image,self.text_tokens,self.attn_mask)
        return logits_per_image

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_classes_per_exp.append(len(get_class_names(self.classes_names, self.class_ids_per_task[task_id])))
        out_text_tokens0 = self.text_preprocess([self.prompt_template.format(c) for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens1 = self.text_preprocess([self.relations[c][0] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens2 = self.text_preprocess([self.relations[c][1] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens3 = self.text_preprocess([self.relations[c][2] for c in self.current_class_names], padding=True, return_tensors="pt")
        self.text_tokens0 = out_text_tokens0["input_ids"].to(self.device)
        self.attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)        
        self.text_tokens1 = out_text_tokens1["input_ids"].to(self.device)
        self.attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)
        self.text_tokens2 = out_text_tokens2["input_ids"].to(self.device)
        self.attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)
        self.text_tokens3 = out_text_tokens3["input_ids"].to(self.device)
        self.attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)        
        self.text_tokens = [self.text_tokens0,self.text_tokens1,self.text_tokens2,self.text_tokens3]
        self.attn_mask = [self.attn_mask0,self.attn_mask1,self.attn_mask2,self.attn_mask3]

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)


    def test(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_classes_per_exp.append(len(get_class_names(self.classes_names, self.class_ids_per_task[task_id])))
        out_text_tokens0 = self.text_preprocess([self.prompt_template.format(c) for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens1 = self.text_preprocess([self.relations[c][0] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens2 = self.text_preprocess([self.relations[c][1] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens3 = self.text_preprocess([self.relations[c][2] for c in self.current_class_names], padding=True, return_tensors="pt")
        self.text_tokens0 = out_text_tokens0["input_ids"].to(self.device)
        self.attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)        
        self.text_tokens1 = out_text_tokens1["input_ids"].to(self.device)
        self.attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)
        self.text_tokens2 = out_text_tokens2["input_ids"].to(self.device)
        self.attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)
        self.text_tokens3 = out_text_tokens3["input_ids"].to(self.device)
        self.attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)        
        self.text_tokens = [self.text_tokens0,self.text_tokens1,self.text_tokens2,self.text_tokens3]
        self.attn_mask = [self.attn_mask0,self.attn_mask1,self.attn_mask2,self.attn_mask3]




    def train(self, task_id, cfg, train_dataset, train_classes_names):
        ### laoding dataset
        if task_id > 0:
            train_loader = DataLoader(train_dataset[task_id], 
                                        batch_size=5, 
                                        shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset[task_id], 
                                        batch_size=cfg.batch_size, 
                                        shuffle=True, num_workers=0)            
        train_iter = iter(train_loader)

        ### hardcoding 
        EPOCH = cfg.n_runs
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        params = [
            v for k, v in self.model.named_parameters() if v.requires_grad
        ]

        # optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=cfg.lr,
                                          momentum=0.9,
                                          weight_decay=1e-5)

        scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=200,
                cycle_mult=1.0,
                max_lr=0.1, min_lr=0.001,
                warmup_steps=50, gamma=1.0
            )
        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print("Using devices", devices)


        if cfg.l2 > 0:
            print("L2 norm")
            l2_model = copy.deepcopy(self.model)
            l2_model.cuda()
            
        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        texts0 = [self.prompt_template.format(c) for c in classnames]
        texts1 = [self.relations[c][0] for c in classnames]
        texts2 = [self.relations[c][1] for c in classnames]
        texts3 = [self.relations[c][2] for c in classnames]
 
        out_text_tokens0 = self.text_preprocess(texts0, padding=True, return_tensors="pt")
        texts0 = out_text_tokens0["input_ids"].to(self.device)
        attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)    

        out_text_tokens1 = self.text_preprocess(texts1, padding=True, return_tensors="pt")
        texts1 = out_text_tokens1["input_ids"].to(self.device)
        attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)       

        out_text_tokens2 = self.text_preprocess(texts2, padding=True, return_tensors="pt")
        texts2 = out_text_tokens2["input_ids"].to(self.device)
        attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)   

        out_text_tokens3 = self.text_preprocess(texts3, padding=True, return_tensors="pt")
        texts3 = out_text_tokens3["input_ids"].to(self.device)
        attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)   

        texts = [texts0,texts1,texts2,texts3]
        attn_mask = [attn_mask0,attn_mask1,attn_mask2,attn_mask3]

        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            # scheduler(iteration)
            try:
                inputs,targets, task_ids, names = next(train_iter)    
            except:
                train_iter = iter(train_loader)
                inputs,targets, task_ids, names = next(train_iter)  
            
            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift =  100 + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "classroom" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift                
            else:
                shift = task_id * cfg.increment
                targets -= shift
            
            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image,logits_per_text =self.model(inputs,texts,attn_mask)
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)

            if cfg.l2 > 0:
                if task_id>0:
                    loss_l2 = l2_loss(self.model, l2_model)
                    loss += cfg.l2 * loss_l2

            optimizer.zero_grad()
            loss.backward()
            if task_id > 0:
                if self.regularization_method == 'balance':
                    reg_lambda = self.num_classes_per_exp[task_id] / sum(self.num_classes_per_exp[:task_id+1])
                    self.model.g_values.grad *= reg_lambda
                    self.model.prompt_proj.weight.grad *= reg_lambda
                    self.model.prompt_proj.bias.grad *= reg_lambda            
            optimizer.step()
            scheduler.step()
            # for i in range(len(names)):
            #     name = names[i]
            #     logging.info(f"Task {task_id} / Iter {iteration} / Sample {name} / Label {targets[i].item()} / Pred {logits_per_image[i].argmax().item()}")
            if iteration % len(train_loader) == 0:
                logging.info(f"Task {task_id} / Iter {iteration} / Loss {loss.item()}")

        self.model.eval()



class FSCIL_classrroom_SCOTG_otherLLM(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model = SCOTG_classroom(L_g=cfg.L_g, deep_g=cfg.deep_g, 
                                                      text_deep_replace_method=cfg.text_deep_replace_method,
                                                      vision_deep_replace_method=cfg.vision_deep_replace_method,
                                                      model_path=cfg.model_name,
                                                      visual=cfg.visual,
                                                      first = cfg.first,
                                                      second =cfg.second,
                                                      third = cfg.third,
                                                      a1 = cfg.a1,
                                                      a2 = cfg.a2,
                                                      a3 = cfg.a3,)
        self.text_preprocess = CLIPProcessor.from_pretrained(cfg.model_name).tokenizer
        self.regularization_method = cfg.regularization_method
        self.lr = cfg.lr
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.num_classes_per_exp = []
        self.text_tokens = None
        if cfg.method == 'GPT_4':
            with open(cfg.workdir+'/re_classroom_4.json','r') as f:
                self.relations = json.load(f)
        elif cfg.method == 'GPT_3.5_turbo':
            with open(cfg.workdir+'/re_classroom_3.5_turbo.json','r') as f:
                self.relations = json.load(f)
    
    def forward(self, image):
        with torch.no_grad():
            logits_per_image,logits_per_text = self.model(image,self.text_tokens,self.attn_mask)
        return logits_per_image

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_classes_per_exp.append(len(get_class_names(self.classes_names, self.class_ids_per_task[task_id])))
        out_text_tokens0 = self.text_preprocess([self.prompt_template.format(c) for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens1 = self.text_preprocess([self.relations[c][0] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens2 = self.text_preprocess([self.relations[c][1] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens3 = self.text_preprocess([self.relations[c][2] for c in self.current_class_names], padding=True, return_tensors="pt")
        self.text_tokens0 = out_text_tokens0["input_ids"].to(self.device)
        self.attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)        
        self.text_tokens1 = out_text_tokens1["input_ids"].to(self.device)
        self.attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)
        self.text_tokens2 = out_text_tokens2["input_ids"].to(self.device)
        self.attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)
        self.text_tokens3 = out_text_tokens3["input_ids"].to(self.device)
        self.attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)        
        self.text_tokens = [self.text_tokens0,self.text_tokens1,self.text_tokens2,self.text_tokens3]
        self.attn_mask = [self.attn_mask0,self.attn_mask1,self.attn_mask2,self.attn_mask3]

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)


    def test(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_classes_per_exp.append(len(get_class_names(self.classes_names, self.class_ids_per_task[task_id])))
        out_text_tokens0 = self.text_preprocess([self.prompt_template.format(c) for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens1 = self.text_preprocess([self.relations[c][0] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens2 = self.text_preprocess([self.relations[c][1] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens3 = self.text_preprocess([self.relations[c][2] for c in self.current_class_names], padding=True, return_tensors="pt")
        self.text_tokens0 = out_text_tokens0["input_ids"].to(self.device)
        self.attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)        
        self.text_tokens1 = out_text_tokens1["input_ids"].to(self.device)
        self.attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)
        self.text_tokens2 = out_text_tokens2["input_ids"].to(self.device)
        self.attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)
        self.text_tokens3 = out_text_tokens3["input_ids"].to(self.device)
        self.attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)        
        self.text_tokens = [self.text_tokens0,self.text_tokens1,self.text_tokens2,self.text_tokens3]
        self.attn_mask = [self.attn_mask0,self.attn_mask1,self.attn_mask2,self.attn_mask3]




    def train(self, task_id, cfg, train_dataset, train_classes_names):
        ### laoding dataset
        if task_id > 0:
            train_loader = DataLoader(train_dataset[task_id], 
                                        batch_size=5, 
                                        shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset[task_id], 
                                        batch_size=cfg.batch_size, 
                                        shuffle=True, num_workers=0)            
        train_iter = iter(train_loader)

        ### hardcoding 
        EPOCH = cfg.n_runs
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        params = [
            v for k, v in self.model.named_parameters() if v.requires_grad
        ]

        # optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=cfg.lr,
                                          momentum=0.9,
                                          weight_decay=1e-5)

        scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=200,
                cycle_mult=1.0,
                max_lr=0.1, min_lr=0.001,
                warmup_steps=50, gamma=1.0
            )
        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print("Using devices", devices)


        if cfg.l2 > 0:
            print("L2 norm")
            l2_model = copy.deepcopy(self.model)
            l2_model.cuda()
            
        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        texts0 = [self.prompt_template.format(c) for c in classnames]
        texts1 = [self.relations[c][0] for c in classnames]
        texts2 = [self.relations[c][1] for c in classnames]
        texts3 = [self.relations[c][2] for c in classnames]
 
        out_text_tokens0 = self.text_preprocess(texts0, padding=True, return_tensors="pt")
        texts0 = out_text_tokens0["input_ids"].to(self.device)
        attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)    

        out_text_tokens1 = self.text_preprocess(texts1, padding=True, return_tensors="pt")
        texts1 = out_text_tokens1["input_ids"].to(self.device)
        attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)       

        out_text_tokens2 = self.text_preprocess(texts2, padding=True, return_tensors="pt")
        texts2 = out_text_tokens2["input_ids"].to(self.device)
        attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)   

        out_text_tokens3 = self.text_preprocess(texts3, padding=True, return_tensors="pt")
        texts3 = out_text_tokens3["input_ids"].to(self.device)
        attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)   

        texts = [texts0,texts1,texts2,texts3]
        attn_mask = [attn_mask0,attn_mask1,attn_mask2,attn_mask3]

        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            # scheduler(iteration)
            try:
                inputs,targets, task_ids, names = next(train_iter)    
            except:
                train_iter = iter(train_loader)
                inputs,targets, task_ids, names = next(train_iter)  
            
            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift =  100 + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "classroom" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift                
            else:
                shift = task_id * cfg.increment
                targets -= shift
            
            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image,logits_per_text =self.model(inputs,texts,attn_mask)
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)

            if cfg.l2 > 0:
                if task_id>0:
                    loss_l2 = l2_loss(self.model, l2_model)
                    loss += cfg.l2 * loss_l2

            optimizer.zero_grad()
            loss.backward()
            if task_id > 0:
                if self.regularization_method == 'balance':
                    reg_lambda = self.num_classes_per_exp[task_id] / sum(self.num_classes_per_exp[:task_id+1])
                    self.model.g_values.grad *= reg_lambda
                    self.model.prompt_proj.weight.grad *= reg_lambda
                    self.model.prompt_proj.bias.grad *= reg_lambda            
            optimizer.step()
            scheduler.step()
            # for i in range(len(names)):
            #     name = names[i]
            #     logging.info(f"Task {task_id} / Iter {iteration} / Sample {name} / Label {targets[i].item()} / Pred {logits_per_image[i].argmax().item()}")
            if iteration % len(train_loader) == 0:
                logging.info(f"Task {task_id} / Iter {iteration} / Loss {loss.item()}")

        self.model.eval()



class FSCIL_classrroom_SCOTG_expand(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model = SCOTG_classroom(L_g=cfg.L_g, deep_g=cfg.deep_g, 
                                                      text_deep_replace_method=cfg.text_deep_replace_method,
                                                      vision_deep_replace_method=cfg.vision_deep_replace_method,
                                                      model_path=cfg.model_name,
                                                      visual=cfg.visual,
                                                      first = cfg.first,
                                                      second =cfg.second,
                                                      third = cfg.third,
                                                      a1 = cfg.a1,
                                                      a2 = cfg.a2,
                                                      a3 = cfg.a3,)
        self.text_preprocess = CLIPProcessor.from_pretrained(cfg.model_name).tokenizer
        self.regularization_method = cfg.regularization_method
        self.lr = cfg.lr
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.num_classes_per_exp = []
        self.text_tokens = None
        with open(cfg.workdir+'/describe_brief_classroom_4_turbo.json','r') as f:
            self.relations = json.load(f)

    
    def forward(self, image):
        with torch.no_grad():
            logits_per_image,logits_per_text = self.model(image,self.text_tokens,self.attn_mask)
        return logits_per_image

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_classes_per_exp.append(len(get_class_names(self.classes_names, self.class_ids_per_task[task_id])))
        out_text_tokens0 = self.text_preprocess([self.prompt_template.format(c) for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens1 = self.text_preprocess([self.relations[c][0] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens2 = self.text_preprocess([self.relations[c][1] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens3 = self.text_preprocess([self.relations[c][2] for c in self.current_class_names], padding=True, return_tensors="pt")
        self.text_tokens0 = out_text_tokens0["input_ids"].to(self.device)
        self.attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)        
        self.text_tokens1 = out_text_tokens1["input_ids"].to(self.device)
        self.attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)
        self.text_tokens2 = out_text_tokens2["input_ids"].to(self.device)
        self.attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)
        self.text_tokens3 = out_text_tokens3["input_ids"].to(self.device)
        self.attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)        
        self.text_tokens = [self.text_tokens0,self.text_tokens1,self.text_tokens2,self.text_tokens3]
        self.attn_mask = [self.attn_mask0,self.attn_mask1,self.attn_mask2,self.attn_mask3]

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)


    def test(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.num_classes_per_exp.append(len(get_class_names(self.classes_names, self.class_ids_per_task[task_id])))
        out_text_tokens0 = self.text_preprocess([self.prompt_template.format(c) for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens1 = self.text_preprocess([self.relations[c][0] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens2 = self.text_preprocess([self.relations[c][1] for c in self.current_class_names], padding=True, return_tensors="pt")
        out_text_tokens3 = self.text_preprocess([self.relations[c][2] for c in self.current_class_names], padding=True, return_tensors="pt")
        self.text_tokens0 = out_text_tokens0["input_ids"].to(self.device)
        self.attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)        
        self.text_tokens1 = out_text_tokens1["input_ids"].to(self.device)
        self.attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)
        self.text_tokens2 = out_text_tokens2["input_ids"].to(self.device)
        self.attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)
        self.text_tokens3 = out_text_tokens3["input_ids"].to(self.device)
        self.attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)        
        self.text_tokens = [self.text_tokens0,self.text_tokens1,self.text_tokens2,self.text_tokens3]
        self.attn_mask = [self.attn_mask0,self.attn_mask1,self.attn_mask2,self.attn_mask3]




    def train(self, task_id, cfg, train_dataset, train_classes_names):
        ### laoding dataset
        if task_id > 0:
            train_loader = DataLoader(train_dataset[task_id], 
                                        batch_size=5, 
                                        shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset[task_id], 
                                        batch_size=cfg.batch_size, 
                                        shuffle=True, num_workers=0)            
        train_iter = iter(train_loader)

        ### hardcoding 
        EPOCH = cfg.n_runs
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        params = [
            v for k, v in self.model.named_parameters() if v.requires_grad
        ]

        # optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=cfg.lr,
                                          momentum=0.9,
                                          weight_decay=1e-5)

        scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=200,
                cycle_mult=1.0,
                max_lr=0.1, min_lr=0.001,
                warmup_steps=50, gamma=1.0
            )
        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print("Using devices", devices)


        if cfg.l2 > 0:
            print("L2 norm")
            l2_model = copy.deepcopy(self.model)
            l2_model.cuda()
            
        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        texts0 = [self.prompt_template.format(c) for c in classnames]
        texts1 = [self.relations[c][0] for c in classnames]
        texts2 = [self.relations[c][1] for c in classnames]
        texts3 = [self.relations[c][2] for c in classnames]
 
        out_text_tokens0 = self.text_preprocess(texts0, padding=True, return_tensors="pt")
        texts0 = out_text_tokens0["input_ids"].to(self.device)
        attn_mask0 = out_text_tokens0["attention_mask"].to(self.device)    

        out_text_tokens1 = self.text_preprocess(texts1, padding=True, return_tensors="pt")
        texts1 = out_text_tokens1["input_ids"].to(self.device)
        attn_mask1 = out_text_tokens1["attention_mask"].to(self.device)       

        out_text_tokens2 = self.text_preprocess(texts2, padding=True, return_tensors="pt")
        texts2 = out_text_tokens2["input_ids"].to(self.device)
        attn_mask2 = out_text_tokens2["attention_mask"].to(self.device)   

        out_text_tokens3 = self.text_preprocess(texts3, padding=True, return_tensors="pt")
        texts3 = out_text_tokens3["input_ids"].to(self.device)
        attn_mask3 = out_text_tokens3["attention_mask"].to(self.device)   

        texts = [texts0,texts1,texts2,texts3]
        attn_mask = [attn_mask0,attn_mask1,attn_mask2,attn_mask3]

        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            # scheduler(iteration)
            try:
                inputs,targets, task_ids, names = next(train_iter)    
            except:
                train_iter = iter(train_loader)
                inputs,targets, task_ids, names = next(train_iter)  
            
            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift =  100 + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "classroom" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift                
            else:
                shift = task_id * cfg.increment
                targets -= shift
            
            inputs, targets = inputs.cuda(), targets.cuda()

            logits_per_image,logits_per_text =self.model(inputs,texts,attn_mask)
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)

            if cfg.l2 > 0:
                if task_id>0:
                    loss_l2 = l2_loss(self.model, l2_model)
                    loss += cfg.l2 * loss_l2

            optimizer.zero_grad()
            loss.backward()
            if task_id > 0:
                if self.regularization_method == 'balance':
                    reg_lambda = self.num_classes_per_exp[task_id] / sum(self.num_classes_per_exp[:task_id+1])
                    self.model.g_values.grad *= reg_lambda
                    self.model.prompt_proj.weight.grad *= reg_lambda
                    self.model.prompt_proj.bias.grad *= reg_lambda            
            optimizer.step()
            scheduler.step()
            # for i in range(len(names)):
            #     name = names[i]
            #     logging.info(f"Task {task_id} / Iter {iteration} / Sample {name} / Label {targets[i].item()} / Pred {logits_per_image[i].argmax().item()}")
            if iteration % len(train_loader) == 0:
                logging.info(f"Task {task_id} / Iter {iteration} / Loss {loss.item()}")

        self.model.eval()

