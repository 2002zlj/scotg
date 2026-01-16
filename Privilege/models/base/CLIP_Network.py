import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils.utils import identify_importance
import numpy as np
import copy
from models.vision_transformer import VisionTransformer
from transformers import CLIPVisionModel,CLIPModel
#todo PKT for domain specific knowledge learning..
#todo PKT with B-Prompt ==> Prefix Tuning 
#todo Need Something to focus on domain specific knowledge learning 
#todo finc inciteness from the Novel Category Discovery 
from .helper import *

class CLIP_ViT_MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.num_features = 768
        if self.args.dataset in ['mini_imagenet']:
            self.num_features = 768
        if self.args.dataset == 'cub200' or self.args.dataset == 'air':
            self.num_features = 768

        if args.scratch:
            self.CLIP_model = CLIPModel.from_pretrained('/data_25T/zlj/FSCIL/PriViLege-main/ckp/clip-vit-base-patch16')
        else:
            self.CLIP_model = CLIPModel.from_pretrained('/data_25T/zlj/FSCIL/PriViLege-main/ckp/clip-vit-base-patch16')        

        #* Prompt
        #todo Head 토큰 없애고 Vision으로 Pool
        self.prompt_length = 2 
        self.expert_length = 2 #* Number of tuning layers
        self.prompt = nn.Parameter(torch.randn(self.prompt_length,self.num_features))   #* VL
        self.expert_prompt = nn.Parameter(torch.randn(self.expert_length, 2, self.num_features))   #* B-Prompt (WC, MP)
        nn.init.uniform_(self.prompt, -1, 1)
        nn.init.uniform_(self.expert_prompt, -1, 1)
        #*------------------------------------------------------
        self.num_tokens = 197
        self.num_heads = self.CLIP_model.vision_model.encoder.layers[0].self_attn.num_heads
        
        self.comp_out = args.comp_out
        self.global_comp = nn.Conv1d(self.num_tokens + self.prompt_length, self.comp_out, kernel_size=1)
        nn.init.uniform_(self.global_comp.weight.data, -1, 1)
        
        self.local_comps = nn.ModuleList([nn.Conv1d(self.num_tokens+self.prompt_length, self.comp_out, kernel_size=1) for _ in range(self.num_heads)])
        for l_comp in self.local_comps:
            nn.init.uniform_(l_comp.weight.data, -1, 1)
        
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc.is_classifier = True
        
        self.seen_classes = args.base_class
    #todo =======================================================================================
    
    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes,device='cuda')
        self.mask[:self.seen_classes]=-torch.inf
        self.seen_classes += len(new_classes)
    
    def forward_metric(self, x, B_tuning=False, eval=False):
        #? Original
        x = self.prompt_encode(x, prompt_feat=True, B_tuning=B_tuning, eval=eval)
        cls_emb, prompt_emb = x 
        logit = self.fc(0.5*(cls_emb+prompt_emb['Vision']))
        
        return logit
    
    def encode(self, x):
        x = self.CLIP_model.vision_model(x).pooler_output
        return x
    
    def prompt_encode(self, img, prompt_feat=False, B_tuning=False, eval=False):
        x = self.CLIP_model.vision_model.embeddings(img)
        #*==============================================================
        #! VL-Prompt tuning
        prompting_tkn = self.prompt
        pos_tkn = prompting_tkn + (self.CLIP_model.vision_model.embeddings.position_embedding(self.CLIP_model.vision_model.embeddings.position_ids))[:,0].expand(self.prompt_length, -1)
        pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)#
        x= self.CLIP_model.vision_model.pre_layrnorm(x)
        #!=============================================================
        #* prefix for B-Prompt (Original)
        if B_tuning:
            #  Encoder
            x = self._forward_blocks(x, self.expert_prompt, eval=eval)
        else:
            x = self.CLIP_model.vision_model.encoder(x)
        
        cls_embed = x[:,0,:]
        if prompt_feat:
            prompt_embed ={}
            #todo Align -> Head
            prompt_embed['Vision'] = x[:,1,:]
            prompt_embed['Language_project'] =  self.CLIP_model.visual_projection(x[:,2,:])
            prompt_embed['Language_noproject'] =  x[:,2,:]
            
            return cls_embed, prompt_embed
        else:
            return cls_embed
    
    def _forward_blocks(self, x, prefix_tkn, eval=False):
        taskblock=[0,1]
        if len(taskblock) == len(self.CLIP_model.vision_model.encoder.layers) or  0 in taskblock:
            hidden_states = x
        for block_idx, block in enumerate(self.CLIP_model.vision_model.encoder.layers):
            if block_idx in taskblock:
                layer_outputs = self._pk_tuning(block, hidden_states, prefix_tkn[taskblock.index(block_idx)], eval=eval)
                hidden_states = layer_outputs[0]
            elif block_idx == 0:
                layer_outputs = block(x)
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = block(hidden_states=hidden_states,attention_mask=None,causal_attention_mask=None)
                hidden_states = layer_outputs[0]
        
        feat = self.CLIP_model.vision_model.post_layernorm(hidden_states)
        return feat

    def _extract_attn_mlp_feat(self, block, hidden_states):
        with torch.no_grad():
            residual = hidden_states
            hidden_states = block.layer_norm1(hidden_states)            
            attn = block.self_attn
            bsz, tgt_len, embed_dim = hidden_states.size()
            query_states = attn.q_proj(hidden_states) * attn.scale
            key_states = attn._shape(attn.k_proj(hidden_states), -1, hidden_states.shape[0])
            value_states = attn._shape(attn.v_proj(hidden_states), -1, hidden_states.shape[0])

            proj_shape = (bsz * attn.num_heads, -1, attn.head_dim)
            query_states = attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)

            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            if attn_weights.size() != (bsz * attn.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * attn.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=attn.dropout, training=attn.training)
            attn_output = torch.bmm(attn_probs, value_states)   # torch.Size([1536, 199, 64])

            if attn_output.size() != (bsz * attn.num_heads, tgt_len, attn.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, attn.num_heads, tgt_len, attn.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.view(bsz, attn.num_heads, tgt_len, attn.head_dim)   # torch.Size([128, 12, 199, 64])
            head_attentions = attn_output.transpose(1, 2)   # torch.Size([128, 199, 12, 64])
            attn_output = head_attentions.reshape(bsz, tgt_len, embed_dim)   # torch.Size([128, 199, 768])

            attn_output = attn.out_proj(attn_output)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = block.layer_norm2(hidden_states)

            mlp_feat = block.mlp(hidden_states)
            hidden_states = residual + mlp_feat
        return head_attentions, mlp_feat
        
    
    def _pk_tuning(self, block, hidden_states, prefix_tkn, eval=False):
        B,N,C = hidden_states.shape
        two,C = prefix_tkn.shape
        prefix_token = prefix_tkn.expand(B,two,C) #* B,2,768
        residual = hidden_states
        xq = block.layer_norm1(hidden_states)
        xk = xq.clone()
        xv = xq.clone()
        if self.args.prefix:
            xk = torch.cat([prefix_token[:,0].unsqueeze(1), xk],dim=1)
            xv = torch.cat([prefix_token[:,1].unsqueeze(1), xv],dim=1)
        else:
            head_attentions, mlp_feat = self._extract_attn_mlp_feat(block, hidden_states)
            global_feat = self.global_comp(mlp_feat).squeeze(1) #* (comp_out, dim) --> (Batch, 1, dim)
            #* Head_attentions: B, N, H, H_dim
            head_attentions = head_attentions.permute(2, 0, 1, 3)
            H, B, N, H_dim = head_attentions.shape
            
            head_feats = []
            for head_attn, local_comp in zip(head_attentions, self.local_comps):
                head_feats.append(local_comp(head_attn))    #* B,1,H_dim
            local_feat = torch.cat(head_feats, dim=1).reshape(B,-1)
            
            xk = torch.cat([(prefix_token[:,0] * local_feat).unsqueeze(1), xk],dim=1)
            xv = torch.cat([(prefix_token[:,1] * global_feat).unsqueeze(1), xv],dim=1)

        bsz, tgt_len, embed_dim = xq.size()
        attn = block.self_attn
        # get query proj
        query_states = attn.q_proj(xq) * attn.scale
        key_states = attn._shape(attn.k_proj(xk), -1, bsz)
        value_states = attn._shape(attn.v_proj(xv), -1, bsz)

        proj_shape = (bsz * attn.num_heads, -1, attn.head_dim)
        query_states = attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=attn.dropout, training=attn.training)
        attn_output = torch.bmm(attn_probs, value_states)   # torch.Size([1536, 199, 64])

        if attn_output.size() != (bsz * attn.num_heads, tgt_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, tgt_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, attn.num_heads, tgt_len, attn.head_dim)   # torch.Size([128, 12, 199, 64])
        head_attentions = attn_output.transpose(1, 2)   # torch.Size([128, 199, 12, 64])
        attn_output = head_attentions.reshape(bsz, tgt_len, embed_dim)   # torch.Size([128, 199, 768])

        attn_output = attn.out_proj(attn_output)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = block.layer_norm2(hidden_states)

        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        
        return outputs
    
    def forward(self, input, prompt_feat=False, B_tuning=False, base=False, query=False, eval=False):
        if base:
            embedding = self.prompt_encode(input, prompt_feat=True, B_tuning=True, eval=eval)
            cls_embed, prompt_embed = embedding
            logit = self.fc(0.5*(prompt_embed['Vision']+cls_embed))
            return logit, cls_embed, prompt_embed
        if query:
            q_feat = self.encode(input)
            return q_feat
        if self.mode == 'encoder':
            embedding = self.prompt_encode(input, prompt_feat=prompt_feat, B_tuning=B_tuning, eval=eval)
            if prompt_feat:
                cls_embed, prompt_embed = embedding
                return cls_embed, prompt_embed
            else:
                return embedding
        elif self.mode != 'encoder':
            input = self.forward_metric(input, B_tuning=B_tuning, eval=eval)
            return input
        
        else:
            raise ValueError('Unknown mode')

    def train_inc(self, dataloader, epochs, session, class_list, word_info, query_info):
        print("[Session: {}]".format(session))
        self.update_fc_avg(dataloader, class_list, query_info)
        
        for idx,batch in enumerate(dataloader):

            data_imgs, data_label = [_.cuda() for _ in batch]
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
            
            word_cur_embed = word_info['cur_embed'].cuda()
            word_embed = word_info['embed'].cuda()
            for epoch in range(epochs):
                self.train()
                cls_feat, prompt_feat = self.prompt_encode(data_imgs ,prompt_feat=True, B_tuning=True)
                
                logits = self.get_logits(0.5*(prompt_feat['Vision'] + cls_feat), self.fc)
                
                loss_ce = F.cross_entropy(logits, data_label)
                if self.args.SKD:
                    loss_kb = self.knowledge_boosting(prompt_feat['Language_project'], prompt_feat['Language_noproject'], word_embed, query_info, data_label)
                    loss = loss_ce + loss_kb
                else:
                    loss = loss_ce
                
                optim.zero_grad()
                loss.backward()
                
                optim.step()
                scheduler.step()
                pred = torch.argmax(logits, dim=1)
                acc = (pred == data_label).sum().item()/data_label.shape[0]*100.
                if self.args.SKD:
                    print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} loss_kb:{loss_kb.item():.4f} ACC: {acc}")
                else:
                    print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} ACC: {acc}")

    def triplet(self,cls_embed, vision_embed, query_info, train_label):
        P_head = query_info['proto'].clone().cuda()
    
        cls_logit = F.linear(cls_embed, P_head)
        cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')   #* B
        
        vis_logit = F.linear(vision_embed, P_head)
        vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')   #* B
        
        cls_vis = F.cross_entropy(cls_logit, torch.softmax(vis_logit, dim=1), reduction='none')   #* B
        loss_tri = -1*((cls_vis.mean() /(cls_vis.mean() + (cls_gt.mean() + vis_gt.mean())))+1e-6).log()
        
        return loss_tri

    def head_reg(self, head_feat, word_cur_feat, label):
        fc_wts = self.fc.weight
        fc_feat_sim = (1. - torch.cosine_similarity(fc_wts[label], head_feat, dim=1)).mean()
        return fc_feat_sim

    def knowledge_boosting(self, lang_embed_project, lang_embed_noproject,word_embed, query_info, label):
        P_head = query_info['proto'].clone().cuda()
        T = 2.
        lang_logit = F.linear(lang_embed_noproject, P_head)
        loss_seman = F.cross_entropy(lang_logit, label)
        
        loss_kd = F.kl_div(F.log_softmax(lang_embed_project/T,dim=1), F.softmax(word_embed[label]/T,dim=1), reduction='batchmean')
        loss = loss_kd + 0.2*loss_seman
        # return 0.5*loss
        return 0.1*loss
    
    def update_fc_avg(self,dataloader,class_list,query_info):
        self.eval()
        query_p=[]
        
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                cls_embed=self.encode(data_imgs).detach()
            
            for class_index in class_list:
                data_index=(label==class_index).nonzero().squeeze(-1)
                embedding = cls_embed[data_index]
                proto=embedding.mean(0)
                query_p.append(proto)
                self.fc.weight.data[class_index]=proto
            query_p = torch.stack(query_p)
        query_info["proto"] = torch.cat([query_info["proto"], query_p.cpu()])
        
        self.train()

    def init_base_fc(self,query,class_list):
        self.eval()
        with torch.no_grad():
            for class_index in class_list:
                self.fc.weight.data[class_index] = query[class_index]
    
    def get_logits(self,x, fc):
        return fc(x)




class CLIP_ViT_text_MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        
        if self.args.dataset in ['cifar100']:
            self.num_features = 768
        if self.args.dataset in ['mini_imagenet']:
            self.num_features = 768
        if self.args.dataset == 'cub200' or self.args.dataset == 'air':
            self.num_features = 768

        if args.scratch:
            self.CLIP_model = CLIPModel.from_pretrained('/data_25T/zlj/FSCIL/PriViLege-main/ckp/clip-vit-base-patch16')
        else:
            self.CLIP_model = CLIPModel.from_pretrained('/data_25T/zlj/FSCIL/PriViLege-main/ckp/clip-vit-base-patch16')        

        #* Prompt
        #todo Head 토큰 없애고 Vision으로 Pool
        self.prompt_length = 2 
        self.expert_length = 2 #* Number of tuning layers
        self.prompt = nn.Parameter(torch.randn(self.prompt_length,self.num_features))   #* VL
        self.expert_prompt = nn.Parameter(torch.randn(self.expert_length, 2, self.num_features))   #* B-Prompt (WC, MP)
        nn.init.uniform_(self.prompt, -1, 1)
        nn.init.uniform_(self.expert_prompt, -1, 1)
        #*------------------------------------------------------
        self.num_tokens = 197
        self.num_heads = self.CLIP_model.vision_model.encoder.layers[0].self_attn.num_heads
        
        self.comp_out = args.comp_out
        self.global_comp = nn.Conv1d(self.num_tokens + self.prompt_length, self.comp_out, kernel_size=1)
        nn.init.uniform_(self.global_comp.weight.data, -1, 1)
        
        self.local_comps = nn.ModuleList([nn.Conv1d(self.num_tokens+self.prompt_length, self.comp_out, kernel_size=1) for _ in range(self.num_heads)])
        for l_comp in self.local_comps:
            nn.init.uniform_(l_comp.weight.data, -1, 1)
                
        self.seen_classes = args.base_class
    #todo =======================================================================================
    
    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes,device='cuda')
        self.mask[:self.seen_classes]=-torch.inf
        self.seen_classes += len(new_classes)
    
    def forward_metric(self, x, B_tuning=False, eval=False):
        #? Original
        x = self.prompt_encode(x, prompt_feat=True, B_tuning=B_tuning, eval=eval)
        cls_emb, prompt_emb = x         
        return cls_emb, prompt_emb
    
    def encode(self, x):
        x = self.CLIP_model.get_image_features(x)
        return x
    
    def prompt_encode(self, img,prompt_feat=False, B_tuning=False, eval=False):
        x = self.CLIP_model.vision_model.embeddings(img)
        #*==============================================================
        #! VL-Prompt tuning
        prompting_tkn = self.prompt
        pos_tkn = prompting_tkn + (self.CLIP_model.vision_model.embeddings.position_embedding(self.CLIP_model.vision_model.embeddings.position_ids))[:,0].expand(self.prompt_length, -1)
        pos_tkn = pos_tkn.expand(x.shape[0],-1,-1)
        x = torch.cat([x[:,0].unsqueeze(1), pos_tkn, x[:,1:]],dim=1)#
        x= self.CLIP_model.vision_model.pre_layrnorm(x)
        #!=============================================================
        #* prefix for B-Prompt (Original)
        if B_tuning:
            #  Encoder
            x = self._forward_blocks(x, self.expert_prompt, eval=eval)
        else:
            x = self.CLIP_model.vision_model.encoder(x)
        
        x =  self.CLIP_model.visual_projection(x)
        cls_embed = x[:,0,:]
        if prompt_feat:
            prompt_embed ={}
            #todo Align -> Head
            prompt_embed['Vision'] = x[:,1,:]
            prompt_embed['Language'] =  x[:,2,:]
            
            return cls_embed, prompt_embed
        else:
            return cls_embed
    
    def _forward_blocks(self, x, prefix_tkn, eval=False):
        taskblock=[0,1]
        if len(taskblock) == len(self.CLIP_model.vision_model.encoder.layers) or  0 in taskblock:
            hidden_states = x
        for block_idx, block in enumerate(self.CLIP_model.vision_model.encoder.layers):
            if block_idx in taskblock:
                layer_outputs = self._pk_tuning(block, hidden_states, prefix_tkn[taskblock.index(block_idx)], eval=eval)
                hidden_states = layer_outputs[0]
            elif block_idx == 0:
                layer_outputs = block(x)
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = block(hidden_states=hidden_states,attention_mask=None,causal_attention_mask=None)
                hidden_states = layer_outputs[0]
        
        feat = self.CLIP_model.vision_model.post_layernorm(hidden_states)
        return feat

    def _extract_attn_mlp_feat(self, block, hidden_states):
        with torch.no_grad():
            residual = hidden_states
            hidden_states = block.layer_norm1(hidden_states)            
            attn = block.self_attn
            bsz, tgt_len, embed_dim = hidden_states.size()
            query_states = attn.q_proj(hidden_states) * attn.scale
            key_states = attn._shape(attn.k_proj(hidden_states), -1, hidden_states.shape[0])
            value_states = attn._shape(attn.v_proj(hidden_states), -1, hidden_states.shape[0])

            proj_shape = (bsz * attn.num_heads, -1, attn.head_dim)
            query_states = attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)

            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            if attn_weights.size() != (bsz * attn.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * attn.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=attn.dropout, training=attn.training)
            attn_output = torch.bmm(attn_probs, value_states)   # torch.Size([1536, 199, 64])

            if attn_output.size() != (bsz * attn.num_heads, tgt_len, attn.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, attn.num_heads, tgt_len, attn.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.view(bsz, attn.num_heads, tgt_len, attn.head_dim)   # torch.Size([128, 12, 199, 64])
            head_attentions = attn_output.transpose(1, 2)   # torch.Size([128, 199, 12, 64])
            attn_output = head_attentions.reshape(bsz, tgt_len, embed_dim)   # torch.Size([128, 199, 768])

            attn_output = attn.out_proj(attn_output)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = block.layer_norm2(hidden_states)

            mlp_feat = block.mlp(hidden_states)
            hidden_states = residual + mlp_feat
        return head_attentions, mlp_feat
        
    
    def _pk_tuning(self, block, hidden_states, prefix_tkn, eval=False):
        B,N,C = hidden_states.shape
        two,C = prefix_tkn.shape
        prefix_token = prefix_tkn.expand(B,two,C) #* B,2,768
        residual = hidden_states
        xq = block.layer_norm1(hidden_states)
        xk = xq.clone()
        xv = xq.clone()
        if self.args.prefix:
            xk = torch.cat([prefix_token[:,0].unsqueeze(1), xk],dim=1)
            xv = torch.cat([prefix_token[:,1].unsqueeze(1), xv],dim=1)
        else:
            head_attentions, mlp_feat = self._extract_attn_mlp_feat(block, hidden_states)
            global_feat = self.global_comp(mlp_feat).squeeze(1) #* (comp_out, dim) --> (Batch, 1, dim)
            #* Head_attentions: B, N, H, H_dim
            head_attentions = head_attentions.permute(2, 0, 1, 3)
            H, B, N, H_dim = head_attentions.shape
            
            head_feats = []
            for head_attn, local_comp in zip(head_attentions, self.local_comps):
                head_feats.append(local_comp(head_attn))    #* B,1,H_dim
            local_feat = torch.cat(head_feats, dim=1).reshape(B,-1)
            
            xk = torch.cat([(prefix_token[:,0] * local_feat).unsqueeze(1), xk],dim=1)
            xv = torch.cat([(prefix_token[:,1] * global_feat).unsqueeze(1), xv],dim=1)

        bsz, tgt_len, embed_dim = xq.size()
        attn = block.self_attn
        # get query proj
        query_states = attn.q_proj(xq) * attn.scale
        key_states = attn._shape(attn.k_proj(xk), -1, bsz)
        value_states = attn._shape(attn.v_proj(xv), -1, bsz)

        proj_shape = (bsz * attn.num_heads, -1, attn.head_dim)
        query_states = attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=attn.dropout, training=attn.training)
        attn_output = torch.bmm(attn_probs, value_states)   # torch.Size([1536, 199, 64])

        if attn_output.size() != (bsz * attn.num_heads, tgt_len, attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, attn.num_heads, tgt_len, attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, attn.num_heads, tgt_len, attn.head_dim)   # torch.Size([128, 12, 199, 64])
        head_attentions = attn_output.transpose(1, 2)   # torch.Size([128, 199, 12, 64])
        attn_output = head_attentions.reshape(bsz, tgt_len, embed_dim)   # torch.Size([128, 199, 768])

        attn_output = attn.out_proj(attn_output)

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = block.layer_norm2(hidden_states)

        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        
        return outputs
    
    def forward(self, img, label, prompt_feat=False, B_tuning=False, base=False, query=False, eval=False):
        if base:
            embedding = self.prompt_encode(img,prompt_feat=True, B_tuning=True, eval=eval)
            cls_embed, prompt_embed = embedding
            return  cls_embed, prompt_embed
        if query:
            q_feat = self.encode(img)
            return q_feat
        if self.mode == 'encoder':
            embedding = self.prompt_encode(img, prompt_feat=prompt_feat, B_tuning=B_tuning, eval=eval)
            if prompt_feat:
                cls_embed, prompt_embed = embedding
                return cls_embed, prompt_embed
            else:
                return embedding
        elif self.mode != 'encoder':
            img = self.forward_metric(img, B_tuning=B_tuning, eval=eval)
            return img
        
        else:
            raise ValueError('Unknown mode')

    def train_inc(self, dataloader, epochs, session, class_list, word_info, query_info):
        print("[Session: {}]".format(session))
        
        for idx,batch in enumerate(dataloader):
            data_imgs, data_label = [_.cuda() for _ in batch]
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
            
            word_cur_embed = word_info['cur_embed'].cuda()
            word_embed = word_info['embed'].cuda()
            pro_embed = query_info['proto']
            alpha = 0.5
            bata = 0.5
            gamma = 1.0
            for epoch in range(epochs):
                self.train()
                cls_embed, prompt_embed = self.prompt_encode(data_imgs ,prompt_feat=True, B_tuning=True)
                
                text_image_embeds = alpha*cls_embed+(1-alpha)*prompt_embed['Language']
                pro_image_embeds = bata*cls_embed+(1-bata)*prompt_embed['Vision']
                logits_text = self.get_text_logits(image_embeds=text_image_embeds,text_embeds=word_embed)
                logits_pro = self.get_pro_logits(image_embeds=pro_image_embeds,pro_embeds=pro_embed)
                logits_ = gamma*logits_text+(1-gamma)*logits_pro
                loss_ce = F.cross_entropy(logits_, data_label)                
                if self.args.SKD:
                    loss_kb = self.knowledge_boosting(prompt_embed['Language'],  word_embed, query_info, data_label)
                    loss = loss_ce + loss_kb
                else:
                    loss = loss_ce
                
                optim.zero_grad()
                loss.backward()
                
                optim.step()
                scheduler.step()
                pred = torch.argmax(logits_, dim=1)
                acc = (pred == data_label).sum().item()/data_label.shape[0]*100.
                if self.args.SKD:
                    print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} loss_kb:{loss_kb.item():.4f} ACC: {acc}")
                else:
                    print(f"[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} ACC: {acc}")

    def triplet(self,cls_embed, vision_embed, query_info, train_label):
        P_head = query_info['proto'].clone().cuda()
    
        cls_logit = F.linear(cls_embed, P_head)
        cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')   #* B
        
        vis_logit = F.linear(vision_embed, P_head)
        vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')   #* B
        
        cls_vis = F.cross_entropy(cls_logit, torch.softmax(vis_logit, dim=1), reduction='none')   #* B
        loss_tri = -1*((cls_vis.mean() /(cls_vis.mean() + (cls_gt.mean() + vis_gt.mean())))+1e-6).log()
        
        return loss_tri

    def head_reg(self, head_feat, word_cur_feat, label):
        fc_wts = self.fc.weight
        fc_feat_sim = (1. - torch.cosine_similarity(fc_wts[label], head_feat, dim=1)).mean()
        return fc_feat_sim

    def knowledge_boosting(self, lang_embed, word_embed, query_info, label):
        T = 2.
        idx= torch.arange(len(label))
        #* Original
        P_head = word_embed
        
        #* =======================================================================
        lang_logit = F.linear(lang_embed, P_head)    #* Soft pred
        loss_seman = F.cross_entropy(lang_logit, label)
        #* KL Feature
        loss_kd = F.kl_div(F.log_softmax(lang_embed/T,dim=1), F.softmax(word_embed[label]/T,dim=1), reduction='batchmean')
        
        loss = loss_kd + 0.2*loss_seman
        return 0.1*loss
    
    def update_fc_avg(self,dataloader,class_list,query_info):
        self.eval()
        query_p=[]
        
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                cls_embed=self.encode(data_imgs).detach()
            
            for class_index in class_list:
                data_index=(label==class_index).nonzero().squeeze(-1)
                embedding = cls_embed[data_index]
                proto=embedding.mean(0)
                query_p.append(proto)
                self.fc.weight.data[class_index]=proto
            query_p = torch.stack(query_p)
        query_info["proto"] = torch.cat([query_info["proto"], query_p.cpu()])
        
        self.train()

    def init_base_fc(self,query,class_list):
        self.eval()
        with torch.no_grad():
            for class_index in class_list:
                self.fc.weight.data[class_index] = query[class_index]
    
    def get_logits(self,x, fc):
        return fc(x)


    def get_text_logits(self,image_embeds,text_embeds):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.CLIP_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()    
        return logits_per_image


    def get_pro_logits(self,image_embeds,pro_embeds):
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        pro_embeds = pro_embeds / pro_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.CLIP_model.logit_scale.exp()
        logits_per_text = torch.matmul(pro_embeds.cuda(), image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()    
        return logits_per_image






