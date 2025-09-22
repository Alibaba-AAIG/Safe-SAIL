import torch
from torch.nn import Module
import einops

from nnsight import LanguageModel
from nnsight.models.LanguageModel import LanguageModelProxy
from nnsight.util import fetch_attr
from activation_function import JumpReLU, TopKReLU, ReLU, ActivationFunction

class SAEEncoder(Module):
    act_fn: ActivationFunction | ReLU
    def __init__(self, sae_params, model_index):
        super().__init__()
        self.pre_encoder_bias = sae_params['pre_encoder_bias._bias_reference'][model_index, :].unsqueeze(0).cuda()
        self.encoder_w = sae_params['encoder.weight'][model_index, :].unsqueeze(0).cuda()
        self.encoder_bias = sae_params['encoder.bias'][model_index, :].unsqueeze(0).cuda()
        # self.act_fn = ReLU()
        
    def forward(self, x):
        x = x - self.pre_encoder_bias
        learned_activations = (
        einops.einsum(
                x,
                self.encoder_w,
                "b s f, h o f -> b s h o",
            )
            + self.encoder_bias
        )
        acts = self.act_fn(learned_activations)
        return acts

class BaseReLUEncoder(SAEEncoder):
    def __init__(self, sae_params, model_index):
        super().__init__(sae_params, model_index)
        relu = ReLU()
        self.act_fn = relu

class TopKReLUEncoder(SAEEncoder):
    def __init__(self, sae_params, model_index, top_k):
        super().__init__(sae_params, model_index)
        topk_relu = TopKReLU(k=top_k)
        self.act_fn = topk_relu

class JumpReLUEncoder(SAEEncoder):
    def __init__(self, sae_params, model_index):
        super().__init__(sae_params, model_index)
        # encoder.theta torch.Size([1, 20480])
        self.encoder_theta = sae_params['encoder.theta'][model_index, :].unsqueeze(0).cuda()
        jump_relu = JumpReLU()
        self.act_fn = jump_relu

    def forward(self, x):
        x = x - self.pre_encoder_bias
        learned_activations = (
        einops.einsum(
                x,
                self.encoder_w,
                "b s f, h o f -> b s h o",
            )
            + self.encoder_bias
        )
        acts = self.act_fn(learned_activations, self.encoder_theta)
        return acts


        

def get_acts(model, inputs, layers):
    w_outs =[fetch_attr(model,'model.layers.'+str(i)+'.mlp.down_proj') for i in layers]
    acts=[]
    with model.trace(inputs):
        for k in range(len(layers)):
            layer_act = w_outs[k].output.detach().save()
            acts.append(layer_act)
    # acts_tensor = torch.cat(acts,dim=0) #(32,seq_ken,14336)
    # acts_tensor = acts_tensor.permute(1,0,2)
    acts_tensor = torch.stack(acts, dim=2) #(batch_size, seq_len, 32, 14336)
    # acts_tensor = acts_tensor.view(-1, 32, 14336) #(batch_size*seq_len, 32, 14336)
    return acts_tensor

def get_learned_activations(sae_encoder, acts_tensor): # (seq_len, n_layers=1, n_activations)
    assert acts_tensor.size(1) == 1, "only support n_layers=1"
    with torch.no_grad():
      learned_activations = sae_encoder(acts_tensor) # (seq_len, n_layers=1, n_learned_activations)
    return learned_activations