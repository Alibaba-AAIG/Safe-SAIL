import os
import torch
import numpy as np
from tqdm import tqdm
from sae import TopKReLUEncoder
from nnsight import LanguageModel
from quantile_utils import approximate_quantile
from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

def setup_source_model(model_path):
    print(torch.cuda.is_available())
    hf_config = LlamaConfig.from_pretrained(model_path, use_cache=False)
    print(hf_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = LanguageModel(model_path,device_map='cuda',torch_dtype="bfloat16")
    model.eval()
    return model, tokenizer

def setup_sae_encoder(model_paths):
    sae_encoder_list = []
    for layer in model_paths:
        local_path = model_paths[layer]
        sae_params = torch.load(local_path, map_location='cuda:0')
        sae_encoder = TopKReLUEncoder(sae_params, model_index=0, top_k=200)
        sae_encoder_list.append(sae_encoder)
    return sae_encoder_list

def setup_selected_neuron_indices(indices_paths):
    neuron_indices_list = []
    for layer in indices_paths:
        local_path = indices_paths[layer]
        neuron_indices = torch.load(local_path, map_location='cpu')
        neuron_indices_list.append(neuron_indices.cpu().tolist())
    return neuron_indices_list

def setup_quantiles(quantile_paths):
    quantile_local_paths = []
    for layer in quantile_paths:
        local_path = quantile_paths[layer]
        quantile_local_paths.append(local_path)
    layer_ids = [0,8,17,26,35]
    quantile_list = []
    for idx, local_path in enumerate(quantile_local_paths):
        quantile = torch.load(local_path)
        k = 10000
        bottom_numpy = quantile['bottom_neuron_activations'].permute((1,0)).cpu().numpy()
        top_numpy = quantile['top_neuron_activations'].permute((1,0)).cpu().numpy()
        key = 1-1e-3
        quantiles = approximate_quantile(
            key,
            200000,
            10000,
            bottom_numpy,
            top_numpy,
        )
        quantiles = torch.tensor(quantiles)
        quantile_list.append(quantiles.to("cuda"))
    return quantile_list



