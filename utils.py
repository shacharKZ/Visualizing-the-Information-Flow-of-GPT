try:
    import hooks
except:
    main_project_path = __file__.split('Noisy_v3')[0]
    import os
    import sys
    sys.path.append(os.path.join(main_project_path, 'Noisy_v3'))
    from deeper_look_v2 import hooks as hooks

import functools
import pandas as pd
import copy

import transformers
import torch

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

# a safe way to get attribute of an object
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

# a safe way to set attribute of an object
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def wrap_model(model,  
               layers_to_check = ['.mlp', '.mlp.c_proj', '.mlp.c_fc', '', '.attn.c_attn', '.attn.c_proj', '.attn'],
               max_len=256):
    '''
    a wrapper function for model to collect hidden states
    returns a dictionary that is updated during the forward pass of the model
    and contains the hidden states of the layers specified in layers_to_check for each layer (collcting inputs and outputs of each)
    the dictionary has the following structure:
    {
        layer_idx: {
            layer_type: {
                'input': [list of hidden states (torch.tensor)],
                'output': [list of hidden states (torch.tensor)]
            }
        }
    }
    you can easily access the hidden states of a specific layer by using the following code:
    hs_collector[layer_idx][layer_type]['input'/'outputs'] # list of hidden states of the input of the layer
    to get the hidden state for the last forward pass, you can use:
    hs_collector[layer_idx][layer_type]['input'/'outputs'][-1] # the last hidden state of the input of the layer

    @ model: a pytorch model (currently only support gpt2 models from transformers library)
    @ layers_to_check: a list of strings that specify the layers to collect hidden states from
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed

    '''
    
    hs_collector = {}

    for layer_idx in range(model.config.n_layer):
        for layer_type in layers_to_check:
            list_inputs = []
            list_outputs = []
            
            layer_with_idx = f'{layer_idx}{layer_type}'
            layer_pointer = rgetattr(model, f"transformer.h.{layer_with_idx}")

            layer_pointer.register_forward_hook(hooks.extract_hs_include_prefix(list_inputs=list_inputs, 
                                                                    list_outputs=list_outputs, 
                                                                    info=layer_with_idx,
                                                                    max_len=max_len))

            if layer_idx not in hs_collector:
                hs_collector[layer_idx] = {}
            
            layer_key = layer_type.strip('.')
            if layer_key not in hs_collector[layer_idx]:
                hs_collector[layer_idx][layer_key] = {}

            hs_collector[layer_idx][layer_key]['input'] = list_inputs
            hs_collector[layer_idx][layer_key]['output'] = list_outputs

    return hs_collector
            

class model_extra:
    '''
    a class that contains extra functions for language models
    @ model: a pytorch model (currently only support gpt2 models from transformers library)
    @ model_name: the name of the model (e.g. 'gpt2'. if None, will be inferred from the model)
    @ tokenizer: the tokenizer of the model (if None, will be inferred from the model/model_name)
    '''
    def __init__(self, model, model_name=None, tokenizer=None, device=device):
        if model_name is None:
            model_name = model.config._name_or_path

        self.model_name = model_name
        
        if tokenizer is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        self.device = device

        # the decoding matrix and its layer norm. uses for logit lens
        self.ln_f = copy.deepcopy(model.transformer.ln_f).to(self.device).requires_grad_(False)
        self.embeding_matrix = copy.deepcopy(model.lm_head).to(self.device).requires_grad_(False)


    def hs_to_probs(self, hs, use_ln_f=True):
        '''
        return the probability of each token given a hidden state

        @ hs: a hidden state (torch.tensor) or a list/dataframe in the length of the model's hidden state
        @ use_ln_f: whether to use the final layer norm of the model (if True, the hs will be normalized before processing by the decoding matrix)
        '''
        if type(hs) != torch.Tensor:
            word_embed = torch.tensor(hs).to(self.device)
        else:
            word_embed = hs.clone().detach().to(self.device)
        if use_ln_f:
            word_embed = self.ln_f(word_embed)
        logic_lens = self.embeding_matrix(word_embed)
        probs = torch.softmax(logic_lens, dim=0).detach()
        return probs
    

    def hs_to_token_top_k(self, hs, k_top=12, k_bottom=12, use_ln_f=True):
        '''
        return the top and bottom k tokens given a hidden state according to logit of its projection by the decoding matrix

        @ hs: a hidden state (torch.tensor) or a list/dataframe in the length of the model's hidden state
        @ k_top: the number of top tokens to return
        @ k_bottom: the number of bottom tokens to return
        @ use_ln_f: whether to use the final layer norm of the model (if True, the hs will be normalized before processing by the decoding matrix)
        '''
        probs = self.hs_to_probs(hs, use_ln_f=use_ln_f)

        top_k = probs.topk(k_top)
        top_k_idx = top_k.indices
        # convert the indices to tokens
        top_k_words = [self.tokenizer.decode(i, skip_special_tokens=True) for i in top_k_idx]
        
        top_k = probs.topk(k_bottom, largest=False)
        top_k_idx = top_k.indices
        bottom_k_words = [self.tokenizer.decode(i, skip_special_tokens=True) for i in top_k_idx]
        
        return {'top_k': top_k_words, 'bottom_k': bottom_k_words}
    

    def infrence(self, model_, line, max_length='auto'):
        '''
        a wrapper for the model's generate function
        '''
        if type(max_length) == str and 'auto' in max_length:
            add = 1
            if "+" in max_length:
                add = int(max_length.split('+')[1])
            max_length = len(self.tokenizer.encode(line)) + add

        encoded_line = self.tokenizer.encode(
            line.rstrip(), return_tensors='pt').to(self.device)

        output = model_.generate(
            input_ids=encoded_line,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )

        answer_ = self.tokenizer.decode(
            output[:, encoded_line.shape[-1]:][0], skip_special_tokens=True)
        return line + answer_


    def infrence_for_grad(self, model_, line):
        '''
        a wrapper for the model's forward function
        '''
        encoded_line = self.tokenizer.encode(
            line.rstrip(), return_tensors='pt').to(self.device)

        return model_(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=True)
    