import torch

def extract_hs_include_prefix(list_inputs, list_outputs, info='', max_len=256):
    '''
    return a hook function that extract the hidden states (hs) before and after the layer

    @ list_inputs: a list. it will be appended with the hs before the layer (torch.tensor)
    @ list_outputs: a list. it will be appended with the hs after the layer (torch.tensor)
    @ info: a string to use while debugging
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed
    
    implemention note for future developers:
    - note we use the easiest way to save the hs, by just appending a copy of the hs to a list
    - if you are going to save this data later to a pickle file, you might want to first change the information 
        from torch.tensor wrapped with list, to pandas or numpy. from our experience that can save a lot of space
    - the information is saved without gradient. if you want to save the gradient you can try and also save it separately
    - use the info parameter to identify the layer you are extracting the hs from (we left the comment from our debugging. it might be useful for you)
    - you should verify that the model is not implemented in a way that the hs is not saved in the same order as the input or it processes 
        them inplace so this information is not representative
    '''
    def hook(module, input, output):
        if list_inputs is not None:
            # print('input[0].shape', input[0].shape, f'[{info}]')
            last_tokens = input[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]
            
            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                list_inputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)

        if list_outputs is not None:
            last_tokens = output[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                # print('last_token.shape', last_token.shape, f'[{info}]')
                list_outputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)
                
    return hook

