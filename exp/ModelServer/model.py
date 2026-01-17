import numpy as np
import torch
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data):
    alias_inputs, A1, A2, items, mask, inputs = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A1 = np.array(A1)
    A1 = trans_to_cuda(torch.Tensor(A1).float())
    A2 = np.array(A2)
    A2 = trans_to_cuda(torch.Tensor(A2).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A1, A2)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return model.compute_scores(seq_hidden, mask)