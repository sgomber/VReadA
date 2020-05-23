import warnings
warnings.filterwarnings('ignore')

import glob
import time
import math

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.utils import download

import gluonnlp as nlp

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

# Note that ctx is short for context
def evaluate(model, data_source, batch_size, ctx):
    # Specify the loss function, in this case, cross-entropy with softmax.
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(
        batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = loss(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def get_perplexity(text,model_data):
    awd_model = model_data[0]
    vocab = model_data[1]
    context = model_data[2]

    bptt = 2
    batch_size = 2*len(context)
    # Batchify for BPTT
    bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
        vocab, bptt, batch_size, last_batch='discard')
    
    data = []
    for token in text:
        data.append(token.text)
    batched_data = bptt_batchify(data)
    test_L = evaluate(awd_model, batched_data, batch_size, context[0])
    return math.exp(test_L)











# print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))

# # print(awd_model)
# # print(vocab)
