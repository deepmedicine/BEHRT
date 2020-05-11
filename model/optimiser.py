import pytorch_pretrained_bert as Bert

def adam(params, config=None):
    if config is None:
        config = {
            'lr': 3e-5,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0}
    ]

    optim = Bert.optimization.BertAdam(optimizer_grouped_parameters,
                                       lr=config['lr'],
                                       warmup=config['warmup_proportion'])
    return optim