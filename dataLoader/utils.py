import random


def code2index(tokens, token2idx, mask_token=None):
    output_tokens = []
    for i, token in enumerate(tokens):
        if token==mask_token:
            output_tokens.append(token2idx['UNK'])
        else:
            output_tokens.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_tokens


def random_mask(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def seq_padding(tokens, max_len, token2idx=None, symbol=None, unkown=True):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                if unkown:
                    seq.append(token2idx.get(tokens[i], token2idx['UNK']))
                else:
                    seq.append(token2idx.get(tokens[i]))
            else:
                seq.append(token2idx.get(symbol))
    return seq