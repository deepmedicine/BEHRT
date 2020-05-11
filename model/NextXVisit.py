import torch
import torch.nn as nn
import pytorch_pretrained_bert as Bert
import numpy as np


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config, feature_dict=None):
        super(BertEmbeddings, self).__init__()

        if feature_dict is None:
            self.feature_dict = {
                'word': True,
                'seg': True,
                'age': True,
                'position': True
            }
        else:
            self.feature_dict = feature_dict

        if feature_dict['word']:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        if feature_dict['seg']:
            self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)

        if feature_dict['age']:
            self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)

        if feature_dict['position']:
            self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size). \
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids, seg_ids, posi_ids):
        embeddings = self.word_embeddings(word_ids)

        if self.feature_dict['seg']:
            segment_embed = self.segment_embeddings(seg_ids)
            embeddings = embeddings + segment_embed

        if self.feature_dict['age']:
            age_embed = self.age_embeddings(age_ids)
            embeddings = embeddings + age_embed

        if self.feature_dict['position']:
            posi_embeddings = self.posi_embeddings(posi_ids)
            embeddings = embeddings + posi_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)


class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, feature_dict):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config, feature_dict=feature_dict)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids, seg_ids, posi_ids, attention_mask,
                output_all_encoded_layers=True):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForMultiLabelPrediction(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels, feature_dict):
        super(BertForMultiLabelPrediction, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, feature_dict)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.MultiLabelSoftMarginLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits