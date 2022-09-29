import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENT_BGN = ['[E0]', '[E1]', '[E2]']
ENT_END = ['[/E0]', '[/E1]', '[/E2]']
ENT_BLN = '[BLANK]'


def create_bert_tokenizer_and_model():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    bert_model = BertModel.from_pretrained('bert-large-uncased').to(DEVICE)

    # add the entity markers and blank

    tokenizer.add_special_tokens({
        'additional_special_tokens': ENT_BGN + ENT_END + [ENT_BLN]
    })
    bert_model.resize_token_embeddings(len(tokenizer))

    return tokenizer, bert_model


def ent_token_ids(tokenizer):
    return {
        'bgn': tokenizer.additional_special_tokens_ids[:3],
        'end': tokenizer.additional_special_tokens_ids[3:6],
        'bln': tokenizer.additional_special_tokens_ids[6]
    }


class BertWithEntityStartPooling(nn.Module):

    def __init__(self, bert, ent_token_ids):
        super(BertWithEntityStartPooling, self).__init__()

        self._bert = bert
        self._ent_bgn_ids = ent_token_ids['bgn']
        self._h = self._bert.config.hidden_size
        self.hidden_size = 2 * self._h

    def forward(self, input):
        batched_hidden_states = self._bert(**input).last_hidden_state
        batched_result = []

        for input_ids, masks, hidden_states in zip(
                input['input_ids'],
                input['attention_mask'],
                batched_hidden_states
        ):
            ei_to_h = [[], [], []]

            for input_id, mask, h in zip(input_ids, masks, hidden_states):
                if not mask:
                    continue

                try:
                    ei = self._ent_bgn_ids.index(input_id)
                    ei_to_h[ei].append(h)
                except:
                    pass

            for ei, hs in enumerate(ei_to_h):
                if hs:
                    ei_to_h[ei] = torch.cat([
                        h.unsqueeze(0)
                        for h in hs
                    ])
                else:
                    ei_to_h[ei] = None

            for ei, hs in enumerate(ei_to_h):
                if hs != None:
                    hs = torch.transpose(hs, 0, 1).unsqueeze(0)
                    ei_to_h[ei] = F.max_pool1d(hs, hs.shape[-1]).squeeze()

                    assert ei_to_h[ei].shape == (self._h,)
                else:
                    ei_to_h[ei] = torch.zeros(self._h).to(DEVICE)

            output = []
            for ei, ej in [(0, 1), (0, 2), (1, 2)]:
                output.append(torch.cat((ei_to_h[ei], ei_to_h[ej])).unsqueeze(0))

            output = torch.cat(output).unsqueeze(0)
            batched_result.append(output)

        return torch.cat(batched_result)


class FullyConnectedLayer(nn.Module):
    """
    A fully connected layer with an optional activation function.
    """

    def __init__(self, input, hidden, output, activation_fn=None):
        super(FullyConnectedLayer, self).__init__()

        self._linear1 = nn.Linear(input, hidden)
        self._activation_fn = activation_fn or nn.Identity()
        self._linear2 = nn.Linear(hidden, output)

    def forward(self, input):
        return self._linear2(self._activation_fn(self._linear1(input)))


class PerClassScore(nn.Module):

    def __init__(self, pcr_size, num_relation_ids):
        super(PerClassScore, self).__init__()
        self._pcr = nn.Parameter(torch.randn(pcr_size, num_relation_ids))

    def forward(self, input):
        result = torch.matmul(input, self._pcr)
        result = F.log_softmax(result, dim=2)
        return result


class TriBlank(nn.Module):

    def __init__(self, bert, ent_token_ids, num_relation_ids):
        super(TriBlank, self).__init__()

        bwesp = BertWithEntityStartPooling(bert, ent_token_ids)
        h = bwesp.hidden_size
        fcl = FullyConnectedLayer(h, h // 2, h // 2)
        pcr = PerClassScore(h // 2, num_relation_ids)

        self._seq = nn.Sequential(
            bwesp,
            fcl,
            pcr
        )

    def forward(self, input):
        return self._seq(input)


