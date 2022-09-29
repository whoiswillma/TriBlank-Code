import torch

from TriBlank import DEVICE, create_bert_tokenizer_and_model, TriBlank, ent_token_ids
from docred_util import format_example, RELATION_ID_TO_TEXT, ALL_RELATION_IDS, DOCRED_TRAIN


def _run_model(model, tokenizer, batch, eid_and_bln=((0, False), (1, False), (2, True))):
    model.eval()

    input = tokenizer(
        [format_example(example, eid_and_bln) for example in batch],
        padding=True,
        return_tensors='pt'
    ).to(DEVICE)

    output = model(input)
    print(output)
    print()

    for i in range(len(batch)):
        example = batch[i]
        for ei in range(len(eid_and_bln) - 1):
            for ej in range(ei + 1, len(eid_and_bln)):
                assert ei + ej - 1 in (0, 1, 2)

                pred = torch.argmax(output[i][ei + ej - 1]).item()
                print(example['vertexSet'][eid_and_bln[ei][0]][0]['name'])
                print(RELATION_ID_TO_TEXT[ALL_RELATION_IDS[pred]])
                print(example['vertexSet'][eid_and_bln[ej][0]][0]['name'])
                print()


def _demo():
    tokenizer, bert_model = create_bert_tokenizer_and_model()
    model = TriBlank(bert_model, ent_token_ids(tokenizer), len(ALL_RELATION_IDS)).to(DEVICE)
    _run_model(model, tokenizer, [DOCRED_TRAIN[5], DOCRED_TRAIN[6]])


if __name__ == '__main__':
    _demo()
