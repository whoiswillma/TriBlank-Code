import random

import torch
from torch import optim

from TriBlank import create_bert_tokenizer_and_model, TriBlank, ent_token_ids, DEVICE

from docred_util import ALL_RELATION_IDS, DOCRED_TRAIN

from datetime import datetime
import os

from schemes.re2 import map_to_re2_dataset, train_epoch_re2, eval_contingency_table_re2


def get_path(model_name):
    return os.path.join('persist', model_name)


def get_default_name(model):
    return type(model).__name__ + '-' + datetime.now().strftime('%m-%d@%H-%M')


def save_model(model, name=None):
    name = name or (get_default_name(model) + '.pt')
    path = get_path(name)

    print('Saving', name)
    torch.save(model.state_dict(), path)
    return model


def load_model(model, name):
    path = get_path(name)
    model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
    return model


def get_default_checkpoint_name(model, epoch):
    return 'Checkpoint-{}-e({}).pt'.format(get_default_name(model), epoch)


def save_checkpoint(model, optim, epoch):
    name = get_default_checkpoint_name(model, epoch)
    path = get_path(name)

    print('Saving', name)
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optim.state_dict()
        },
        path
    )
    return model


def load_checkpoint(model, optim, name):
    path = get_path(name)

    checkpoint = torch.load(path, map_location=torch.device(DEVICE))

    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']


def main():
    tokenizer, bert_model = create_bert_tokenizer_and_model()
    re2_dataset = map_to_re2_dataset(DOCRED_TRAIN)
    model = TriBlank(bert_model, ent_token_ids(tokenizer), len(ALL_RELATION_IDS)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=2e-6)

    save_checkpoint(model, opt, 0)
    save_model(model)

    for epoch in range(1, 4):
        random.shuffle(re2_dataset)
        print(f'Training epoch {epoch}')
        train_epoch_re2(
            model, tokenizer, opt, DOCRED_TRAIN, re2_dataset[:2000],
            batch_size=3, max_grad_norm=None, blank_alpha=0.7
        )
        save_checkpoint(model, opt, epoch)
        save_model(model)

        random.shuffle(re2_dataset)
        tbl = eval_contingency_table_re2(model, tokenizer, DOCRED_TRAIN, re2_dataset[:5000])
        acc_train = (
                sum(tbl[i][i] for i in range(len(tbl)))
                / sum(tbl[i][j] for i in range(len(tbl)) for j in range(len(tbl)))
        )
        print(f'Sampled Training Accuracy (n=5000): {acc_train}')


if __name__ == '__main__':
    main()
