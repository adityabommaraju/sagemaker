import argparse
import json
import logging
import os
import sys
import pickle
import random
import datasets
import pandas as pd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_collate_fn(pad_index):
    
    def collate_fn(batch):
        batch_en_ids = [torch.tensor(example["en_ids"]) for example in batch]
        batch_de_ids = [torch.tensor(example["de_ids"]) for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch
    
    return collate_fn

def _get_train_data_loader(batch_size, training_dir,pad_index, is_distributed, **kwargs):
    logger.info("Get train data loader")

    # dataset  = pd.read_csv('train_data.csv')
    dataset = datasets.load_from_disk('train_data')
    # pickle_file = os.path.join(training_dir, 'train_data.pkl')

    # Load the data from the pickle file
    # with open(pickle_file, 'rb') as f:
    #     dataset = pickle.load(f)

    collate_fn = get_collate_fn(pad_index)

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
        **kwargs
    )

def _get_test_data_loader(batch_size, training_dir,pad_index,**kwargs):
    logger.info("Get test data loader")

    # dataset  = pd.read_csv('test_data.csv')
    dataset = datasets.load_from_disk('test_data')
    # pickle_file = os.path.join(training_dir, 'test_data.pkl')

    # # Load the data from the pickle file
    # with open(pickle_file, 'rb') as f:
    #     dataset = pickle.load(f)

    collate_fn = get_collate_fn(pad_index)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs
    )

def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)###input sentence
        trg = batch["en_ids"].to(device)###output sentence
        optimizer.zero_grad()### reset gradients to zero
        output = model(src, trg, teacher_forcing_ratio)### Forward prop(Encoder + Decoder)

        ### Reshaping outputs and target
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)###calculate loss
        loss.backward()### Calculate gradients through backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)### gradient clipping for handling exploding gradient
        optimizer.step()### update gradient
        epoch_loss += loss.item()###epoch loss

    return epoch_loss / len(data_loader)

def test(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0) # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    is_distributed=False

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir,args.pad_index, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, args.pad_index, **kwargs)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    input_dim = args.input_dim
    output_dim = args.output_dim

    encoder_embedding_dim = 2
    decoder_embedding_dim = 2
    hidden_dim = 1
    n_layers = 1
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    clip = 1.0
    teacher_forcing_ratio = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    model.apply(init_weights)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    # model = Net().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_index)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            src = batch["de_ids"].to(device)###input sentence
            trg = batch["en_ids"].to(device)###output sentence
            optimizer.zero_grad()### reset gradients to zero
            output = model(src, trg, teacher_forcing_ratio)### Forward prop(Encoder + Decoder)
            
            ### Reshaping outputs and target
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)###calculate loss
            loss.backward()### Calculate gradients through backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)### gradient clipping for handling exploding gradient
            optimizer.step()### update gradient
            epoch_loss += loss.item()###epoch loss

        test_loss = test(model, test_loader,criterion, device)

        print(f"\tTrain Loss: {epoch_loss:7.3f}")
        print(f"\tTest Loss: {test_loss:7.3f}")


    save_model(model, args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=2,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )
    parser.add_argument(
        "--pad-index",
        type=int,
        default=1,
        help="index for padding in the vocabulary",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=7853,
        help="size of input vocab",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=5893,
        help="size of output vocab",
    )

    # Container environment
    parser.add_argument("--hosts", type=list)
    parser.add_argument("--current-host", type=str)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--num-gpus", type=int)

    train(parser.parse_args())

