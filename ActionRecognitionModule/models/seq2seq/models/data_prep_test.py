import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random
import math
import time
from seq2seq import Encoder, Decoder, Seq2Seq

SEED = 1234

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(
        tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True
    )

    TRG = Field(
        tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True
    )

    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(SRC, TRG)
    )

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print(vars(train_data.examples[0]))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device,
    )

    print("train_iterator", train_iterator)

    for i, batch in enumerate(train_iterator):

        src = batch.src
        print("src", src)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters())

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    def train(model, iterator, optimizer, criterion, clip):

        model.train()

        epoch_loss = 0

        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(model, iterator, criterion):

        model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for _, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = model(src, trg, 0)  # turn off teacher forcing
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )
