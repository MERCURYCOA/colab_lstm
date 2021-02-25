import torch
from torch import nn, optim
from torchtext import data, datasets
import numpy as np

# import en_core_web_sm
# nlp = en_core_web_sm.load()

print('GPU:', torch.cuda.is_available())
torch.manual_seed(123)

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

print('len of train data:', len(train_data))
print('len of test data:', len(test_data))

print(train_data.examples[15].text)
print(train_data.examples[15].label)

TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

batchsz = 80
device = torch.device('cuda')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batchsz,
    device=device
)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out
rnn = RNN(len(TEXT.vocab), 100, 256)
pretraining_embedding = TEXT.vocab.vectors
print('pretrained_embedding:', pretraining_embedding.shape)
rnn.embedding.weight.data.copy_(pretraining_embedding)
print('embedding layer initiated')

optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
criteon = nn.BCEWithLogitsLoss().to(device)
rnn.to(device)

def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc
def train(rnn, iterator, optimizer, criteon):
    avg_acc = []
    rnn.train()

    for i, batch in enumerate(iterator):
        pred = rnn(batch.text).squeeze(1)
        loss = criteon(pred, batch.label)
        acc = binary_acc(pred, batch.label).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10 == 0:
            print(i, acc)
        avg_acc = np.array(avg_acc).mean()
        print('avg_acc:', avg_acc)


def eval(rnn, iterator, criteon):
    avg_acc = []
    rnn.eval()
    with torch.no_grad():
        for batch in iterator:
            pred = rnn(batch.text).squeeze(1)
            loss = criteon(pred, batch.label)
            acc = binary_acc(pred, batch.label).item()
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    print('>>test:', avg_acc)

for epoch in range(10):
    eval(rnn, test_iterator, criteon)
    train(rnn, train_iterator, optimizer, criteon)
