import torch
import string, re
import torchtext
from torchtext.data import Field, Iterator, TabularDataset

import torch.nn.functional as F
import torch.nn as nn
import torchkeras

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20


# 过滤点低频词
def filter_low_freq_words(arr, vocab):
    arr = [[x if x < MAX_WORDS else 0 for x in example] for example in arr]
    return arr


# 1.定义各个字段的预处理方法
TEXT = Field(
    sequential=True,
    tokenize=lambda x: re.sub('[%s]' % string.punctuation, "", x).split(" "),
    lower=True,
    fix_length=MAX_LEN,
    postprocessing=filter_low_freq_words)
LABEL = Field(sequential=False, use_vocab=False)

# 2.构建表格型dataset
ds_train, ds_test = TabularDataset.splits(path='./data/',
                                          train='train.tsv',
                                          test='test.tsv',
                                          format='tsv',
                                          fields=[('label', LABEL),
                                                  ('text', TEXT)],
                                          skip_header=False)

# 3.构建词典
TEXT.build_vocab(ds_train)

# 4.构建数据管道迭代器
train_iter, test_iter = Iterator.splits((ds_train, ds_test),
                                        sort_within_batch=True,
                                        sort_key=lambda x: len(x.text),
                                        batch_sizes=(BATCH_SIZE, BATCH_SIZE),
                                        device='cuda:4')


# 将数据管道组织成torch.utils.data.DataLoader相似的features,label输出形式
class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意：此处调整features为 batch first，并调整label的shape和dtype
        for batch in self.data_iter:
            yield (torch.transpose(batch.text, 0, 1),
                   torch.unsqueeze(batch.label.float(), dim=1))


dl_train = DataLoader(train_iter)
dl_test = DataLoader(test_iter)

torch.random.seed()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # matmul and scale
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # softmax
        attn = self.dropout(F.softmax(attn, dim=-1))

        # matmul
        output = torch.matmul(attn, v)

        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiheadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # pass through the pre-attention projection: b x lg x (n*dv)
        # seperate different heads: b x lg x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_qs(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_qs(v).view(sz_b, len_v, n_head, d_v)

        # transpose for attention dor product: b x n x lg x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # for head axis broadcasting

        q, attn = self.attention(q, k, v, mask=mask)

        # transpose to move the head dimension back: b x lq x n x dv
        # combine the last two dimensions to concatenate all the heads together:b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class BiLSTM_Attention(torchkeras.Model):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(MAX_WORDS, 100, 1)
        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        # self.attention = MultiheadAttention(8, 256, 64, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.dense = nn.Sequential()
        self.dense.add_module("linear", nn.Linear(128 * 2, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, 128 * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2),
                            soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context

    def forward(self, x):
        x = self.embedding(x)
        x, (final_hidden_state, final_cell_state) = self.lstm(x)
        x = self.dropout(x)
        x = self.attention_net(x, final_hidden_state)
        y = self.dense(x)
        return y


model = BiLSTM_Attention()


# 准确率
def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5,
                         torch.ones_like(y_pred, dtype=torch.float32),
                         torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1 - torch.abs(y_true - y_pred))
    return acc


model.to('cuda:4')

model.compile(loss_func=nn.BCELoss(),
              optimizer=torch.optim.Adagrad(model.parameters(), lr=0.02),
              metrics_dict={"accuracy": accuracy},
              device='cuda:4')

# 有时候模型训练过程中不收敛，需要多试几次
dfhistory = model.fit(20, dl_train, dl_val=dl_test, log_step_freq=200)