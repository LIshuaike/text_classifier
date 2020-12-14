import torch
import string, re
import torchtext
from torchtext.data import Field, Iterator, TabularDataset

import torch.nn as nn
import torchkeras

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20

# 分词方法
# tokenizer = lambda x: re.sub(f'{string.punctuation}', '', x).split(' ')
# tokenizer = lambda x: re.sub('[%s]' % string.punctuation, "", x).split(" ")


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
ds_train, ds_test = TabularDataset.splits(path='./data',
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


class Net(torchkeras.Model):
    def __init__(self):
        super(Net, self).__init__()

        #设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings=MAX_WORDS,
                                      embedding_dim=100,
                                      padding_idx=1)
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_1", nn.Conv1d(in_channels=100,
                                out_channels=16,
                                kernel_size=5))
        self.conv.add_module("pool_1", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_1", nn.ReLU())
        self.conv.add_module(
            "conv_2", nn.Conv1d(in_channels=16,
                                out_channels=128,
                                kernel_size=2))
        self.conv.add_module("pool_2", nn.MaxPool1d(kernel_size=2))
        self.conv.add_module("relu_2", nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module("flatten", nn.Flatten())
        self.dense.add_module("linear", nn.Linear(6144, 1))
        self.dense.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        y = self.dense(x)
        return y


model = Net()


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
dfhistory = model.fit(10, dl_train, dl_val=dl_test, log_step_freq=200)