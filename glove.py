import torch
from collections import defaultdict
from tqdm.auto import tqdm


def load_glove_embedding(load_path):
    with open(load_path, 'r', encoding='utf-8') as fin:
        # 第一行为词向量大小
        # n, d = map(int, fin.readline().split())
        # print(f'词表大小为{n},词向量维度为{d}')
        tokens = []
        embeds = []
        for index, line in tqdm(enumerate(fin), desc='glove embedding loading'):
            line = line.rstrip().split(' ')  # line.rstrip()返回删除 string 字符串末尾的指定字符,默认为空
            try:
                token, embed = line[0], list(map(float, line[1:]))
            except ValueError:
                print(index, line)
            else:
                tokens.append(token)
                embeds.append(embed)
        vocab = Vocab(tokens)
        embeds = torch.tensor(embeds, dtype=torch.float)
    return vocab, embeds


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()  # 词表
        self.token_to_idx = dict()  # 词表及对应单词位置

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 标记每个单词的位置
            self.unk = self.token_to_idx['<unk>']  # 开始符号的位置

    @classmethod
    # 不需要实例化，直接类名.方法名()来调用 不需要self参数，但第一个参数需要是表示自身类的cls参数,
    # 因为持有cls参数，可以来调用类的属性，类的方法，实例化对象等
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小，即词表中有多少个互不相同的标记
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入标记对应的索引值，如果该标记不存在，则返回标记<unk>的索引值（0）
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标记对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in indices]


if __name__ == '__main__':
    glove_embedding_path = '../../../../dEFEND-Pytorch-master/data/embeddings_data/glove.6B.100d.txt'
    vocab_, embeds_ = load_glove_embedding(glove_embedding_path)
    # print(embeds.shape)
    # print(vocab.idx_to_token)
