import os

import pandas as pd
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import re
import nltk
from nltk import tokenize

import torch
from torch.utils.data import random_split, Dataset, DataLoader

from glove import Vocab
from config import Config


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


class Preprocessor:

    @staticmethod
    def basic_pipeline(sentences):
        sentences = Preprocessor.replaceImagesURLs(sentences)  # 数据清洗
        sentences = Preprocessor.tokenize_nltk(sentences)  # 分词
        sentences = Preprocessor.removeStopwords(sentences)  # 去停用词
        return sentences

    @staticmethod
    def replaceImagesURLs(sentences):
        out = []
        url_token = 'URLTOKEN'
        img_token = 'IMGTOKEN'

        for s in sentences:
            s = re.sub(r'(http://)?www.*?(\s|$)', ' ' + url_token + '\\2', s)  # URL containing www
            s = re.sub(r'http://.*?(\s|$)', ' ' + url_token + '\\1', s)  # URL starting with http
            s = re.sub(r'\w+?@.+?\\.com.*', ' ' + url_token, s)  # email
            s = re.sub(r'\[img.*?\]', ' ' + img_token, s)  # image
            s = re.sub(r'< ?img.*?>', ' ' + img_token, s)
            out.append(s)
        return out

    @staticmethod
    def stopwordsList():
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.append('...')
        stopwords.append('___')
        stopwords.append('<url>')
        stopwords.append('<img>')
        stopwords.append('URLTOKEN')
        stopwords.append('IMGTOKEN')
        stopwords.append("can't")
        stopwords.append("i've")
        stopwords.append("i'll")
        stopwords.append("i'm")
        stopwords.append("that's")
        stopwords.append("n't")
        stopwords.append('rrb')
        stopwords.append('lrb')
        return stopwords

    @staticmethod
    def removeStopwords(sentences):
        stopwords = Preprocessor.stopwordsList()
        finished_sentences = []
        for sentence in sentences:
            finished_sentence = []
            for word in sentence:
                if word not in stopwords:
                    finished_sentence.append(word)
            finished_sentences.append(finished_sentence)
        return finished_sentences

    @staticmethod
    def tokenize_simple(sentences):
        return [sentence.split(' ') for sentence in sentences]

    @staticmethod
    def tokenize_nltk(sentences):
        return [nltk.word_tokenize(sentence) for sentence in sentences]


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None时，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def load_data(data_filepath, platform='politifact'):
    """
    :param data_filepath: 文件路径
    :param platform: 数据集类别
    """
    data_train = pd.read_csv(data_filepath + os.sep + platform + '_content_no_ignore.tsv', sep='\t')

    contents, labels, texts, ids = [], [], [], []
    for idx in tqdm(range(data_train.content.shape[0]), desc='contents processing'):
        text = data_train.content[idx]
        text = BeautifulSoup(text, features="html5lib")
        text = clean_str(text.get_text().encode('ascii', 'ignore').decode())
        texts.append(text)

        sentences = tokenize.sent_tokenize(text)  # 分句 返回的是一个列表
        sentences = Preprocessor.basic_pipeline(sentences)
        contents.append(sentences)
        ids.append(data_train.id[idx])
        labels.append(data_train.label[idx])

    # load user comments
    comments, comments_text = [], []
    comments_train = pd.read_csv(data_filepath + os.sep + platform + '_comment_no_ignore.tsv', sep='\t')
    content_ids = set(ids)

    for idx in tqdm(range(comments_train.comment.shape[0]), desc='comments processing'):
        if comments_train.id[idx] in content_ids:
            com_text = comments_train.comment[idx]
            com_text = BeautifulSoup(com_text, features="html5lib")
            com_text = clean_str(com_text.get_text().encode('ascii', 'ignore').decode())

            tmp_comments = [ct for ct in com_text.split('::')]
            tmp_comments = Preprocessor.basic_pipeline(tmp_comments)
            tmp_comments = [comment for comment in tmp_comments if len(comment) != 0]
            comments.append(tmp_comments)
            comments_text.extend(tmp_comments)

    return contents, comments, labels


def build_vocab(contents, comments):
    """
    词表映射
    :param contents: 新闻内容
    :param comments: 评论内容
    """
    assert len(contents) == len(comments)
    all_tokens = []
    for ind in range(len(contents)):
        content_tokens = [word for content in contents[ind] for word in content]
        comments_tokens = [word for comment in comments[ind] for word in comment]
        all_tokens.append(content_tokens)
        all_tokens.append(comments_tokens)

    return Vocab.build(all_tokens)


def tokens_to_ids(vocab, data_split):
    """
    将tokens转换为ids
    :param vocab: 词表
    :param data_split: 切分后的数据集
    """
    data_split_ids = []
    for (contents, comments, label) in data_split:
        content_ids = [vocab.convert_tokens_to_ids(content) for content in contents]
        comment_ids = [vocab.convert_tokens_to_ids(comment) for comment in comments]
        data_split_ids.append((content_ids, comment_ids, label))
    return data_split_ids


class DataMapping(Dataset):
    """Dataset Mapping"""
    def __init__(self, data):
        self.dataset = data
        self.lens = len(data)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample

    def __len__(self):
        return len(self.dataset)


class FakeNewsDataset:
    def __init__(self, config):
        self.data_filepath = config.dataset_dir
        self.platform = config.platform
        self.validation_split = config.validation_split
        self.max_sentence_length = config.max_sentence_length
        self.max_sentence_count = config.max_sentence_count
        self.max_comment_length = config.max_comment_length
        self.max_comment_count = config.max_comment_count
        self.batch_size = config.batch_size
        self.batch_first = True
        self.padding_value = 0

    def data_process(self, only_test=False):
        # 1.obtain data
        contents, comments, labels = load_data(self.data_filepath, self.platform)
        data = [(contents[ind], comments[ind], labels[ind]) for ind in range(len(contents))]
        # 2.build dictionary
        vocab = build_vocab(contents, comments)
        # 3.dataset split
        train_data, val_data = self.split_data(data)

        if only_test:
            val_data_map = val_data
            # 4.map tokens to ids
            val_data = tokens_to_ids(vocab, val_data)
            # 5.dataset mapping
            val_data = DataMapping(val_data)
            # 6.dataset encapsulation for iteration
            val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
            return val_iter, vocab, val_data_map

        # 4.map tokens to ids
        train_data = tokens_to_ids(vocab, train_data)  # (content_ids, comment_ids, label)
        val_data = tokens_to_ids(vocab, val_data)
        # 5.dataset mapping
        train_data = DataMapping(train_data)
        val_data = DataMapping(val_data)
        # 6.dataset encapsulation for iteration
        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        return train_iter, val_iter, vocab

    def split_data(self, data):
        """
        data split
        """
        num_val = int(len(data) * self.validation_split)
        num_train = len(data) - num_val
        train_data, val_data = random_split(data, [num_train, num_val], generator=torch.Generator().manual_seed(123))
        # train_data, val_data = random_split(data, [num_train, num_val])
        return train_data, val_data

    def generate_batch(self, batch_data):
        batch_contents = torch.zeros((len(batch_data), self.max_sentence_count, self.max_sentence_length), dtype=torch.long)
        batch_comments = torch.zeros((len(batch_data), self.max_comment_count, self.max_comment_length), dtype=torch.long)
        batch_content_lengths, batch_content_counts, batch_comment_lengths, batch_comment_counts = [], [], [], []
        for idx, (contents, comments, label) in enumerate(batch_data):
            # news contents
            contents_tensor, content_lengths, content_count = self.pad_data(contents, self.max_sentence_length,
                                                                            self.max_sentence_count)
            batch_contents[idx][:len(contents_tensor)] = contents_tensor
            batch_content_lengths.append(content_lengths)
            batch_content_counts.append(content_count)

            # user comments
            comments_tensor, comment_lengths, comment_count = self.pad_data(comments, self.max_comment_length,
                                                                            self.max_comment_count)
            batch_comments[idx][:len(comments_tensor)] = comments_tensor
            batch_comment_lengths.append(comment_lengths)
            batch_comment_counts.append(comment_count)

        batch_content_counts = torch.tensor(batch_content_counts, dtype=torch.long)
        batch_comment_counts = torch.tensor(batch_comment_counts, dtype=torch.long)
        labels = torch.tensor([label for (contents, comments, label) in batch_data], dtype=torch.long)
        batch = (batch_contents, batch_content_lengths, batch_content_counts, batch_comments, batch_comment_lengths,
                 batch_comment_counts, labels)
        return batch

    def pad_data(self, sentences, max_length, max_count):
        """
        此函数用于将一条样本（新闻和评论）中的序列补齐
        :param sentences: 新闻或评论序列 sentences -> list 一条样本  sentences[0] -> list 一条样本中的第一个句子  sentences[0][0] -> str word
        :param max_length: 新闻或评论中句子的最大长度
        :param max_count: 新闻中句子的数量或评论的最大数量
        """
        sentence_tensor = [torch.tensor(sentence, dtype=torch.long) for sentence in sentences]
        sentence_tensor = pad_sequence(sentence_tensor, batch_first=self.batch_first, max_len=max_length,
                                       padding_value=self.padding_value)
        sentence_tensor = sentence_tensor[:max_count]

        sentence_lengths = []
        for sentence in sentences:
            if len(sentence) >= max_length:
                sentence_lengths.append(max_length)
            else:
                sentence_lengths.append(len(sentence))

        # 记录文档的长度，即文档中句子的个数
        if len(sentence_lengths) >= max_count:
            sentence_lengths = sentence_lengths[:max_count]
            sentence_count = max_count  # int
        else:
            sentence_count = len(sentence_lengths)

        sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.long)
        return sentence_tensor, sentence_lengths, sentence_count


def activation_maps(test_data_map, sentences_prob, comments_prob, y_true, y_prob, saved_filepath):
    """
    :param test_data_map: the text of contents and comments
    :param sentences_prob: contents attention weights
    :param comments_prob: comments attention weights
    :param y_true: true label
    :param y_prob: predict label
    :param saved_filepath: file save path
    """
    content_comment_weights = []
    for idx in range(len(test_data_map)):
        content, comment, label = test_data_map[idx]
        content_att_prob = sentences_prob[idx]
        comment_att_prob = comments_prob[idx]
        # 此处由于在计算时将句子按照最大长度阶段或补齐, 因此要考虑序列长度的问题
        if len(content_att_prob) >= len(content):
            con_and_att = [(con, round(content_att_prob[i].item(), 4)) for i, con in enumerate(content)]
        else:
            con_and_att = [(content[i], round(con_w.item(), 4)) for i, con_w in enumerate(content_att_prob)]

        if len(comment_att_prob) >= len(comment):
            com_and_att = [(com, round(comment_att_prob[j].item(), 4)) for j, com in enumerate(comment)]
        else:
            com_and_att = [(comment[j], round(com_w.item(), 4)) for j, com_w in enumerate(comment_att_prob)]
        content_comment_weights.append((con_and_att, com_and_att))

    with open(saved_filepath, 'w', encoding='utf-8') as outfile:
        outfile.writelines('the attention weights for sentences in the news contents as well as comments \n')
        outfile.writelines('the test data counts: {}\n'.format(len(content_comment_weights)))
        for idx, (con_and_att, com_and_att) in tqdm(enumerate(content_comment_weights),
                                                    desc='saving the attention weights'):
            outfile.writelines('\n')
            outfile.writelines('sample: {} true label: {}  predict label: {}\n'.format(idx+1, y_true[idx], y_prob[idx]))
            outfile.writelines('sample: {} news contents:\n'.format(idx+1))
            # 新闻句子
            # 按注意力权重排序
            contents_dic = {}
            for (con, con_weight) in con_and_att:
                con_str = ' '.join(con)
                contents_dic[con_str] = con_weight
            contents_sorted = sorted(contents_dic.items(), key=lambda x: x[1], reverse=True)
            # 写入文件
            for i, (con, con_weight) in enumerate(contents_sorted):
                con_line = '{} {} {}'.format(i+1, con, con_weight)
                outfile.writelines(con_line + '\n')

            outfile.writelines('sample: {} user comments:\n'.format(idx+1))

            # 用户评论
            # 按注意力权重排序
            comments_dic = {}
            for (com, com_weight) in com_and_att:
                com_str = ' '.join(com)
                comments_dic[com_str] = com_weight
            comments_sorted = sorted(comments_dic.items(), key=lambda x: x[1], reverse=True)
            # 写入文件
            for j, (com, com_weight) in enumerate(comments_sorted):
                com_line = '{} {} {}'.format(j+1, com, com_weight)
                outfile.writelines(com_line + '\n')


if __name__ == "__main__":
    torch.manual_seed(1234)
    config_ = Config()
    dataset = FakeNewsDataset(config_)
    train_iter_, val_iter_, vocab_ = dataset.data_process(only_test=False)
    # print(len(train_data))
    # print(len(val_data))








