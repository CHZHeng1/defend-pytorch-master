import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttLayer(nn.Module):
    def __init__(self, attention_dim=100):
        """
        Attention layer used for the calcualting attention in word and sentence levels
        词和句子交互的注意力层
        :param attention_dim: 输入的隐含层向量维度
        """
        super(AttLayer, self).__init__()
        self.hidden_dim = attention_dim

        self.W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.b = nn.Parameter(torch.Tensor(self.hidden_dim, ))
        self.u = nn.Parameter(torch.Tensor(self.hidden_dim, 1))

        eps = torch.tensor(1e-7, dtype=torch.float)
        self.register_buffer('eps', eps)  # 注册为模型参数，但不参与训练

        self._resnet_parameters()

    def _resnet_parameters(self):
        """模型参数初始化"""
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, hidden_state, mask=None):
        """
        :param hidden_state: [batch_size, max_seq_len, hidden_dim]
        :param mask: [batch_size,]
        """
        assert len(hidden_state.shape) == 3
        # uit = tanh(xW+b)
        uit = torch.tanh(torch.matmul(hidden_state, self.W) + self.b)  # [batch_size, seq_len, hidden_dim]
        ait = torch.matmul(uit, self.u)  # [batch_size, seq_len, 1]
        ait = torch.squeeze(ait, -1)  # [batch_size, seq_len]
        ait = torch.exp(ait)

        if mask is not None:  # 注意力掩码，掩盖掉补齐后的单词序列信息
            ait *= mask.float()

        div_term = torch.sum(ait, dim=1, keepdim=True) + self.eps  # [batch_size, 1]
        ait = ait / div_term.float()  # [batch_size, seq_len] 这里得到的注意力权重表示词级别注意力权重，即源序列应更关注于哪些词

        ait = torch.unsqueeze(ait, -1)  # [batch_size, seq_len, 1]
        weighted_input = hidden_state * ait  # [batch_size, seq_len, hidden_dim]
        output = torch.sum(weighted_input, dim=1)  # [batch_size, hidden_dim]  vi

        return output


class CoAttention(nn.Module):
    def __init__(self, latent_dim=200, k=80):
        """
        Co-Attention 对应论文4.3
        """
        super(CoAttention, self).__init__()
        self.latent_dim = latent_dim
        self.k = k

        self.Wl = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))

        self.Wc = nn.Parameter(torch.Tensor(self.k, self.latent_dim))
        self.Ws = nn.Parameter(torch.Tensor(self.k, self.latent_dim))

        self.whs = nn.Parameter(torch.Tensor(1, self.k))
        self.whc = nn.Parameter(torch.Tensor(1, self.k))

        self._resnet_parameters()

    def _resnet_parameters(self):
        """模型参数初始化"""
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, comment_rep, sentence_rep, comment_mask=None, sentence_mask=None, predict=False):
        """
        :param comment_rep: 评论文本 word embedding -> Bi-GRU -> attention layer -> C
                            [batch_size, num_comments, hidden_dim]
        :param sentence_rep: 内容文本 word embedding -> Bi-GRU -> attention layer -> sentence embedding -> Bi-GRU -> S
                            [batch_size, num_sentences, hidden_dim] shape有待进一步探究
        :param comment_mask: 评论注意力掩码 [batch_size, max_comment_count]
        :param sentence_mask: 句子注意力掩码 [batch_size, max_sentence_count]
        :param predict: 是否打开预测模式
        return [batch_size, hidden_dim*2]
        """
        sentence_rep_trans = sentence_rep.transpose(2, 1)  # [batch_size, hidden_dim,num_sentences]
        comment_rep_trans = comment_rep.transpose(2, 1)  # [batch_size, hidden_dim, num_comments]

        # [batch_size, num_comments, hidden_dim] * [hidden_dim, hidden_dim] = [batch_size, num_comments, hidden_dim]
        # [batch_size, num_comments, hidden_dim] * [batch_size, hidden_dim, num_sentences] =
        #                                                                      [batch_size, num_comments, num_sentences]
        L = torch.tanh(torch.einsum('btd,dD,bDn->btn', comment_rep, self.Wl, sentence_rep_trans))
        L_trans = L.transpose(2, 1)  # [batch_size, num_sentences, num_comments]

        # [k, hidden_dim] * [batch_size, hidden_dim, num_sentences] = [batch_size, k, num_sentences]
        WsS = torch.einsum('kd,bdn->bkn', self.Ws, sentence_rep_trans)

        # [k, hidden_dim] * [batch_size, hidden_dim, num_comments] = [batch_size, k, num_comments]
        # [batch_size, k, num_comments] * [batch_size, num_comments, num_sentences] = [batch_size, k, num_sentences]
        WcCF = torch.einsum('kd,bdt,btn->bkn', self.Wc, comment_rep_trans, L)

        Hs = torch.tanh(WsS + WcCF)  # [batch_size, k, num_sentences]

        # [k, hidden_dim] * [batch_size, hidden_dim, num_comments] = [batch_size, k, num_comments]
        WcC = torch.einsum('kd,bdt->bkt', self.Wc, comment_rep_trans)

        # [k, hidden_dim] * [batch_size, hidden_dim, num_sentences] = [batch_size, k, num_sentences]
        # [batch_size, k, num_sentences] * [batch_size, num_sentences, num_comments] = [batch_size, k, num_comments]
        WsSF = torch.einsum('kd,bdn,bnt->bkt', self.Ws, sentence_rep_trans, L_trans)

        Hc = torch.tanh(WcC + WsSF)  # [batch_size, k, num_comments]

        # [1, k] * [batch_size, k, num_sentences] = [batch_size, 1, num_sentences] -> [batch_size, num_sentences]
        As = torch.einsum('yk,bkn->bn', self.whs, Hs)
        if sentence_mask is not None:
            As = As.masked_fill(sentence_mask, float('-inf'))

        As = F.softmax(As, dim=1)  # [batch_size, num_sentences]

        # [1, k] * [batch_size, k, num_comments] = [batch_size, 1, num_comments] -> [batch_size, num_comments]
        Ac = torch.einsum('yk,bkt->bt', self.whc, Hc)
        if comment_mask is not None:
            Ac = Ac.masked_fill(comment_mask, float('-inf'))
        Ac = F.softmax(Ac, dim=1)  # [batch_size, num_comments]

        # [batch_size, hidden_dim, num_sentences] * [batch_size, num_sentences]
        # [batch_size, hidden_dim, num_sentences] * [batch_size, num_sentences, 1] = [batch_size, hidden_dim]
        co_s = torch.einsum('bdn,bn->bd', sentence_rep_trans, As)
        # [batch_size, hidden_dim, num_comments] * [batch_size, num_comments, 1] = [batch_size, hidden_dim]
        co_c = torch.einsum('bdt,bt->bd', comment_rep_trans, Ac)
        co_sc = torch.concat([co_s, co_c], dim=1)  # [batch_size, hidden_dim*2]

        if predict:
            return co_sc, As, Ac
        else:
            return co_sc


class WordEncoder(nn.Module):
    def __init__(self, embedding_dim=100, words_hidden_dim=100, batch_first=True):
        super(WordEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.words_hidden_dim = words_hidden_dim
        self.batch_first = batch_first
        self.bi_gru_hidden_dim = self.words_hidden_dim * 2

        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.words_hidden_dim,
                          batch_first=self.batch_first, bidirectional=True)
        self.words_attention_layer = AttLayer(attention_dim=self.bi_gru_hidden_dim)

        # self._resnet_parameters()

    def _resnet_parameters(self):
        """模型参数初始化"""
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, word_embedding, sentence_lengths=None):
        """
        :param word_embedding: [batch_size, max_sentence_len, embedding_dim]
        :param sentence_lengths: [batch_size, max_sentence_len]
        """
        if sentence_lengths is not None:
            contents_pack = pack_padded_sequence(word_embedding, sentence_lengths.cpu(), batch_first=self.batch_first,
                                                 enforce_sorted=False)
            hidden_state, _ = self.gru(contents_pack)
            hidden_state, _ = pad_packed_sequence(hidden_state, batch_first=self.batch_first)
            attention_mask = length_to_mask(sentence_lengths)
        else:
            hidden_state, _ = self.gru(word_embedding)
            attention_mask = None
        # [batch_size, bi_gru_hidden_dim]
        words_attention_outputs = self.words_attention_layer(hidden_state, mask=attention_mask)
        return words_attention_outputs


def length_to_mask(lengths):
    """
    将序列的长度转换成mask矩阵，忽略序列补齐后padding部分的信息
    :param lengths: [batch,]
    :return: batch * max_len
    """
    max_len = torch.max(lengths).long()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask


class SentenceEncoder(nn.Module):
    def __init__(self, words_attention_dim=200, words_hidden_dim=100, batch_first=True):
        super(SentenceEncoder, self).__init__()
        self.words_attention_dim = words_attention_dim
        self.words_hidden_dim = words_hidden_dim
        self.batch_first = batch_first

        self.gru = nn.GRU(input_size=self.words_attention_dim, hidden_size=self.words_hidden_dim,
                          batch_first=self.batch_first, bidirectional=True)

        # self._resnet_parameters()

    def _resnet_parameters(self):
        """模型参数初始化"""
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, hidden_state, sentence_counts=None):
        """
        :param hidden_state: [batch_size, max_sentence_count, hidden_dim]
        :param sentence_counts: [batch_size,]
        """
        if sentence_counts is not None:
            sentences_pack = pack_padded_sequence(hidden_state, sentence_counts.cpu(), batch_first=self.batch_first,
                                                  enforce_sorted=False)
            sentence_encoder_outputs, _ = self.gru(sentences_pack)  # [batch_size, num_sentences, hidden_dim*2]
            sentence_encoder_outputs, _ = pad_packed_sequence(sentence_encoder_outputs, batch_first=self.batch_first)
        else:
            sentence_encoder_outputs, _ = self.gru(hidden_state)

        return sentence_encoder_outputs


class TokenEmbedding(nn.Module):
    def __init__(self, vocab, pt_vocab, pt_embeddings):
        """
        :param vocab: 基于数据集构建的词典
        :param pt_vocab: 基于预训练词向量构建的词典
        :param pt_embeddings: 预训练词向量 [num_words, embedding_dim]
        """
        super(TokenEmbedding, self).__init__()
        embedding_dim = pt_embeddings.shape[1]
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.normal_(0, 0.1)
        # nn.init.xavier_uniform_(self.embeddings.weight.data)
        # 使用预训练词向量对词向量层进行初始化
        for idx, token in enumerate(vocab.idx_to_token):
            pt_idx = pt_vocab[token]
            # 只初始化预训练词典中存在的词，对于未出现在预训练词典中的词，保留其随机初始化向量
            if pt_idx != pt_vocab.unk:
                self.embeddings.weight[idx].data.copy_(pt_embeddings[pt_idx])

    def forward(self, tokens):
        return self.embeddings(tokens)


class Defend(nn.Module):
    def __init__(self, vocab=None, pt_vocab=None, pt_embeddings=None, hidden_dim=100, co_attention_weight_dim=80,
                 num_classes=2, batch_first=True):
        super(Defend, self).__init__()

        self.embedding_dim = pt_embeddings.shape[1]
        self.words_hidden_dim = hidden_dim
        self.sentences_input_dim = self.words_hidden_dim * 2
        self.co_attention_weight_dim = co_attention_weight_dim
        self.fc_layer_input_dim = self.sentences_input_dim * 2
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.activation = torch.tanh

        # word embedding layer
        self.embedding_layer = TokenEmbedding(vocab, pt_vocab, pt_embeddings)

        # word encoder layer
        self.words_encoder_layer = WordEncoder(embedding_dim=self.embedding_dim, words_hidden_dim=self.words_hidden_dim,
                                               batch_first=self.batch_first)

        # sentence encoder layer
        self.sentences_encoder_layer = SentenceEncoder(words_attention_dim=self.sentences_input_dim,
                                                       words_hidden_dim=self.words_hidden_dim,
                                                       batch_first=self.batch_first)
        # co-attention layer
        self.co_attention_layer = CoAttention(latent_dim=self.sentences_input_dim, k=self.co_attention_weight_dim)

        # full connection layer
        # self.fc_layer = nn.Linear(self.fc_layer_input_dim, self.fc_layer_input_dim)

        # layer norm layer
        # self.ln_layer = nn.LayerNorm(self.fc_layer_input_dim)

        # classifier layer
        self.classifier_layer = nn.Linear(self.fc_layer_input_dim, self.num_classes)

        # self._resnet_parameters()

    def _resnet_parameters(self):
        """模型参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, contents=None, content_lengths=None, content_counts=None, comments=None,
                comment_lengths=None, comment_counts=None, predict=False):
        """
        :param contents: tensor [batch_size, max_sentence_count, max_sentence_length]
        :param content_lengths: list len(list)=batch_size, len(list[0])=sentence_count, list[0]=sentence_lengths->tensor
        :param content_counts: tensor [batch_size, ]
        :param comments: tensor [batch_size, max_comment_count, max_comment_length]
        :param comment_lengths: list len(list)=batch_size, len(list[0])=sentence_count, list[0]=sentence_lengths->tensor
        :param comment_counts: tensor [batch_size, ]
        :param predict: True or False
        """
        # [batch_size, max_sentence_count, max_sentence_length, embedding_dim]
        content_embeddings = self.embedding_layer(contents)
        # [batch_size, max_comment_count, max_comment_length, embedding_dim]
        comment_embeddings = self.embedding_layer(comments)

        batch_size, max_sentence_count = content_embeddings.shape[0], content_embeddings.shape[1]
        bsz, max_comment_count = comment_embeddings.shape[0], comment_embeddings.shape[1]

        # [batch_size, max_sentence_count, hidden_dim]
        word_outputs = torch.zeros(batch_size, max_sentence_count, self.sentences_input_dim,
                                   device=content_embeddings.device, requires_grad=False)
        # [batch_size, max_comment_count, hidden_dim]
        comment_outputs = torch.zeros(bsz, max_comment_count, self.sentences_input_dim,
                                      device=comment_embeddings.device, requires_grad=False)

        for idx in range(len(content_embeddings)):
            if content_lengths is not None:
                # [sentence_counts, max_sentence_length, embedding_dim]
                word_encoder_inputs = content_embeddings[idx][:content_lengths[idx].size(0)]
                # [sentence_counts, hidden_dim]
                word_encoder_outputs = self.words_encoder_layer(word_encoder_inputs,
                                                                sentence_lengths=content_lengths[idx])
            else:
                word_encoder_inputs = content_embeddings[idx]
                word_encoder_outputs = self.words_encoder_layer(word_encoder_inputs, sentence_lengths=None)

            word_outputs[idx][:len(word_encoder_outputs)] = word_encoder_outputs.detach()

            if comment_lengths is not None:
                # [comment_counts, max_comment_length, embedding_dim]
                comment_encoder_inputs = comment_embeddings[idx][:comment_lengths[idx].size(0)]
                # [comments_counts, hidden_dim]
                comment_encoder_outputs = self.words_encoder_layer(comment_encoder_inputs,
                                                                   sentence_lengths=comment_lengths[idx])
            else:
                comment_encoder_inputs = comment_embeddings[idx]
                comment_encoder_outputs = self.words_encoder_layer(comment_encoder_inputs, sentence_lengths=None)

            comment_outputs[idx][:len(comment_encoder_outputs)] = comment_encoder_outputs.detach()

        word_outputs.requires_grad = True
        comment_outputs.requires_grad = True

        if content_counts is not None:
            # [batch_size, max_sentence_count, hidden_dim]
            sentence_outputs = self.sentences_encoder_layer(word_outputs, sentence_counts=content_counts)
        else:
            sentence_outputs = self.sentences_encoder_layer(word_outputs, sentence_counts=None)

        sen_att_prob, comm_att_prob = None, None
        if content_counts is not None and comment_counts is not None:
            content_mask = length_to_mask(content_counts)==False  # [batch_size, max_sentence_count]
            comment_mask = length_to_mask(comment_counts)==False  # [batch_size, max_comment_count]
            if predict:
                # [batch_size, hidden_dim*2]
                co_attention_outputs, sen_att_prob, comm_att_prob = \
                    self.co_attention_layer(comment_outputs, sentence_outputs, comment_mask=comment_mask,
                                            sentence_mask=content_mask, predict=predict)
            else:
                co_attention_outputs = \
                    self.co_attention_layer(comment_outputs, sentence_outputs, comment_mask=comment_mask,
                                            sentence_mask=content_mask, predict=predict)

        else:
            if predict:
                co_attention_outputs, sen_att_prob, comm_att_prob = \
                    self.co_attention_layer(comment_outputs, sentence_outputs, comment_mask=None,
                                            sentence_mask=None, predict=predict)
            else:
                co_attention_outputs = \
                    self.co_attention_layer(comment_outputs, sentence_outputs, comment_mask=None,
                                            sentence_mask=None, predict=predict)

        # outputs = self.fc_layer(co_attention_outputs)
        # outputs = self.activation(self.ln_layer(outputs))
        outputs = self.classifier_layer(co_attention_outputs)

        if predict:
            return outputs, sen_att_prob, comm_att_prob
        else:
            return outputs


if __name__ == '__main__':
    torch.manual_seed(123)
    # attention_model = AttLayer(attention_dim=256)
    # co_attention_model = CoAttention(latent_dim=256, k=80)
    # inputs = torch.randn(16, 20, 256)
    # outputs = attention_model(inputs)
    # print(outputs.shape)
    # print(outputs)

    # content_inputs = torch.randn(16, 10, 256)  # [batch_size, num_sentences, hidden_dim]
    # comment_inputs = torch.randn(16, 15, 256)  # [batch_size, num_comments, hidden_dim]
    # co_ = co_attention_model(comment_inputs, content_inputs)

    # print(co_.shape)
    # print(co_)
