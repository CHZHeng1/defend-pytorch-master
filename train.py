import os
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import lr_scheduler

from data_utils import FakeNewsDataset, activation_maps
from glove import load_glove_embedding
from defend import Defend
from metrics import cal_precision, cal_recall, cal_f1
from config import Config

import warnings
warnings.filterwarnings('ignore', message='The input looks more like a filename than markup.*')


def train(config):
    data_loader = FakeNewsDataset(config)
    model_save_path = os.path.join(config.model_save_dir, 'Defend_model.pt')
    train_iter, val_iter, vocab = data_loader.data_process(only_test=False)
    glove_vocab, glove_embeddings = load_glove_embedding(config.glove_embedding_filepath)
    model = Defend(vocab=vocab, pt_vocab=glove_vocab, pt_embeddings=glove_embeddings, hidden_dim=config.hidden_dim,
                   co_attention_weight_dim=config.k_dim, num_classes=config.num_class)

    model.to(config.device)
    criterion = nn.CrossEntropyLoss()

    # lr_lambda = lambda epoch: 0.85 ** epoch
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()

    max_f = 0
    for epoch in range(config.epochs):
        total_loss = 0
        total_acc = 0
        labels = []
        for batch in tqdm(train_iter, desc=f'Training Epoch {epoch + 1}'):
            contents, content_lengths, content_counts, comments, comment_lengths, comment_counts, targets = batch
            contents, content_counts, comments, comment_counts, targets = \
                [x.to(config.device) for x in [contents, content_counts, comments, comment_counts, targets]]
            content_lengths = [x.to(config.device) for x in content_lengths]
            comment_lengths = [x.to(config.device) for x in comment_lengths]

            outputs = model(contents, content_lengths, content_counts, comments, comment_lengths, comment_counts)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (outputs.argmax(dim=1) == targets).sum().item()
            total_acc += acc
            total_loss += loss.item()

            batch_labels = targets.tolist()
            labels.extend(batch_labels)

        # scheduler.step()
        print(f'Train Loss:{total_loss:.4f}    Train Accuracy:{total_acc / len(labels):.4f}')
        val_loss, acc, p, r, f, _, _ = evaluate(model, criterion, val_iter, config.device)
        print(f'Val Loss:{val_loss:.4f}    Val Accuracy:{acc:.4f}')
        print(f"Val Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate(model, criterion, data_iter, device, inference=False):
    test_acc = 0
    test_loss = 0
    model.eval()  # 切换到测试模式
    with torch.no_grad():  # 不计算梯度
        y_true, y_prob = [], []
        sentences_prob, comments_prob = [], []
        for idx, batch in enumerate(data_iter):
            contents, content_lengths, content_counts, comments, comment_lengths, comment_counts, targets = batch
            contents, content_counts, comments, comment_counts, targets = \
                [x.to(device) for x in [contents, content_counts, comments, comment_counts, targets]]
            content_lengths = [x.to(device) for x in content_lengths]
            comment_lengths = [x.to(device) for x in comment_lengths]

            if inference:
                outputs, sen_att_prob, comm_att_prob = \
                    model(contents, content_lengths, content_counts, comments, comment_lengths, comment_counts,
                          predict=inference)
                sentences_prob.extend(sen_att_prob)
                comments_prob.extend(comm_att_prob)

            else:
                outputs = model(contents, content_lengths, content_counts, comments, comment_lengths, comment_counts,
                                predict=inference)

            loss = criterion(outputs, targets)
            acc = (outputs.argmax(dim=1) == targets).sum().item()
            test_acc += acc
            test_loss += loss.item()

            batch_prob = outputs.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = targets.tolist()
            y_prob.extend(batch_prob)
            y_true.extend(batch_true)

        acc = test_acc / len(y_true)
        p = cal_precision(y_true, y_prob)
        r = cal_recall(y_true, y_prob)
        f = cal_f1(y_true, y_prob)

    model.train()
    if inference:
        return test_loss, acc, p, r, f, y_true, y_prob, sentences_prob, comments_prob
    else:
        return test_loss, acc, p, r, f, y_true, y_prob


def predict(config, inference=False):
    data_loader = FakeNewsDataset(config)
    test_iter, vocab, test_data_map = data_loader.data_process(only_test=True)
    glove_vocab, glove_embeddings = load_glove_embedding(config.glove_embedding_filepath)
    model = Defend(vocab=vocab, pt_vocab=glove_vocab, pt_embeddings=glove_embeddings, hidden_dim=config.hidden_dim,
                   co_attention_weight_dim=config.k_dim, num_classes=config.num_class)

    model_save_path = os.path.join(config.model_save_dir, 'Defend_model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        print('成功载入已有模型，进行预测......')

    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    if inference:
        test_loss, acc, p, r, f, y_true, y_prob, sentences_prob, comments_prob = \
            evaluate(model, criterion, test_iter, config.device, inference=inference)
        print(f'Test Loss:{test_loss:.4f}    Test Accuracy:{acc:.4f}')
        print(f"Test Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")

        activation_maps(test_data_map, sentences_prob, comments_prob, y_true, y_prob, config.attention_weight_filepath)
        print(f'the attention weights were saved.')

    else:
        test_loss, acc, p, r, f, _, _ = evaluate(model, criterion, test_iter, config.device, inference=inference)
        print(f'Test Loss:{test_loss:.4f}    Test Accuracy:{acc:.4f}')
        print(f"Test Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


if __name__ == '__main__':
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model_config = Config()
    # train(model_config)
    predict(model_config, inference=True)











