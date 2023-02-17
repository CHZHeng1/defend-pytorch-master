import os
import torch


class Config:
    """此类用于定义超参数"""
    def __init__(self):
        self.project_dir = os.getcwd()  # 获取当前脚本所在路径
        self.dataset_dir = os.path.join(self.project_dir, 'data')  # 数据集文件夹
        self.model_save_dir = os.path.join(self.project_dir, 'results')  # 模型存放文件夹
        self.attention_weight_filepath = os.path.join(self.model_save_dir, 'attention_weights.txt')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.platform = 'politifact'
        self.glove_embedding_filepath = os.path.join(self.dataset_dir, 'embeddings_data' + os.sep + 'glove.6B.100d.txt')
        self.validation_split = 0.25
        self.max_sentence_length = 120
        self.max_sentence_count = 50
        self.max_comment_length = 120
        self.max_comment_count = 50

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 10  # default 30
        self.batch_size = 20  # default 20
        self.hidden_dim = 100  # default 100
        self.k_dim = 80  # co-attention dim  default 80
        self.num_class = 2
        self.learning_rate = 0.001  # default 0.001


if __name__ == '__main__':
    config = Config()
    print(config.glove_embedding_filepath)
