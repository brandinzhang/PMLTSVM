import numpy as np


# -----------ranking based metrics---------------

def one_error(y_true, y_score):
    num_samples = y_true.shape[0]
    error_count = 0
    for i in range(num_samples):
        top_label_index = np.argmax(y_score[i])
        if y_true[i, top_label_index] != 1:
            error_count += 1
    return error_count / num_samples

def coverage(y_true, y_score):
    """
    进行了这么复杂的处理是因为birds数据集存在不少全0多标签样例
    """
    num_samples = y_true.shape[0]
    coverage_sum = 0
    wrong_samples = []
    valid_samples = 0  # 记录有效样本数(真实标签不全为0的样本)
    for i in range(num_samples):
        positive_indices = np.where(y_true[i] == 1)[0]
        # 跳过真实标签全为0的样本
        if len(positive_indices) == 0:
            # 打印位于第几行
            wrong_samples.append(i)
            continue
        # 计算覆盖误差
        ranks = np.argsort(-y_score[i])  # 得分从高到低排序的索引
        max_rank = max([np.where(ranks == idx)[0][0] for idx in positive_indices])
        coverage_sum += max_rank + 1
        valid_samples += 1
    # 避免除以0
    # print(f"wrong samples: {wrong_samples}")
    return coverage_sum / num_samples 

def ranking_loss(y_true, y_score):
    num_samples = y_true.shape[0]
    loss = 0
    for i in range(num_samples):
        pos = np.where(y_true[i] == 1)[0]
        neg = np.where(y_true[i] == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        cnt = 0
        for p in pos:
            for n in neg:
                if y_score[i][n] >= y_score[i][p]:
                    cnt += 1
        loss += cnt / (len(pos) * len(neg))
    return loss / num_samples


# 计算平均精度
def average_precision(y_true, y_score):
    num_samples = y_true.shape[0]
    aps = []
    
    for i in range(num_samples):
        true_labels = y_true[i]
        scores = y_score[i]
        
        # 获取正样本索引
        pos_indices = np.where(true_labels == 1)[0]
        if len(pos_indices) == 0:
            continue  # 跳过全负样本
            
        # 按得分降序排序
        ranked_indices = np.argsort(-scores)
        
        # 计算每个相关位置的精度
        precisions = []
        correct_count = 0
        
        for rank, idx in enumerate(ranked_indices):
            if idx in pos_indices:
                correct_count += 1
                precisions.append(correct_count / (rank + 1))
                
        # 单个样本的AP值
        ap = np.mean(precisions) if precisions else 0
        aps.append(ap)
        
    return np.mean(aps) if aps else 0.0

# -----------------------bipartite based metrics-------------------

# hamm_loss 函数
def hamming_loss(y_true, y_pred):
    num_samples = y_true.shape[0]
    loss = 0
    for i in range(num_samples):
        # 计算汉明损失
        loss += np.sum(y_true[i] != y_pred[i])
    return loss / (num_samples * y_true.shape[1])

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    修正后的Recall计算(样例级别宏平均)
    公式:R = (有效样本的TP之和) / (有效样本的TP+FN之和)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1)
    
    valid_mask = (tp + fn) > 0  # 过滤真实标签全0的样本
    if np.any(valid_mask):
        return np.sum(tp[valid_mask]) / np.sum(tp[valid_mask] + fn[valid_mask])
    return 0.0

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    修正后的Precision计算(样例级别宏平均)
    公式:P = (有效样本的TP之和) / (有效样本的TP+FP之和)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1)
    
    valid_mask = (tp + fp) > 0  # 过滤预测标签全0的样本
    if np.any(valid_mask):
        return np.sum(tp[valid_mask]) / np.sum(tp[valid_mask] + fp[valid_mask])
    return 0.0

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    修正后的F1-score计算(基于宏平均)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)