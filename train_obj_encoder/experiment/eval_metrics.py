import torch
from sklearn.metrics import balanced_accuracy_score

# -------------------------------------------------
# 5. 평가 함수
# -------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, topk=(1, 5, 10), device="cuda"):
    model.eval()
    correct_topk = torch.zeros(len(topk), dtype=torch.long)
    total = 0
    all_pred = []
    all_true = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += yb.size(0)

        # --- Top‑k 정확도 계산 ---
        #   torch.topk: (B, max_k)
        max_k = max(topk)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # index
        pred = pred.t()                     # (max_k, B)
        correct = pred.eq(yb.view(1, -1))   # (max_k, B) bool

        for i, k in enumerate(topk):
            correct_k = correct[:k].any(dim=0).sum()   # k개 중 하나라도 맞으면 1
            correct_topk[i] += correct_k.cpu()

        all_pred.append(pred[0].cpu())  # top‑1 예측
        all_true.append(yb.cpu())

    topk_acc = (correct_topk.float() / total).tolist()

    # --- Mean Class Accuracy (balanced_accuracy) ---
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    mean_acc = balanced_accuracy_score(y_true, y_pred)

    return topk_acc, mean_acc


@torch.no_grad()
def evaluate_topk_mean_acc(model, loader, num_classes: int, topk=(1, 5, 10), device="cuda"):
    """
    Top‑k Overall Accuracy + Top‑k Mean‑Class Accuracy(= balanced accuracy) 계산.

    Parameters
    ----------
    model        : nn.Module
    loader       : DataLoader  (test set)
    num_classes  : int         (예: 160)
    topk         : tuple[int]  (k 값 목록)

    Returns
    -------
    overall_topk : dict{k: acc}      # 전체 정확도
    mean_topk    : dict{k: acc_mean} # 클래스별 recall 평균
    """
    device = next(model.parameters()).device
    max_k  = max(topk)
    k_idx  = {k: i for i, k in enumerate(topk)}

    # 누적용 변수
    total_samples   = 0
    correct_count   = torch.zeros(len(topk), dtype=torch.long).to(device)
    class_total     = torch.zeros(num_classes, dtype=torch.long).to(device)
    class_correct_k = torch.zeros(len(topk), num_classes, dtype=torch.long).to(device)

    model.eval()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)

        # ---------- Top‑k 예측 ----------
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # (B, max_k)
        pred = pred.t()                          # (max_k, B)
        correct = pred.eq(yb.view(1, -1))        # (max_k, B) bool

        batch_size = yb.size(0)
        total_samples += batch_size
        class_total.index_add_(0, yb, torch.ones_like(yb, dtype=torch.long).to(device))

        # k별 집계
        for i, k in enumerate(topk):
            # ① 전체 정확도
            correct_k = correct[:k].any(dim=0).sum()  # Top‑k 안에 있으면 1
            correct_count[i] += correct_k.cpu()

            # ② 클래스별 correct
            mask = correct[:k].any(dim=0)             # (B,) bool
            if mask.any():
                class_correct_k[i].index_add_(
                    0, yb[mask], torch.ones(mask.sum(), dtype=torch.long).to(device)
                )

    # ----- 결과 정리 -----
    overall_topk = {k: (correct_count[k_idx[k]].item() / total_samples)
                    for k in topk}

    mean_topk = {}
    valid = class_total > 0                          # 데이터가 있는 클래스만
    for i, k in enumerate(topk):
        per_class_recall = torch.zeros(num_classes, dtype=torch.float).to(device)
        per_class_recall[valid] = (
            class_correct_k[i, valid].float() / class_total[valid].float()
        )
        mean_topk[k] = per_class_recall[valid].mean().item()

    # classwise accuracy 계산
    classwise_topk = {}

    for i, k in enumerate(topk):
        acc = torch.zeros(num_classes, dtype=torch.float).to(device)
        acc[valid] = class_correct_k[i, valid].float() / class_total[valid].float()
        classwise_topk[k] = acc.cpu()
    
    return overall_topk, mean_topk, classwise_topk
