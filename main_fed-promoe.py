import os
import copy
import random
from typing import List, Tuple, Dict
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_JIT"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.optim import Adam, SGD

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils.options import args_parser
from utils.network_ShuffleNetV2 import ShuffleNetV2

try:
    from utils.triplet import TripletLoss
except ImportError:
    print("[Error] 找不到 utils.triplet 模块，请确保文件存在！")


    class TripletLoss(nn.Module):
        def __init__(self, margin): super().__init__()

        def forward(self, x, y): return torch.tensor(0.0), 0

warnings.filterwarnings("ignore", category=FutureWarning)


def set_random_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu: int):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    else:
        return torch.device("cpu")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = os.path.join(save_dir, "plots")
        ensure_dir(self.save_dir)

    def plot_confusion_matrix(self, y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[Visual] Saved Confusion Matrix: {save_path}")

    def plot_bar_comparison(self, metrics: Dict[str, float], filename):
        names = list(metrics.keys())
        values = list(metrics.values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylim(0, 100)
        plt.title('Performance Comparison')
        plt.ylabel('Accuracy (%)')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_hybrid_mechanism(self, total, local_count, local_acc, cloud_count, cloud_acc, filename):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        labels = [f'Local\n({local_count})', f'Cloud\n({cloud_count})']
        sizes = [local_count, cloud_count]
        colors = ['#ff9999', '#66b3ff']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
        ax1.set_title(f'Traffic Distribution (Total: {total})')

        overall_acc = 0
        if total > 0:
            overall_acc = (local_count * local_acc + cloud_count * cloud_acc) / total

        paths = ['Local Path', 'Cloud Path', 'Overall']
        accs = [local_acc, cloud_acc, overall_acc]
        bars = ax2.bar(paths, accs, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_ylim(0, 100)
        ax2.set_title('Accuracy by Path')
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_tsne(self, features, labels, filename="tsne_features.png"):
        print(f"[Visual] Running T-SNE on {len(features)} samples... (This may take a moment)")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(12, 10))
        unique_labels = np.unique(labels)
        colors = sns.color_palette("hls", len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        color=colors[i], label=f'Class {label}',
                        alpha=0.6, s=30, edgecolors='w', linewidth=0.5)

        plt.title('T-SNE Visualization of Learned Features (Cloud Gate Only)')
        plt.legend(loc='best', title="Classes")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[Visual] Saved T-SNE plot: {save_path}")


class SignalLocalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.signals[index], self.labels[index]


def build_fed_datasets(args) -> Tuple[List[SignalLocalDataset], SignalLocalDataset]:
    num_clients = args.num_locs
    dis_list = args.dis_list
    data_root = args.data_root

    local_datasets: List[SignalLocalDataset] = []

    for cid in range(num_clients):
        dev_id = cid + 1
        xs_train, ys_train = [], []

        for dis in dis_list:
            train_dir = os.path.join(data_root, args.train_dir, f"{dis}_STFT")
            xtr_path = os.path.join(train_dir, f"device{dev_id}_x_train.npy")
            ytr_path = os.path.join(train_dir, f"device{dev_id}_y_train.npy")

            if os.path.exists(xtr_path) and os.path.exists(ytr_path):
                xs_train.append(np.load(xtr_path))
                ys_train.append(np.load(ytr_path))

        if len(xs_train) == 0:
            print(f"[Warn] Client {cid} (device{dev_id}) 在 dis_list={dis_list} 下没有找到 *训练* 数据文件")
            x_train = np.empty((0, 1, 244, 244), dtype=np.float32)
            y_train = np.empty((0,), dtype=int)
        else:
            x_train = np.concatenate(xs_train, axis=0)
            y_train = np.concatenate(ys_train, axis=0).reshape(-1)

        local_datasets.append(SignalLocalDataset(x_train, y_train))
        print(f"[Data] client={cid} (device{dev_id}): train={len(y_train)}, labels={np.unique(y_train)}")

    xs_test, ys_test = [], []
    for dis in dis_list:
        test_dir = os.path.join(data_root, args.test_dir, f"{dis}_STFT")
        xte_path = os.path.join(test_dir, "global_x_test.npy")
        yte_path = os.path.join(test_dir, "global_y_test.npy")

        if os.path.exists(xte_path) and os.path.exists(yte_path):
            xs_test.append(np.load(xte_path))
            ys_test.append(np.load(yte_path))

    if len(xs_test) == 0:
        raise RuntimeError(f"[Data] 在 dis_list={dis_list} 下没有找到 global_x_test.npy")

    x_test = np.concatenate(xs_test, axis=0)
    y_test = np.concatenate(ys_test, axis=0).reshape(-1)
    global_test_dataset = SignalLocalDataset(x_test, y_test)

    print(f"[Data] Global Test Set: test={len(y_test)}, labels={np.unique(y_test)}")

    return local_datasets, global_test_dataset


class ExpertNet(ShuffleNetV2):
    def __init__(self, n_class: int):
        super().__init__(n_class=n_class)


class GateNet(ShuffleNetV2):
    def __init__(self, num_experts: int):
        super().__init__(n_class=num_experts)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    def collate_fn(batch):
        xs, ys = zip(*batch)
        xs = torch.from_numpy(np.stack(xs, axis=0))
        ys = torch.tensor(ys, dtype=torch.long)
        return xs, ys

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)


def train_single_expert(expert: nn.Module, loader: DataLoader, device: torch.device, epochs: int,
                        lr: float, momentum: float, weight_decay: float, log_prefix: str = ""):
    expert.to(device)
    expert.train()
    opt = SGD(expert.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for ep in range(epochs):
        correct, total, running_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = expert(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        print(f"{log_prefix} Epoch {ep + 1}/{epochs} | Loss {avg_loss:.4f} | Acc {acc:.2f}% ({correct}/{total})")


def build_labeled_proxy(local_datasets: List[SignalLocalDataset], proxy_per_client: int,
                        device: torch.device, num_classes: int,
                        mixup_alpha: float) -> SignalLocalDataset:
    xs, ys = [], []
    print(f"[Phase 0] 开始构建原型代理集 (Feature-based), 每客户端提炼 {proxy_per_client} 个真实样本...")
    feature_extractor = ExpertNet(n_class=num_classes).to(device)
    feature_extractor.eval()

    for cid, ds in enumerate(local_datasets):
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        all_features = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _, feats = feature_extractor(x, return_features=True)
                all_features.append(feats.cpu().numpy())
        if len(all_features) == 0: continue
        flat_features = np.concatenate(all_features, axis=0)
        N = flat_features.shape[0]
        raw_signals = ds.signals;
        raw_labels = ds.labels

        if N <= proxy_per_client:
            print(f"  Client {cid}: 样本数 {N} <= {proxy_per_client}, 全选。", end="")
            raw_x, raw_y = raw_signals, raw_labels
        else:
            print(f"  Client {cid}: 提取特征 -> K-Means...", end="", flush=True)
            kmeans = KMeans(n_clusters=proxy_per_client, n_init=3, random_state=2025)
            kmeans.fit(flat_features)
            centers = kmeans.cluster_centers_
            selected_indices = []
            for center in centers:
                dists = np.linalg.norm(flat_features - center, axis=1)
                nearest_idx = np.argmin(dists)
                selected_indices.append(nearest_idx)
            selected_indices = list(set(selected_indices))
            while len(selected_indices) < proxy_per_client:
                remain = list(set(range(N)) - set(selected_indices))
                if not remain: break
                selected_indices.append(random.choice(remain))
            selected_indices = np.array(selected_indices)
            raw_x, raw_y = raw_signals[selected_indices], raw_labels[selected_indices]
            print(f"  完成.", end="")

        if mixup_alpha > 0:
            tx = torch.from_numpy(raw_x)
            perm_indices = torch.randperm(len(raw_x))
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)
            mixed_x = lam * tx + (1 - lam) * tx[perm_indices]
            selected_x = mixed_x.numpy()
            unique_labels, counts = np.unique(raw_y, return_counts=True)
            major_label = unique_labels[np.argmax(counts)]
            print(f" -> Privacy Mixup (alpha={mixup_alpha:.1f}, Label={major_label})")
        else:
            selected_x = raw_x
            unique_labels, counts = np.unique(raw_y, return_counts=True)
            major_label = unique_labels[np.argmax(counts)]
            print(f" -> No Mixup (Label={major_label})")

        xs.append(selected_x);
        ys.append(raw_y)

    del feature_extractor;
    torch.cuda.empty_cache()
    if not xs: raise RuntimeError("[Phase 0] 无法构建代理数据集！")
    xs = np.concatenate(xs, axis=0);
    ys = np.concatenate(ys, axis=0).reshape(-1)
    print(f"[Phase 0] 代理集构建完成: Total={len(ys)}, Shape={xs.shape}")
    return SignalLocalDataset(xs, ys)


def cloud_neg_finetune_experts(experts: nn.ModuleList, proxy_dataset: SignalLocalDataset, device: torch.device,
                               epochs: int, batch_size: int, lr: float, neg_per_expert_max: int,
                               mixup_alpha: float = 0.0, triplet_margin: float = 1.0, triplet_weight: float = 1.0):
    xs = proxy_dataset.signals;
    ys = proxy_dataset.labels.astype(int)
    if len(ys) == 0: return
    num_classes = max(int(ys.max()) + 1, len(experts))
    idxs_per_class = [np.where(ys == c)[0] for c in range(num_classes)]
    print(f"[Cloud Neg] Config: Mixup={mixup_alpha > 0}, Triplet (w={triplet_weight}), Best Model=True")
    triplet_criterion = TripletLoss(margin=triplet_margin)

    for c in range(num_classes):
        if c >= len(experts): continue
        exp = experts[c].to(device);
        exp.train()
        pos_idx = idxs_per_class[c]
        if pos_idx.size == 0:
            print(f"[Cloud Neg] Expert[{c}] 在 proxy 中没有正样本，跳过。")
            continue
        x_pos, y_pos = xs[pos_idx], ys[pos_idx]
        n_pos = len(y_pos)

        if num_classes > 1:
            neg_idx_all_list = [idxs_per_class[k] for k in range(num_classes) if k != c and idxs_per_class[k].size > 0]
            if not neg_idx_all_list:
                neg_idx_all = np.array([], dtype=int)
            else:
                neg_idx_all = np.concatenate(neg_idx_all_list, axis=0)
        else:
            neg_idx_all = np.array([], dtype=int)

        if neg_idx_all.size == 0:
            x_train, y_train = x_pos, y_pos
        else:
            n_neg = min(neg_per_expert_max, neg_idx_all.size)
            sel = np.random.choice(neg_idx_all.size, size=n_neg, replace=False)
            neg_idx = neg_idx_all[sel]
            x_neg, y_neg = xs[neg_idx], ys[neg_idx]
            x_train = np.concatenate([x_pos, x_neg], axis=0)
            y_train = np.concatenate([y_pos, y_neg], axis=0)

        ds_c = SignalLocalDataset(x_train, y_train)
        loader = make_loader(ds_c, batch_size=batch_size, shuffle=True)
        print(f"[Cloud Neg] Expert[{c}] Fine-tune: total={len(y_train)} (Pos={n_pos})")

        opt = Adam(exp.parameters(), lr=lr)
        best_acc = 0.0;
        best_model_wts = copy.deepcopy(exp.state_dict())

        for ep in range(epochs):
            correct, total, running_loss, running_triplet = 0, 0, 0.0, 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()

                logits_clean, features_clean = exp(xb, return_features=True)
                loss_trip, _ = triplet_criterion(features_clean, yb)

                if mixup_alpha > 0.0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    batch_size_curr = xb.size(0)
                    index = torch.randperm(batch_size_curr).to(device)
                    mixed_x = lam * xb + (1 - lam) * xb[index, :]
                    y_a, y_b = yb, yb[index]
                    logits_mix, _ = exp(mixed_x, return_features=True)
                    loss_ce = lam * F.cross_entropy(logits_mix, y_a) + (1 - lam) * F.cross_entropy(logits_mix, y_b)
                    pred = logits_clean.argmax(dim=1)
                else:
                    loss_ce = F.cross_entropy(logits_clean, yb)
                    pred = logits_clean.argmax(dim=1)

                loss = loss_ce + triplet_weight * loss_trip
                loss.backward();
                opt.step()

                running_loss += loss.item() * yb.size(0)
                running_triplet += loss_trip.item() * yb.size(0)
                correct += (pred == yb).sum().item();
                total += yb.size(0)

            epoch_acc = 100.0 * correct / total
            if epoch_acc > best_acc:
                best_acc = epoch_acc;
                best_model_wts = copy.deepcopy(exp.state_dict())

            if (ep + 1) % 20 == 0:
                avg_loss = running_loss / total
                avg_trip = running_triplet / total
                print(
                    f"  Expert[{c}] Ep {ep + 1}/{epochs} | Loss {avg_loss:.4f} (Trip {avg_trip:.4f}) | Acc {epoch_acc:.2f}% (Best: {best_acc:.2f}%)")

        exp.load_state_dict(best_model_wts)
        print(f"  Expert[{c}] Finished. Loaded Best Acc: {best_acc:.2f}%")


def kld_loss_T(logits_student, logits_teacher, T=2.0):
    p_student = F.log_softmax(logits_student / T, dim=1)
    p_teacher = F.softmax(logits_teacher / T, dim=1)
    loss = F.kl_div(p_student, p_teacher.detach(), reduction='batchmean') * (T * T)
    return loss


def cloud_mutual_distill_experts(experts: nn.ModuleList, proxy_dataset: SignalLocalDataset, device: torch.device,
                                 epochs: int, batch_size: int, lr: float, alpha_kd: float = 0.1, T: float = 2.0):
    if epochs <= 0: return
    if len(proxy_dataset) == 0: return
    num_experts = len(experts)
    for e in experts: e.to(device); e.train()
    loader = make_loader(proxy_dataset, batch_size=batch_size, shuffle=True)
    params = [p for e in experts for p in e.parameters()]
    opt = Adam(params, lr=lr)
    total_samples = len(proxy_dataset)

    print(f"[Mutual KD] Start (Anchor CE + Soft KD): experts={num_experts}, ep={epochs}, alpha={alpha_kd}, T={T}")

    for ep in range(epochs):
        epoch_loss_ce = np.zeros(num_experts);
        epoch_loss_kld = np.zeros(num_experts);
        epoch_correct = np.zeros(num_experts)
        for x, y in loader:
            x = x.to(device);
            y = y.to(device);
            batch_size = x.size(0)
            all_logits = [e(x) for e in experts]
            all_logits_sum = torch.zeros_like(all_logits[0])
            for logits in all_logits: all_logits_sum += logits
            batch_loss_ce_sum = 0.0;
            batch_loss_kld_sum = 0.0
            for i in range(num_experts):
                logits_student = all_logits[i]
                L_CE = F.cross_entropy(logits_student, y)
                batch_loss_ce_sum += L_CE
                logits_teacher_sum = all_logits_sum - logits_student
                logits_teacher_avg = logits_teacher_sum / (num_experts - 1)
                L_KLD = kld_loss_T(logits_student, logits_teacher_avg.detach(), T)
                batch_loss_kld_sum += L_KLD
                epoch_loss_ce[i] += L_CE.item() * batch_size;
                epoch_loss_kld[i] += L_KLD.item() * batch_size
                pred = logits_student.argmax(dim=1);
                epoch_correct[i] += (pred == y).sum().item()
            avg_batch_ce = batch_loss_ce_sum / num_experts
            avg_batch_kd = batch_loss_kld_sum / num_experts
            loss = (1.0 - alpha_kd) * avg_batch_ce + alpha_kd * avg_batch_kd
            opt.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0);
            opt.step()

        avg_accs = 100.0 * epoch_correct / total_samples
        avg_l_ce = epoch_loss_ce / total_samples;
        avg_l_kld = epoch_loss_kld / total_samples
        print(f"\n[Mutual KD] === Epoch {ep + 1}/{epochs} Summary ===")
        print("| Expert | Acc (%) | L_CE   | L_KLD  |")
        print("|:-------|:--------|:-------|:-------|")
        for i in range(num_experts):
            print(f"| Exp {i}  | {avg_accs[i]:<7.2f} | {avg_l_ce[i]:<6.4f} | {avg_l_kld[i]:<6.4f} |")
        print(f"| **Avg** | {avg_accs.mean():<7.2f} | {avg_l_ce.mean():<6.4f} | {avg_l_kld.mean():<6.4f} |")
        print("---")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience;
        self.min_delta = min_delta
        self.counter = 0;
        self.best_loss = None;
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss; self.counter = 0


def train_cloud_gate(gate: GateNet, experts: nn.ModuleList, proxy_dataset: SignalLocalDataset, device: torch.device,
                     epochs: int, batch_size: int, lr: float, lambda_gate_sup: float):
    gate.to(device)
    for e in experts: e.to(device); e.eval()
    loader = make_loader(proxy_dataset, batch_size, shuffle=True)
    opt = Adam(gate.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    for ep in range(epochs):
        gate.train();
        total_loss, total, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                expert_logits = torch.cat([e(x).unsqueeze(0) for e in experts], dim=0)
            gate_logits = gate(x);
            q = F.softmax(gate_logits, dim=1)
            mix_logits = (q.transpose(0, 1).unsqueeze(-1) * expert_logits).sum(dim=0)
            loss = F.cross_entropy(mix_logits, y)
            if lambda_gate_sup > 0.0: loss += lambda_gate_sup * F.cross_entropy(gate_logits, y)
            opt.zero_grad();
            loss.backward();
            opt.step()
            total_loss += loss.item() * y.size(0)
            correct += (mix_logits.argmax(dim=1) == y).sum().item();
            total += y.size(0)
        avg_loss = total_loss / total;
        acc = 100.0 * correct / total
        if (ep + 1) % 2 == 0: print(f"[Cloud Gate] Ep {ep + 1}/{epochs} | Loss {avg_loss:.4f} | Acc {acc:.2f}%")
        if acc >= 99.5 or early_stopping.early_stop: break
        early_stopping(avg_loss)


def analyze_feature_space(gate, experts, dataset, device, batch_size, visualizer):
    gate.to(device).eval();
    for e in experts: e.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    mixed_feats = [];
    all_labels = []
    print("[Analysis] Extracting features for T-SNE...")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            q = F.softmax(gate(x), dim=1).transpose(0, 1).unsqueeze(-1)
            expert_feats = torch.cat([e(x, return_features=True)[1].unsqueeze(0) for e in experts], dim=0)
            mix_feat = (q * expert_feats).sum(dim=0)
            mixed_feats.append(mix_feat.cpu().numpy());
            all_labels.append(y.cpu().numpy())
    visualizer.plot_tsne(np.concatenate(mixed_feats, axis=0), np.concatenate(all_labels, axis=0),
                         filename="tsne_cloud_gate_mixed.png")


def evaluate_cloud_gate_only_with_cm(gate, experts, dataset, device, batch_size, visualizer, title_suffix=""):
    gate.to(device).eval();
    for e in experts: e.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            expert_logits = torch.cat([e(x).unsqueeze(0) for e in experts], dim=0)
            q = F.softmax(gate(x), dim=1).transpose(0, 1).unsqueeze(-1)
            mix_logits = (q * expert_logits).sum(dim=0)
            y_true.extend(y.cpu().numpy());
            y_pred.extend(mix_logits.argmax(dim=1).cpu().numpy())
    acc = accuracy_score(y_true, y_pred) * 100.0
    visualizer.plot_confusion_matrix(y_true, y_pred, f"Cloud Gate Only {title_suffix}\n(Acc: {acc:.2f}%)",
                                     f"cm_cloud_gate{title_suffix}.png")
    return acc


def evaluate_and_analyze_hybrid(gate, cloud_experts, local_experts, dataset, device, batch_size, alpha, conf_th,
                                seen_eids, visualizer, cid):
    gate.to(device).eval();
    for e in cloud_experts: e.to(device).eval()
    for e in local_experts: e.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    y_true, y_pred = [], []
    total_count, local_path_count, local_path_correct, cloud_path_count, cloud_path_correct = 0, 0, 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if len(seen_eids) > 0:
                p_local = sum([F.softmax(local_experts[eid](x), dim=1) for eid in seen_eids]) / len(seen_eids)
            else:
                p_local = torch.full((x.size(0), local_experts[0].output_size), 1.0 / local_experts[0].output_size,
                                     device=device)
            local_conf, local_pred_idx = p_local.max(dim=1)
            expert_logits = torch.cat([e(x).unsqueeze(0) for e in cloud_experts], dim=0)
            q = F.softmax(gate(x), dim=1)
            top2_prob, top2_idx = torch.topk(q, k=2, dim=1)
            B = x.size(0)
            p_cloud = torch.zeros_like(p_local)
            for b in range(B):
                idx1, idx2 = top2_idx[b, 0], top2_idx[b, 1]
                w1, w2 = top2_prob[b, 0], top2_prob[b, 1]
                p_cloud[b] = (w1 * F.softmax(expert_logits[idx1, b], dim=0) + w2 * F.softmax(expert_logits[idx2, b],
                                                                                             dim=0)) / (w1 + w2 + 1e-8)
            cloud_pred_idx = p_cloud.argmax(dim=1)
            use_local = (local_conf >= conf_th)
            total_count += B
            local_path_count += use_local.sum().item()
            cloud_path_count += (~use_local).sum().item()
            local_path_correct += ((local_pred_idx == y) & use_local).sum().item()
            cloud_path_correct += ((cloud_pred_idx == y) & (~use_local)).sum().item()
            p_hybrid = use_local.float().unsqueeze(1) * p_local + (~use_local).float().unsqueeze(1) * (
                        (1 - alpha) * p_local + alpha * p_cloud)
            y_true.extend(y.cpu().numpy());
            y_pred.extend(p_hybrid.argmax(dim=1).cpu().numpy())
    acc = accuracy_score(y_true, y_pred) * 100.0
    acc_loc = (local_path_correct / local_path_count * 100) if local_path_count else 0
    acc_cld = (cloud_path_correct / cloud_path_count * 100) if cloud_path_count else 0
    visualizer.plot_hybrid_mechanism(total_count, local_path_count, acc_loc, cloud_path_count, acc_cld,
                                     f"hybrid_mechanism_client{cid}.png")
    visualizer.plot_confusion_matrix(y_true, y_pred, f"Hybrid - Client {cid}", f"confusion_matrix_client{cid}.png")
    return acc


def evaluate_local_seen_ensemble(experts, dataset, device, batch_size, seen_eids):
    if len(seen_eids) == 0: return 0.0
    for e in experts: e.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = sum([F.softmax(experts[eid](x), dim=1) for eid in seen_eids])
            if len(seen_eids) > 0: correct += (p.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def evaluate_cloud_ensemble(experts, dataset, device, batch_size):
    for e in experts: e.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = sum([F.softmax(e(x), dim=1) for e in experts])
            correct += (p.argmax(dim=1) == y).sum().item();
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def evaluate_cloud_gate_only(gate, experts, dataset, device, batch_size):
    gate.to(device).eval()
    for e in experts: e.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            expert_logits = torch.cat([e(x).unsqueeze(0) for e in experts], dim=0)
            q = F.softmax(gate(x), dim=1).transpose(0, 1).unsqueeze(-1)
            pred = (q * expert_logits).sum(dim=0).argmax(dim=1)
            correct += (pred == y).sum().item();
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


class FedUAVMoEPlanB:
    def __init__(self, args):
        self.args = args;
        self.device = get_device(args.gpu)
        print(f"[Init] Use device: {self.device}")
        self.visualizer = Visualizer(args.save_dir)
        self.local_datasets, self.global_test_dataset = build_fed_datasets(args)
        self.proxy_dataset = build_labeled_proxy(self.local_datasets, self.args.proxy_per_client, self.device,
                                                 self.args.num_classes, self.args.proxy_mixup_alpha)
        self.client_seen_labels = {}
        for cid, ds in enumerate(self.local_datasets):
            self.client_seen_labels[cid] = sorted(np.unique(ds.labels).astype(int).tolist())
            print(f"[Align] client={cid} seen train labels: {self.client_seen_labels[cid]}")
        self.num_experts = self.args.moe_num_experts
        self.experts = nn.ModuleList([ExpertNet(n_class=self.args.num_classes) for _ in range(self.num_experts)])
        self.gate = GateNet(num_experts=self.num_experts)
        self.client_experts = []

    def local_train_experts(self):
        print("=== Phase 1: 本地专家在本地正样本上训练 ===")
        for cid, ds in enumerate(self.local_datasets):
            if len(ds) == 0: continue
            uniq = np.unique(ds.labels).astype(int)
            if len(uniq) != 1: continue
            train_single_expert(self.experts[uniq[0]], make_loader(ds, self.args.local_bs, True), self.device,
                                self.args.local_ep, self.args.lr, self.args.momentum, self.args.weight_decay,
                                f"[Local Train] client={cid} training expert={uniq[0]}")

    def mutual_learn_experts(self):
        print("=== Phase 1.5: 云端精调 (Mixup + Triplet) ===")
        cloud_neg_finetune_experts(self.experts, self.proxy_dataset, self.device, self.args.mutual_learn_epochs,
                                   self.args.proxy_bs, self.args.mutual_learn_lr, self.args.neg_per_expert_max,
                                   self.args.mixup_alpha, triplet_margin=1.0, triplet_weight=1.0)

    def mutual_kd_experts(self):
        epochs = getattr(self.args, "mutual_kd_epochs", 0)
        if epochs > 0:
            print("=== Phase 1.75: 专家互蒸馏 (O(E) + CE + KD) ===")
            cloud_mutual_distill_experts(self.experts, self.proxy_dataset, self.device, epochs, self.args.proxy_bs,
                                         getattr(self.args, "mutual_kd_lr", 1e-4),
                                         getattr(self.args, "mutual_kd_alpha", 0.1),
                                         getattr(self.args, "mutual_kd_T", 2.0))
        else:
            print("[Phase 1.75] 已关闭。")

    def cloud_train_gate(self):
        print("=== Phase 2: 云端 Gate 训练 ===")
        train_cloud_gate(self.gate, self.experts, self.proxy_dataset, self.device, self.args.gate_cloud_epochs,
                         self.args.proxy_bs, self.args.kd_lr, self.args.gate_sup_lambda)

    def downlink_experts_to_clients(self):
        print("=== Phase 2.x: 下发专家 ===")
        self.client_experts = []
        for _ in range(self.args.num_locs):
            local_modules = nn.ModuleList()
            for exp in self.experts:
                new_exp = ExpertNet(n_class=self.args.num_classes);
                new_exp.load_state_dict(exp.state_dict());
                local_modules.append(new_exp)
            self.client_experts.append(local_modules)

    def evaluate(self):
        print("=== Phase 3: 测试阶段 ===")
        bs = self.args.bs;
        ds = self.global_test_dataset

        print("[Eval] Cloud Ensemble")
        acc_ens = evaluate_cloud_ensemble(self.experts, ds, self.device, bs)
        print(f"[Cloud Ensemble] Global Acc={acc_ens:.2f}%")

        print("[Eval] Cloud Gate Only")
        acc_gate = evaluate_cloud_gate_only_with_cm(self.gate, self.experts, ds, self.device, bs, self.visualizer,
                                                    "_Final")
        print(f"[Cloud Gate Only] Global Acc={acc_gate:.2f}%")

        analyze_feature_space(self.gate, self.experts, ds, self.device, bs, self.visualizer)

        print("[Eval] Hybrid")
        hybrid_accs = []
        for cid in range(self.args.num_locs):
            acc = evaluate_and_analyze_hybrid(self.gate, self.experts, self.client_experts[cid], ds, self.device, bs,
                                              self.args.hybrid_alpha, self.args.hybrid_conf_th,
                                              self.client_seen_labels[cid], self.visualizer, cid)
            hybrid_accs.append(acc)
            print(f"[Hybrid] client={cid} | Acc={acc:.2f}%")

        self.visualizer.plot_bar_comparison(
            {'Ensemble': acc_ens, 'Gate': acc_gate, 'Hybrid': sum(hybrid_accs) / len(hybrid_accs)}, "comparison.png")
        self.extra_eval_per_class()

    def extra_eval_per_class(self):
        print("=== Extra Eval ===")
        for c in range(self.args.num_classes):
            mask = (self.global_test_dataset.labels == c)
            if not np.any(mask): continue
            ds_c = SignalLocalDataset(self.global_test_dataset.signals[mask], self.global_test_dataset.labels[mask])
            oracle = evaluate_local_seen_ensemble(self.experts, ds_c, self.device, self.args.bs, [c])
            gate = evaluate_cloud_gate_only(self.gate, self.experts, ds_c, self.device, self.args.bs)
            print(f"Label {c}: Oracle={oracle:.2f}% | Gate={gate:.2f}%")


if __name__ == '__main__':
    args = args_parser()
    ensure_dir(args.save_dir);
    set_random_seed(args.seed)
    print("=== RUNNING MoE-FL (Plan B Final + TSNE) ===")
    fed = FedUAVMoEPlanB(args)
    fed.local_train_experts()
    fed.mutual_learn_experts()
    fed.mutual_kd_experts()
    fed.cloud_train_gate()
    fed.downlink_experts_to_clients()
    fed.evaluate()