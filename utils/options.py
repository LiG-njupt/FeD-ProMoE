import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser(description="UAV MoE-FL (Plan B: Phase 0 + 1.5 with Mixup)")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./runs/uav_moe_mixup")

    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--train_dir", type=str, default="train_Power_Normalization")
    parser.add_argument("--test_dir", type=str, default="test_Power_Normalization")

    parser.add_argument("--dis_list", type=str, nargs="+", default=["D00", "D01", "D10"])

    parser.add_argument("--num_locs", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=8)

    parser.add_argument("--local_ep", type=int, default=10)
    parser.add_argument("--local_bs", type=int, default=32)
    parser.add_argument("--bs", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--moe_num_experts", type=int, default=8)

    parser.add_argument("--proxy_per_client", type=int, default=200)
    parser.add_argument("--proxy_mixup_alpha", type=float, default=0.05)
    parser.add_argument("--mutual_learn_epochs", type=int, default=80)
    parser.add_argument("--mutual_learn_lr", type=float, default=2e-3)
    parser.add_argument("--neg_per_expert_max", type=int, default=1400)

    parser.add_argument("--mixup_alpha", type=float, default=0.4)

    parser.add_argument("--mutual_kd_epochs", type=int, default=5)
    parser.add_argument("--mutual_kd_lr", type=float, default=1e-4)
    parser.add_argument("--mutual_kd_alpha", type=float, default=0.1)

    parser.add_argument("--proxy_bs", type=int, default=128)
    parser.add_argument("--gate_cloud_epochs", type=int, default=24)
    parser.add_argument("--kd_lr", type=float, default=2e-3)
    parser.add_argument("--gate_sup_lambda", type=float, default=0.03)

    parser.add_argument("--hybrid_mode", type=str, default="cloud_gate_top2", choices=["local_only", "cloud_gate_top2", "cloud_ensemble"])
    parser.add_argument("--hybrid_alpha", type=float, default=1.0)
    parser.add_argument("--hybrid_conf_th", type=float, default=0.8)

    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--hard_rounds", type=int, default=1)
    parser.add_argument("--cloud_kd_epochs", type=int, default=0)
    parser.add_argument("--proxy_batches", type=int, default=0)
    parser.add_argument("--kd_T", type=float, default=2.0)
    parser.add_argument("--cloud_kd_div", type=float, default=0.0)
    parser.add_argument("--route_tau", type=float, default=1.0)
    parser.add_argument("--route_eps", type=float, default=0.0)
    parser.add_argument("--moe_kd_weight", type=float, default=0.0)
    parser.add_argument("--moe_kd_T", type=float, default=2.0)
    parser.add_argument("--moe_kd_pair", type=float, default=0.0)
    parser.add_argument("--moe_kd_div", type=float, default=0.0)
    parser.add_argument("--moe_kd_topk", type=int, default=0)
    parser.add_argument("--mutual_learn_alpha", type=float, default=1.0)
    parser.add_argument("--mutual_learn_T", type=float, default=2.0)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    return args
