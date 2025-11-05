import argparse


def lth_args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--lr", default=0.05, type=float, help="Learning rate")
    parser.add_argument("--lr", default=0.002, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--lth_epoch_iters", default=3, type=int)
    parser.add_argument(
        "--dataset",
        default="creditcard",
        type=str,
    )
    parser.add_argument(
        "--arch_type",
        default="resnet18",
        type=str,
    )
    parser.add_argument(
        "--setting",
        default="",
        type=str,
    )
    parser.add_argument(
        "--prune_percent", default=25, type=float, help="Pruning percent"
    )
    parser.add_argument("--prune_target", default=80, type=int, help="Pruning target")
    parser.add_argument(
        # "--com_rounds", type=int, default=4, help="rounds of fedavg training"
        "--com_rounds", type=int, default=8, help="rounds of fedavg training"
    )
    parser.add_argument(
        "--la_epochs",
        type=int,
        default=15,
        help="rounds of training for local alt optimization",
    )
    parser.add_argument("--iid", action="store_true",default=True, help="whether i.i.d or not")
    parser.add_argument("--num_users", type=int, default=30, help="number of users: K")
    parser.add_argument(
        "--shard_per_user", type=int, default=2, help="classes per user"
    )
    parser.add_argument("--local_bs", type=int, default=32, help="local batch size: B")
    parser.add_argument(
        # "--frac", type=float, default=0.1, help="the fraction of clients: C"
        "--frac", type=float, default=0.1, help="the fraction of clients: C"

    )
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--model", type=str, default="mlp", help="model name")
    parser.add_argument("--bs", type=int, default=128, help="test batch size")
    parser.add_argument("--lth_freq", type=int, default=1, help="frequency of lth")
    parser.add_argument("--pretrained_init", action="store_true")
    parser.add_argument("--clipgradnorm", action="store_true")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--test_size", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default="prune_rate_vary")
    parser.add_argument(
        "--server_data_ratio",
        type=float,
        default=0.0,
        help="The percentage of data that servers also have across data of all clients.",
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    # 添加 FLM non-iid 划分所需要的参数
    parser.add_argument('--split_dataset_type', type=int, default=1,
                        help="type of non-iid split for creditcard dataset")
    parser.add_argument('--split_dataset_ratio', type=list, default=[0.4, 0.3, 0.3],
                        help="ratio of non-iid split for creditcard dataset")

    parser.add_argument("--fed_type", type=int, default=3,
                        help="联邦聚合方式  1-FedAVG，2-FedMEAN，3-FedRWA, 4-FedProx, 5-Moon")
    parser.add_argument("--risk_type", type=int, default=3,
                        help="风险权值类型   1-数据集大小，2-数据集欺诈样本数量，3-数据集欺诈样本金额")
    args = parser.parse_args()
    return args