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
    args = parser.parse_args()
    return args


class Args:
    def __init__(self):
        # federated arguments
        self.epochs = 200  #rounds of training
        self.num_users = 30  #number of users: K
        self.frac = 0.4  #the fraction of clients: C
        self.local_ep = 2  #the number of local epochs: E
        self.local_bs = 128  #local batch size: B
        self.bs = 128  #test batch size
        self.lr = 0.01  #learning rate
        self.momentum = 0.5 #SGD momentum (default: 0.5)
        # self.split = 'user'  # train-test split type, user or sample

        # model arguments
        self.model = 'mlp'  # model name

        # other arguments
        self.dataset = 'creditcard'  # name of dataset
        self.iid = True  # whether i.i.d or not
        self.num_classes = 2  # number of classes
        # self.num_channels = 1  # number of channels of imges
        self.gpu = -1  # GPU ID, -1 for CPU
        # self.stopping_rounds = 10  # rounds of early stopping
        self.verbose = True  # verbose print
        self.seed = 1  # random seed (default: 1)
        self.all_clients = False  # aggregation over all clients

        # new added arguments
        self.split_dataset_type = 1 # 1-根据数据集大小划分，2-根据欺诈样本比例划分(此时各节点数据集大小相同)，3-根据欺诈金额比例划分(此时各节点数据集大小相同)
        self.split_dataset_ratio = [1,2,2,3,3,4,4,1,2,3,1,2,2,3,3,4,3,1,2,3]  #划分数据集时的数据比例
        self.fed_type = 3 #联邦聚合方式  1-FedAVG，2-FedMEAN，3-FedRWA, 4-FedProx, 5-Moon
        self.risk_type = 3 #风险权值类型   1-数据集大小，2-数据集欺诈样本数量，3-数据集欺诈样本金额