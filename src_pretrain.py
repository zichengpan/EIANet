import argparse
import os
import torch.optim as optim
import network
import random
from utils import *
import shutil
import os.path as osp


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_source(args):
    dset_loaders = dataset_load(args)
    ## set base network
    netF = network.ResNet_FE().cuda()
    netC = network.ETFClassifier(n_features=args.bottleneck, n_classes=args.class_num).cuda()

    netF.self_attn.set_mode(args.mode)

    for k, v in netC.named_parameters():
        v.requires_grad = False

    optimizer = optim.SGD(
        [
            {"params": netF.feature_layers.parameters(), "lr": args.lr},
            {"params": netF.bottle.parameters(), "lr": args.lr * 10},
            {"params": netF.bn.parameters(), "lr": args.lr * 10},
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    acc_init = 0
    for epoch in range(args.max_epoch):
        netF.train()
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source, labels_source, _) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

            output = netF(inputs_source)
            output = netC(output)

            loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth
            )(output, labels_source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        netF.eval()
        netC.eval()
        acc_s_tr = cal_acc_(dset_loaders["source_te"], netF, netC)
        log_str = "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
            args.dset, epoch + 1, args.max_epoch, acc_s_tr * 100
        )
        args.out_file.write(log_str + "\n")
        args.out_file.flush()
        print(log_str)

        if acc_s_tr >= acc_init:
            acc_init = acc_s_tr
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))


def test_target(args):
    dset_loaders = dataset_load(args)
    ## set base network
    netF = network.ResNet_FE().cuda()
    netC = network.ETFClassifier(n_features=args.bottleneck, n_classes=args.class_num).cuda()

    args.modelpath = args.output_dir + "/source_F.pt"
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + "/source_C.pt"
    netC.load_state_dict(torch.load(args.modelpath))

    netF.self_attn.set_mode(args.mode)

    netF.eval()
    netC.eval()

    acc = cal_acc_(dset_loaders["test"], netF, netC)
    log_str = "Task: {}, Accuracy = {:.2f}%".format(args.dset, acc * 100)
    args.out_file.write(log_str + "\n")
    args.out_file.flush()
    print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source-Free Domain Adaptation")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=20, help="maximum epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--mode", action="store_true", default=False)
    parser.add_argument("--worker", type=int, default=0, help="number of workers")
    parser.add_argument("--dset", type=str)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--class_num", type=int, default=0)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="weight")
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--office31", action="store_true")
    parser.add_argument("--cub", action="store_true")
    parser.add_argument("--bird31", action="store_true")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True


    current_folder = "./"
    args.output_dir = osp.join(
        current_folder, args.output, "seed" + str(args.seed), args.dset
    )
    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.office31:
        task = ["a", "d", "w"]
        args.class_num = 31
    elif args.home:
        task = ["c", "a", "p", "r"]
        args.class_num = 65
    elif args.bird31:
        task = ["c", "i", "n"]
        args.class_num = 31
    elif args.cub:
        task = ["c", "p"]
        args.class_num = 200

    task_s = args.dset.split("2")[0]
    task.remove(task_s)
    task_all = [task_s + "2" + i for i in task]
    for task_sameS in task_all:
        path_task = (
            os.getcwd()
            + "/"
            + args.output
            + "/seed"
            + str(args.seed)
            + "/"
            + task_sameS
        )
        if not osp.exists(path_task):
            os.mkdir(path_task)

    if not osp.exists(osp.join(args.output_dir + "/source_F.pt")):
        args.out_file = open(osp.join(args.output_dir, "log_src.txt"), "w")
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()
        train_source(args)
        test_target(args)

    file_f = osp.join(args.output_dir + "/source_F.pt")
    file_c = osp.join(args.output_dir + "/source_C.pt")
    task.remove(args.dset.split("2")[1])
    task_remain = [task_s + "2" + i for i in task]
    for task_sameS in task_remain:
        path_task = (
            os.getcwd()
            + "/"
            + args.output
            + "/seed"
            + str(args.seed)
            + "/"
            + task_sameS
        )
        pathF_copy = osp.join(path_task, "source_F.pt")
        pathC_copy = osp.join(path_task, "source_C.pt")
        shutil.copy(file_f, pathF_copy)
        shutil.copy(file_c, pathC_copy)