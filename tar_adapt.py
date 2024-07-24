import argparse
import os
import os.path as osp
import torch.optim as optim
import network
import random
from utils import *
import torch.nn.functional as F

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 5e-4
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer

def train_target(args):
    dset_loaders = dataset_load(args)
    ## set base network

    netF = network.ResNet_FE().cuda()
    oldC = network.ETFClassifier(n_features=args.bottleneck, n_classes=args.class_num).cuda()

    modelpath = args.output_dir + "/source_F.pt"
    netF.load_state_dict(torch.load(modelpath), strict=False)
    modelpath = args.output_dir + "/source_C.pt"
    oldC.load_state_dict(torch.load(modelpath))

    netF.self_attn.set_temperature(args.temperature)  # Update the temperature
    netF.self_attn.set_mode(args.mode)

    for k, v in oldC.named_parameters():
        v.requires_grad = False

    optimizer = optim.SGD(
        [
            {"params": netF.feature_layers.parameters(), "lr": args.lr * args.lr_encoder},
            {"params": netF.bottle.parameters(), "lr": args.lr * 2},
            {"params": netF.bn.parameters(), "lr": args.lr * 2},  # 10
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    optimizer = op_copy(optimizer)

    acc_init = 0
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, args.bottleneck)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            output = netF.forward(inputs)  # a^t
            output_norm = F.normalize(output)
            outputs = oldC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()

    while iter_num < max_iter:

        netF.train()
        # iter_target = iter(dset_loaders["target"])

        try:
            inputs_test, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1

        inputs_target = inputs_test.cuda()

        features_test = netF(inputs_target)

        output = oldC(features_test)
        softmax_out = nn.Softmax(dim=1)(output)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            pred_bs = softmax_out

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = pred_bs.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        )  # batch x K x C

        loss = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)
        )

        alpha = 0.4

        # other prediction scores as negative pairs
        mask = torch.ones((inputs_target.shape[0], inputs_target.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = softmax_out.T

        dot_neg = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()

            acc1 = cal_acc_(dset_loaders["test"], netF, oldC)  # 1
            log_str = "Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%".format(
                args.dset, iter_num, max_iter, acc1 * 100
            )
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)
    if acc1 >= acc_init:
        acc_init = acc1
        best_netF = netF.state_dict()
        best_netC = oldC.state_dict()

        torch.save(best_netF, osp.join(args.output_dir, "F_final.pt"))
        torch.save(best_netC, osp.join(args.output_dir, "C_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain Adaptation"
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=60, help="maximum epoch")
    parser.add_argument("--batch_size", type=int, default=108, help="batch_size")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--worker", type=int, default=0, help="number of workers")
    parser.add_argument("--dset", type=str, default="a2d")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lr_encoder", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--mode", action="store_true", default=False)
    parser.add_argument("--class_num", type=int, default=0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--output", type=str, default="weight")
    parser.add_argument("--file", type=str, default="log_tar")
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

    if args.office31:
        args.class_num = 31
    elif args.home:
        args.class_num = 65
    elif args.bird31:
        args.class_num = 31
    elif args.cub:
        args.class_num = 200

    current_folder = "./"
    args.output_dir = osp.join(
        current_folder, args.output, "seed" + str(args.seed), args.dset
    )
    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + ".txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()

    train_target(args)

