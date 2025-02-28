import os
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import train, valid, test
from model import PremiseSelectionModel
from dataset import FormulaGraphDataset
from utils import set_recorder, dump_pickle_file, py_plot


def hyper_parameters():
    params = argparse.ArgumentParser()
    params.add_argument(
        "--model_save",
        type=str,
        default="/data2/lanyongqi/TS_GNN/model_save",
        help="the directory to save models")
    params.add_argument(
        "--root_dir",
        type=str,
        default="/data2/lanyongqi/TS_GNN/dataset",
        help="the directory to save data")
    params.add_argument("--node_out_channels",
                        type=int,
                        default=512,
                        help="the dimension of node")
    params.add_argument("--layers",
                        type=int,
                        default=2,
                        help="the number of message passing steps")
    params.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="device name {cpu, cuda:0, cuda:1}")
    params.add_argument("--epochs",
                        type=int,
                        default=100,
                        help='Number of training episodes')
    params.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Initial learning rate for Adam")
    params.add_argument("--weight_decay",
                        type=float,
                        default=1e-4,
                        help="L2 normalization penality")
    params.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch Size")
    args = params.parse_args(args=[])
    return args


def main():
    args = hyper_parameters()

    if not os.path.exists(args.model_save):
        os.makedirs(args.model_save)
    recorder = set_recorder("TS_GNN",
                            os.path.join(args.model_save, "record.log"))

    if args.device == "cpu":
        torch.manual_seed(24)
    else:
        torch.cuda.manual_seed_all(24)

    params_info = ''
    for key, value in vars(args).items():
        params_info += '\n{}: {}'.format(key, value)
    recorder.info(params_info)

    model = PremiseSelectionModel(793,
                                  args.node_out_channels,
                                  args.layers).to(device=args.device)

    optimizer = Adam(params=[{'params': model.parameters()}], lr=args.lr,
                     weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    recorder.info('------DATA LOADING------')

    train_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "train"),
                                        "train",
                                        os.path.join(args.root_dir,
                                                     "statements"),
                                        os.path.join(args.root_dir,
                                                     "node_dict.pkl"),
                                        rename=True)
    valid_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "valid"),
                                        "valid",
                                        os.path.join(args.root_dir,
                                                     "statements"),
                                        os.path.join(args.root_dir,
                                                     "node_dict.pkl"),
                                        rename=True)
    test_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "test"),
                                       "test",
                                       os.path.join(args.root_dir,
                                                    "statements"),
                                       os.path.join(args.root_dir,
                                                    "node_dict.pkl"),
                                       rename=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              follow_batch=["x_s", "x_t"])
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              follow_batch=["x_s", "x_t"])
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             follow_batch=["x_s", "x_t"])

    recorder.info('------DATA LOADED------')

    recorder.info('------PROCESS START------')
    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": [],
               "train_f1": [], "valid_f1": [], "train_recall": [], "valid_recall": [],
               "train_precision": [], "valid_precision": [], "test_loss": None,
               "test_acc": None, "test_f1": None, "test_recall": None, "test_precision": None}

    best_epoch = -1
    best_state_dict = {"model": None}
    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        recorder.info('------learning rate is {}------'.format(
            optimizer.param_groups[0]["lr"]))

        train_loss, train_acc, train_f1, train_recall, train_precision = train(
            epoch, train_loader, model, optimizer, args.device, recorder)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["train_recall"].append(train_recall)
        history["train_precision"].append(train_precision)

        valid_loss, valid_acc, valid_f1, valid_recall, valid_precision = valid(
            epoch, valid_loader, model, args.device, recorder)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)
        history["valid_f1"].append(valid_f1)
        history["valid_recall"].append(valid_recall)
        history["valid_precision"].append(valid_precision)

        if best_valid_loss > valid_loss:
            best_epoch = epoch
            best_state_dict["model"] = model.state_dict()
            best_valid_loss = valid_loss

        lr_scheduler.step(valid_loss)

    torch.save(best_state_dict, os.path.join(args.model_save, "best.pt"))

    recorder.info('------the best epoch is {}------'.format(best_epoch))

    model.load_state_dict(best_state_dict["model"])
    test_loss, test_acc, test_f1, test_recall, test_precision = test(test_loader, model,
                                                                     args.device, recorder)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["test_f1"] = test_f1
    history["test_recall"] = test_recall
    history["test_precision"] = test_precision

    dump_pickle_file(history, os.path.join(args.model_save, "history.pkl"))
    py_plot("evaluation", history["train_loss"], history["valid_loss"],
            history["train_acc"], history["valid_acc"], history["train_f1"],
            history["valid_f1"], history["train_recall"],
            history["valid_recall"], history["train_precision"], history["valid_precision"],
            os.path.join(args.model_save, "figure"))

    recorder.info('------PROCESS FINISH------')


if __name__ == "__main__":
    main()