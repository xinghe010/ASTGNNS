import json
import pickle
import logging
import matplotlib.pyplot as plt
import numpy as np


def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines


def dumps_list_to_json(obj, file_path):
    with open(file_path, "w+") as f:
        f.write("\n".join([json.dumps(element) for element in obj]))


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_pickle_file(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def set_recorder(name, logfile):
    recorder = logging.getLogger(name)
    recorder.setLevel(logging.INFO)
    rf_handler = logging.StreamHandler()
    rf_handler.setLevel(logging.INFO)
    rf_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    recorder.addHandler(rf_handler)
    recorder.addHandler(f_handler)
    return recorder


class Statements:
    def __init__(self, statements_file):
        self.statements = self.build_statements(statements_file)

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, name):
        return self.statements[name]

    def __iter__(self):
        return self.statements.__iter__()

    def build_statements(self, statements_file):
        statements = dict()
        lines = read_file(statements_file)
        for line in lines:
            name = line.split(',')[0].replace("fof(", "")
            statements[name] = line.replace(" ", "")
        return statements


def py_plot(title, train_loss, valid_loss, train_acc, valid_acc, train_f1,
            valid_f1, train_recall, valid_recall, train_precision, valid_precision,
            save_file):
    assert len(train_loss) == len(valid_loss)
    assert len(train_acc) == len(valid_acc)
    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.subplot(231)
    plt.plot(epochs, train_loss, "-", color="salmon", label="train loss")
    plt.plot(epochs, valid_loss, "-", color="saddlebrown", label="valid loss")
    plt.title("loss")
    plt.legend()
    plt.grid()

    plt.subplot(232)
    plt.plot(epochs, train_acc, "-", color="salmon", label="train acc")
    plt.plot(epochs, valid_acc, "-", color="saddlebrown", label="valid acc")
    plt.title("accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(233)
    plt.plot(epochs, train_f1, "-", color="salmon", label="train f1")
    plt.plot(epochs, valid_f1, "-", color="saddlebrown", label="valid f1")
    plt.title("f1")
    plt.legend()
    plt.grid()

    plt.subplot(234)
    plt.plot(epochs, train_recall, "-", color="salmon", label="train recall")
    plt.plot(epochs,
             valid_recall,
             "-",
             color="saddlebrown",
             label="valid recall")
    plt.title("recall")
    plt.legend()
    plt.grid()

    plt.subplot(235)
    plt.plot(epochs, train_precision, "-", color="salmon", label="train recall")
    plt.plot(epochs,
             valid_precision,
             "-",
             color="saddlebrown",
             label="valid recall")
    plt.title("precision")
    plt.legend()
    plt.grid()

    plt.suptitle(title)
    plt.savefig(save_file, dpi=2000)
    plt.close()


if __name__ == "__main__":
    history = load_pickle_file(
        "/home/dell/lanyongqi/TS_GNN/model_save/history.pkl")
    py_plot(
        "evaluation", history["train_loss"], history["valid_loss"],
        history["train_acc"], history["valid_acc"],
        "/home/dell/lanyongqi/TS_GNN/model_save//figure")
