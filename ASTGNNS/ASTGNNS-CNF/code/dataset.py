import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, dataset
from torch.nn.functional import one_hot
import json
import re

from utils import load_pickle_file, Statements, read_file
from formula_parser import fof_formula_transformer
from graph import Graph

VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")
CONNECTIVE_PATTERN = {"!", "?", "|", "&", "=>", "<=>", "~"}
BOOL = "$true"


class PairData(Data):
    def __init__(self,
                 x_s=None,
                 treelet_index_s=None,
                 term_walk_index_s=None,
                 parent_index_s=None,
                 child_index_s=None,
                 treelet_index_reverse_s=None,
                 x_t=None,
                 treelet_index_t=None,
                 term_walk_index_t=None,
                 parent_index_t=None,
                 child_index_t=None,
                 treelet_index_reverse_t=None,
                 y=None):
        super().__init__()
        self.x_s = x_s
        self.x_t = x_t
        self.term_walk_index_s = term_walk_index_s
        self.term_walk_index_t = term_walk_index_t
        self.treelet_index_reverse_s = treelet_index_reverse_s
        self.treelet_index_reverse_t = treelet_index_reverse_t
        self.treelet_index_s = treelet_index_s
        self.treelet_index_t = treelet_index_t
        self.parent_index_s = parent_index_s
        self.parent_index_t = parent_index_t
        self.child_index_s = child_index_s
        self.child_index_t = child_index_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "term_walk_index_s":
            return self.x_s.size(0)
        if key == "term_walk_index_t":
            return self.x_t.size(0)
        if key == "treelet_index_reverse_s":
            return self.x_s.size(0)
        if key == "treelet_index_reverse_t":
            return self.x_t.size(0)
        if key == "treelet_index_s":
            return self.x_s.size(0)
        if key == "treelet_index_t":
            return self.x_t.size(0)
        if key == "parent_index_s":
            return self.x_s.size(0)
        if key == "parent_index_t":
            return self.x_t.size(0)
        if key == "child_index_s":
            return self.x_s.size(0)
        if key == "child_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class FormulaGraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_class,
                 statements_file,
                 node_dict_file,
                 rename=True):
        self.root = root
        self.data_class = data_class
        self.statements = Statements(statements_file)
        self.rename = rename
        self.node_dict = load_pickle_file(node_dict_file)
        super(FormulaGraphDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.json".format(self.data_class)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.data_class)]

    def graph_process(self, G):
        nodes = []
        treelet_indices = []
        term_walk_indices = []
        parent_indices = []
        child_indices = []
        treelet_indices_reverse = []

        for node in G:
            nodes.append(node.name)
            for parent in node.parents:
                parent_indices.append([node.id, parent.id])

            for child in node.children:
                child_indices.append([node.id, child.id])

            if node.parents and node.children:
                for parent in node.parents:
                    for child in node.children:
                        term_walk_indices.append([parent.id,
                                                  node.id,
                                                  child.id])

            if node.children and len(node.children) > 1:
                children_indices = []
                for child in node.children:
                    children_indices.append(child.id)
                for i in range(len(children_indices) - 1):
                    treelet_indices.append([children_indices[i], node.id, children_indices[i + 1]])

            if node.parents and len(node.parents) > 1:
                parents_indices = []
                for child in node.children:
                    parents_indices.append(child.id)
                for i in range(len(parents_indices) - 1):
                    treelet_indices_reverse.append([parents_indices[i], node.id, parents_indices[i + 1]])

        treelet_indices = np.array(
            treelet_indices, dtype=np.int64).reshape(-1, 3).T
        treelet_indices_reverse = np.array(
            treelet_indices_reverse, dtype=np.int64).reshape(-1, 3).T
        term_walk_indices = np.array(
            term_walk_indices, dtype=np.int64).reshape(-1, 3).T
        parent_indices = np.array(
            parent_indices, dtype=np.int64).reshape(-1, 2).T
        child_indices = np.array(
            child_indices, dtype=np.int64).reshape(-1, 2).T
        return nodes, term_walk_indices, parent_indices, child_indices, treelet_indices, treelet_indices_reverse

    def vectorization(self, objects, object_dict):
        indices = [object_dict[obj] for obj in objects]
        onehot = one_hot(torch.LongTensor(indices), len(object_dict)).float()
        return onehot

    def process(self):
        raw_examples = \
            [json.loads(line) for line in read_file(self.raw_paths[0])]
        dataList = []
        for example in raw_examples:
            conj, prem, label = example
            conj_graph = Graph(fof_formula_transformer(self.statements[conj]),
                               rename=self.rename)
            prem_graph = Graph(fof_formula_transformer(self.statements[prem]),
                               rename=self.rename)
            c_nodes, c_term_walk_indices, c_parent_indices, c_child_indices, c_treelet_indices,\
                c_treelet_indices_reverse = self.graph_process(conj_graph)
            p_nodes, p_term_walk_indices, p_parent_indices, p_child_indices, p_treelet_indices,\
                p_treelet_indices_reverse = self.graph_process(prem_graph)
            data = PairData(
                x_s=self.vectorization(c_nodes, self.node_dict),
                term_walk_index_s=torch.from_numpy(c_term_walk_indices),
                parent_index_s=torch.from_numpy(c_parent_indices),
                child_index_s=torch.from_numpy(c_child_indices),
                treelet_index_s=torch.from_numpy(c_treelet_indices),
                treelet_index_reverse_s=torch.from_numpy(c_treelet_indices_reverse),
                x_t=self.vectorization(p_nodes, self.node_dict),
                term_walk_index_t=torch.from_numpy(p_term_walk_indices),
                parent_index_t=torch.from_numpy(p_parent_indices),
                child_index_t=torch.from_numpy(p_child_indices),
                treelet_index_t=torch.from_numpy(p_treelet_indices),
                treelet_index_reverse_t=torch.from_numpy(p_treelet_indices_reverse),
                y=torch.LongTensor([label]))
            dataList.append(data)
        data, slices = self.collate(data_list=dataList)
        torch.save((data, slices), self.processed_paths[0])


