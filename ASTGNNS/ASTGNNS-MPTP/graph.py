import re
import matplotlib.pyplot as plt
import networkx as nx

from formula_parser import fof_formula_transformer


NEGATIVE_CONNECTIVE = {"~"}
BINARY_CONNECTIVE = {"=>", "<=>"}
BINARY_CONNECTIVE1 = {"=>"}
ASSOC_CONNECTIVE = {"|", "&"}
QUANTIFIER = {"!", "?"}
EQUAL = {"=", "!="}
BOOL = {"$true"}
VARIABLE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
'''functor contain predicate, constant and functor'''
FUNCTOR_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")


class Node:

    def __init__(self, name):
        self.id = self.__class__.id
        self.name = name
        self.children = []
        self.parents = []
        self.quantifer = None
        self.quantified_variable = []
        self.scoped = []
        self.__class__.id += 1
        self.time = 0

    @classmethod
    def rest_id(cls):
        cls.id = 0

    def __str__(self):
        extra = ''
        if re.match(VARIABLE_PATTERN, self.name):
            extra = 'quantifer {}'.format(
                self.quantifer.id if self.quantifer is not None else '')
        elif self.name in QUANTIFIER:
            extra = 'quantified variable {}'.format(
                ' '.join(str(x.id) for x in self.quantified_variable))



        parents_info = ' '.join([str(x.id) for x in self.parents])
        children_info = ' '.join([str(x.id) for x in self.children])
        scoped_info = ' '.join(([str(x.id) for x in self.scoped]))
        return '<{}>: token {} | parents {} | children {} | scoped {} | {}'.\
            format(
                self.id,
                self.name,
                parents_info,
                children_info,
                scoped_info,
                extra)

    def __repr__(self):
        return self.__str__()


class Graph:
    def __init__(self, formula, rename = True):
        self.graph = []
        self.id2subterm = dict()
        self.convert(formula, rename)
        self.rename = rename


    def __iter__(self):
        return self.graph.__iter__()

    def __getitem__(self, index):
        return self.graph[index]

    def __len__(self):
        return len(self.graph)


    def create_quantifer_node(self, name, parent):
        quantifer_node = Node(name)
        if parent:
            quantifer_node.parents.append(parent)
            quantifer_node.scoped.extend(parent.scoped)
            if parent.name in QUANTIFIER:
                quantifer_node.scoped.append(parent)

        self.graph.append(quantifer_node)
        return quantifer_node



    def create_functor_node(self, name, parent):
        functor_node = Node(name)
        self.graph.append(functor_node)
        if parent:
            functor_node.parents.append(parent)
            for node in parent.scoped:
                if node not in functor_node.scoped:
                    functor_node.scoped.append(node)
            if parent.name in QUANTIFIER:
                functor_node.scoped.append(parent)

        return functor_node


    def create_variable_node(self, name, parent):
        assert parent is not None, \
            "a variable should have at least one parent"

        if parent.name in QUANTIFIER:
            variable_node = Node(name)
            variable_node.quantifer = parent
            variable_node.scoped.extend(parent.scoped)
            variable_node.scoped.append(parent)
            parent.quantified_variable.append(variable_node)
            self.graph.append(variable_node)
        else:
            for node in parent.scoped:
                for child in node.children:
                    if child.name == name:
                        variable_node = child
                        break
        variable_node.parents.append(parent)
        return variable_node


    def create_constant_node(self, name, parent):
        exsiting_same_constant_nodes = [
            node for node in self.graph if node.name == name]
        if len(exsiting_same_constant_nodes) == 1 and \
                exsiting_same_constant_nodes[0].children == []:
            constant_node = exsiting_same_constant_nodes[0]
        else:
            constant_node = Node(name)
            self.graph.append(constant_node)
        if parent:
            constant_node.parents.append(parent)
            for node in parent.scoped:
                if node not in constant_node.scoped:
                    constant_node.scoped.append(node)
        return constant_node


    def create_connective_node(self, name, parent):
        connective_node = Node(name)
        if parent:
            connective_node.parents.append(parent)
            connective_node.scoped.extend(parent.scoped)
            if parent.name in QUANTIFIER:
                connective_node.scoped.append(parent)
        self.graph.append(connective_node)
        return connective_node


    def create_negative_node(self, name, parent):
        negative_node = Node(name)
        if parent:
            negative_node.parents.append(parent)
            negative_node.scoped.extend(parent.scoped)
            if parent.name in QUANTIFIER:
                negative_node.scoped.append(parent)
        self.graph.append(negative_node)
        return negative_node


    def merge_sub(self, formula, parent):
        for id in self.id2subterm:
            if self.id2subterm[id][0] == formula:
                pre_node = self.id2subterm[id][1]
                pre_node.parents.append(parent)
                return pre_node


    def check_merge(self, formula, parent):
        if formula in [self.id2subterm[id][0] for id in self.id2subterm]:
            for id in self.id2subterm:
                if self.id2subterm[id][0] == formula:
                    pre_node = self.id2subterm[id][1]
                    var_flag = self.check_variable(pre_node)
                    if not var_flag:
                        return True
                    else:
                        if set(pre_node.scoped).issubset(set(parent.scoped)):
                            return True
                        else:
                            return False


    def check_variable(self, node):
        if re.match(VARIABLE_PATTERN, node.name):
            return True
        else:
            if not node.children:
                return False
            else:
                for child in node.children:
                    flag = self.check_variable(child)
                    if flag:
                        return True
                else:
                    return False


    def formula_to_dense_graph(self, formula, parent=None):
        number = 0
        if isinstance(formula, str):
            if re.match(VARIABLE_PATTERN, formula):
                variable_node = self.create_variable_node(formula, parent)
                return variable_node
            if re.match(FUNCTOR_PATTERN, formula) or (formula in BOOL):
                constant_node = self.create_constant_node(formula, parent)
                return constant_node

        if isinstance(formula[0], str) and \
                formula[0] in QUANTIFIER and len(formula) == 3:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                if number == 0:
                    quantifer_node = self.create_quantifer_node(formula[0], parent)
                    self.id2subterm[quantifer_node.id] = (formula, quantifer_node)
                    for variable in formula[1]:
                        variable_node = self.create_variable_node(
                            variable, quantifer_node)
                        quantifer_node.children.append(variable_node)
                if formula[2][0] is not None:
                    if formula[2][0] == formula[0]:
                       number = 1
                       for variable in formula[2][1]:
                            variable_node = self.create_variable_node(
                               variable, quantifer_node)
                            quantifer_node.children.append(variable_node)
                       if formula[2][2][0] != formula[2][0]:
                           node = self.formula_to_dense_graph(formula[2][2], quantifer_node)
                       if formula[2][2][0] == formula[2][0]:
                           for variable in formula[2][2][1]:
                               variable_node = self.create_variable_node(
                                    variable, quantifer_node)
                               quantifer_node.children.append(variable_node)
                           if formula[2][2][2][0] != formula[2][2][0]:
                               node = self.formula_to_dense_graph(formula[2][2][2], quantifer_node)
                           if formula[2][2][2][0] == formula[2][2][0]:
                               number = 1
                               for variable in formula[2][2][2][1]:
                                    variable_node = self.create_variable_node(
                                        variable, quantifer_node)
                                    quantifer_node.children.append(variable_node)
                               if formula[2][2][2][2][0] != formula[2][2][2][0]:
                                  node = self.formula_to_dense_graph(formula[2][2][2][2], quantifer_node)
                               if formula[2][2][2][2][0] == formula[2][2][2][0]:
                                   number = 1
                                   for variable in formula[2][2][2][2][1]:
                                       variable_node = self.create_variable_node(
                                           variable, quantifer_node)
                                       quantifer_node.children.append(variable_node)
                                   if formula[2][2][2][2][2][0] != formula[2][2][2][2][0]:
                                       node = self.formula_to_dense_graph(formula[2][2][2][2][2],
                                                                          quantifer_node)
                                   if formula[2][2][2][2][2][0] == formula[2][2][2][2][0]:
                                       number = 1
                                       for variable in formula[2][2][2][2][2][1]:
                                           variable_node = self.create_variable_node(
                                               variable, quantifer_node)
                                           quantifer_node.children.append(variable_node)
                                       if formula[2][2][2][2][2][2][0] != formula[2][2][2][2][2][0]:
                                           node = self.formula_to_dense_graph(formula[2][2][2][2][2][2],
                                                                              quantifer_node)
                                       if formula[2][2][2][2][2][2][0] == formula[2][2][2][2][2][0]:
                                           number = 1
                                           for variable in formula[2][2][2][2][2][2][1]:
                                               variable_node = self.create_variable_node(
                                                   variable, quantifer_node)
                                               quantifer_node.children.append(variable_node)
                                           if formula[2][2][2][2][2][2][2][0] != formula[2][2][2][2][2][2][0]:
                                               node = self.formula_to_dense_graph(formula[2][2][2][2][2][2][2],
                                                                                  quantifer_node)
                                           if formula[2][2][2][2][2][2][2][0] == formula[2][2][2][2][2][2][0]:
                                               number = 1
                                               for variable in formula[2][2][2][2][2][2][2][1]:
                                                   variable_node = self.create_variable_node(
                                                       variable, quantifer_node)
                                                   quantifer_node.children.append(variable_node)
                                               if formula[2][2][2][2][2][2][2][2][0] != formula[2][2][2][2][2][2][2][0]:
                                                   node = self.formula_to_dense_graph(formula[2][2][2][2][2][2][2][2],
                                                                                      quantifer_node)
                                               if formula[2][2][2][2][2][2][2][2][0] == formula[2][2][2][2][2][2][2][0]:
                                                   number = 1
                                                   for variable in formula[2][2][2][2][2][2][2][2][1]:
                                                       variable_node = self.create_variable_node(
                                                           variable, quantifer_node)
                                                       quantifer_node.children.append(variable_node)
                                                   if formula[2][2][2][2][2][2][2][2][2][0] != formula[2][2][2][2][2][2][2][2][0]:
                                                       node = self.formula_to_dense_graph(
                                                           formula[2][2][2][2][2][2][2][2][2],
                                                           quantifer_node)
                                                   if formula[2][2][2][2][2][2][2][2][2][0] == formula[2][2][2][2][2][2][2][2][0]:
                                                       number = 1
                                                       for variable in formula[2][2][2][2][2][2][2][2][2][1]:
                                                           variable_node = self.create_variable_node(
                                                               variable, quantifer_node)
                                                           quantifer_node.children.append(
                                                               variable_node)
                                                       if formula[2][2][2][2][2][2][2][2][2][2][0] != formula[2][2][2][2][2][2][2][2][2][0]:
                                                           node = self.formula_to_dense_graph(
                                                               formula[2][2][2][2][2][2][2][2][2][2],
                                                               quantifer_node)
                                                       if formula[2][2][2][2][2][2][2][2][2][2][0] == formula[2][2][2][2][2][2][2][2][2][0]:
                                                           number = 1
                                                           for variable in formula[2][2][2][2][2][2][2][2][2][2][1]:
                                                               variable_node = self.create_variable_node(variable, quantifer_node)
                                                               quantifer_node.children.append(variable_node)
                                                           node = self.formula_to_dense_graph(formula[2][2][2][2][2][2][2][2][2][2][2], quantifer_node)
                    else:
                        number = 0
                        node = self.formula_to_dense_graph(formula[2], quantifer_node)
                    quantifer_node.children.append(node)
                return quantifer_node

        if isinstance(formula[0], str) and \
                formula[0] in NEGATIVE_CONNECTIVE and len(formula) == 2:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                negative_node = self.create_negative_node(formula[0], parent)
                self.id2subterm[negative_node.id] = (formula, negative_node)

                node = self.formula_to_dense_graph(formula[1], negative_node)
                negative_node.children.append(node)
                return negative_node

        if isinstance(formula[1], str) and \
                formula[1] in (BINARY_CONNECTIVE | ASSOC_CONNECTIVE | EQUAL) \
                and len(formula) == 3:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                connective_node = self.create_connective_node(
                    formula[1], parent)
                self.id2subterm[connective_node.id] = (
                    formula, connective_node)
                left_node = self.formula_to_dense_graph(
                    formula[0], connective_node)
                connective_node.children.append(left_node)
                right_node = self.formula_to_dense_graph(
                    formula[2], connective_node)
                connective_node.children.append(right_node)
                return connective_node

        if isinstance(formula[0], str) and \
                re.match(FUNCTOR_PATTERN, formula[0]) and len(formula) == 2:
            if self.check_merge(formula, parent):
                return self.merge_sub(formula, parent)
            else:
                functor_node = self.create_functor_node(formula[0], parent)
                self.id2subterm[functor_node.id] = (formula, functor_node)
                for argument in formula[1]:
                    argument_node = self.formula_to_dense_graph(
                        argument, functor_node)
                    functor_node.children.append(argument_node)
                return functor_node

    def convert(self, formula, rename):
        Node.rest_id()
        self.formula_to_dense_graph(formula)
     #   negative_nodes = [node for node in self.graph if node.name in NEGATIVE_CONNECTIVE]
        function_nodes = [node for node in self.graph if re.match(
                FUNCTOR_PATTERN, node.name)]
        connective_nodes1 = [node for node in self.graph if node.name in BINARY_CONNECTIVE1]

       # for node in negative_nodes:
            #for child in node.children:
            #    if child in function_nodes:
             #       child.time = 1
              #  for child1 in child.children:
               #     if child.time == 0 and not re.match(VARIABLE_PATTERN, child1.name):
               #         node.children.append(child1)
                #        child1.parents.append(node)

       # for node in connective_nodes1:
       #     for i in range(len(node.children) - 1):
       #        node.children[i].children.append(node.children[i + 1])
       #        node.children[i + 1].parents.append(node.children[i])

        #for node in function_nodes:
         #   if len(node.children) > 1 and node.children[1] not in node.children[0].children and node.children[0] not in node.children[1].parents:
          #      node.children[0].children.append(node.children[1])
           #     node.children[1].parents.append(node.children[0])
            #    if len(node.children) > 2 and node.children[2] not in node.children[2].children and node.children[1] not in node.children[2].parents:
            #       node.children[1].children.append(node.children[2])
            #       node.children[2].parents.append(node.children[1])
             #      if len(node.children) > 3 and node.children[3] not in node.children[2].children and node.children[2] not in node.children[3].parents:
            #           node.children[2].children.append(node.children[3])
             #          node.children[3].parents.append(node.children[2])

        if rename:
            variable_nodes = [node for node in self.graph if re.match(
               VARIABLE_PATTERN, node.name)]
            for node in variable_nodes:
                node.name = 'VAR'
        return self.graph


def Draw(graph):
    MG = nx.MultiDiGraph()
    for node in graph.graph:
        for child in node.children:
            MG.add_edge(node.id, child.id)
    nx.draw_networkx(MG, pos=nx.nx_agraph.graphviz_layout(
        MG, root=0), with_labels=True)
    plt.show()
    return MG


A = fof_formula_transformer("fof(l72_asympt_1, axiom,  (! [A] :  (m2_subset_1(A, k1_numbers, k5_numbers) =>  (r1_xxreal_0(aa1, A) => r1_xxreal_0(k4_power(k10_real_1(k2_nat_1(A, aa1), A), A), k4_power(k10_real_1(k2_nat_1(A, 2), k2_nat_1(A, aa1)), k2_nat_1(A, aa1)))) ) ) ).")
print(Graph(A).graph)
# [<0>: token ! | parents  | children 1 2 | scoped  | quantified variable 1, <1>: token VAR | parents 0 3 4 | children  | scoped 0 | quantifer 0, <2>: token => | parents 0 | children 3 4 | scoped 0 | , <3>: token v1_xboole_0 | parents 2 | children 1 | scoped 0 | , <4>: token = | parents 2 | children 1 5 | scoped 0 | , <5>: token k1_xboole_0 | parents 4 | children  | scoped 0 | ]

'''
if number == 0:
    quantifer_node = self.create_quantifer_node(formula[0], parent)  
    self.id2subterm[quantifer_node.id] = (formula, quantifer_node)  
for variable in formula[1]:
    variable_node = self.create_variable_node(
        variable, quantifer_node)  
    quantifer_node.children.append(variable_node)  
if formula[2][0] == formula[0]:
    number = 1
    for variable in formula[2][1]:
        variable_node = self.create_variable_node(
            variable, quantifer_node)  
        quantifer_node.children.append(variable_node)  
    node = self.formula_to_dense_graph(formula[2][2], quantifer_node)  
else:
    number = 0
    node = self.formula_to_dense_graph(formula[2], quantifer_node)
quantifer_node.children.append(node)
return quantifer_node


        
'''