from lark import Lark, Transformer

fof_parser = Lark(r"""
    annotated_formula: "fof(" name "," formula_role ","  fof_formula ")."
    
    name: NAME
    NAME: UPPER_LOW_ALPHA_NUMERIC+
    
    formula_role: FORMULA_ROLE
    FORMULA_ROLE: "axiom"
    
    ?fof_formula: unitary_formula | binary_formula
    
    ?unitary_formula: literal | type_bool | quantified_formula | negative "(" fof_formula ")" | constant   
    
    ?binary_formula: assoc_formula | non_assoc_formula
    
    non_assoc_formula: "(" fof_formula binary_connective fof_formula ")"
    binary_connective: BINARY_CONNECTIVE
    
    assoc_formula: "(" fof_formula assoc_connective fof_formula ")"
    assoc_connective: ASSOC_CONNECTIVE
    
    quantified_formula: "(" quantifier variable_list fof_formula ")"
    quantifier: QUANTIFIER
    variable_list: "[" variable ( "," variable )* "]" ":"
    
    ?literal: atom | negative "(" atom ")" |  negative atom | "(" atom ")"
    negative: NEGATIVE
    
    atom: predicate "(" term_argument ")" | term equal term | "(" term equal term ")" | negative term | "(" term ")" | term equal "(" term ")"
    predicate: PREDICATE
    equal: EQUAL
     
    ?term: functional_term | variable | constant | atom | "(" atom ")" | | "(" functional_term ")"
    functional_term: functor "(" term_argument ")"
    term_argument: term ("," term)*
    functor: FUNCTOR
    variable: VARIABLE
    constant: CONSTANT
    
    PREDICATE: LOWER_ALPHA LOW_ALPHA_NUMERIC*
    FUNCTOR: LOWER_ALPHA LOW_ALPHA_NUMERIC*
    VARIABLE: UPPER_ALPHA UPPER_ALPHA_NUMERIC*
    CONSTANT: NUMERIC+ | LOWER_ALPHA LOW_ALPHA_NUMERIC*
    
    ?type_bool: type_true | type_false | "(" type_true ")" | "(" type_false ")"
    type_true: TYPE_TRUE
    type_false: TYPE_FALSE
    TYPE_TRUE: "$true"
    TYPE_FALSE: "$false"
    
    EQUAL: "=" | "|" | "&" | "<=>" | "=>" | "!="
    QUANTIFIER: "!" | "?"
    NEGATIVE: "~"
    BINARY_CONNECTIVE: "<=>" | "=>"
    ASSOC_CONNECTIVE : "&" | "|"
    
    LOW_ALPHA_NUMERIC : LOWER_ALPHA | NUMERIC | "_"
    UPPER_ALPHA_NUMERIC: UPPER_ALPHA | NUMERIC | "_"
    UPPER_LOW_ALPHA_NUMERIC : UPPER_ALPHA | LOWER_ALPHA | NUMERIC | "_"
    LOWER_ALPHA : "a" .. "z"
    UPPER_ALPHA : "A" .. "Z"
    NUMERIC : "0" .. "9" 
    %ignore " "
    """, start='annotated_formula')


class Transform(Transformer):
    annotated_formula = lambda self, a: a[2]
    name = lambda self, a: a[0][:]
    formula_role = lambda self, a: a[0][:]
    fof_formula = lambda self, a: a
    unitary_formula = lambda self, a: a
    binary_formula = lambda self, a: a
    assoc_formula = lambda self, a: a
    non_assoc_formula = lambda self, a: a
    quantified_formula = lambda self, a: a
    literal = lambda self, a: a
    atom = lambda self, a: a
    term = lambda self, a: a
    term_argument = lambda self, a: a
    functional_term = lambda self, a: a
    variable_list = lambda self, a: a
    type_bool = lambda self, a: a
    constant = lambda self, a: a[0][:]
    variable = lambda self, a: a[0][:]
    predicate = lambda self, a: a[0][:]
    functor = lambda self, a: a[0][:]
    quantifier = lambda self, a: a[0][:]
    negative = lambda self, a: a[0][:]
    binary_connective = lambda self, a: a[0][:]
    assoc_connective = lambda self, a: a[0][:]
    equal = lambda self, a: a[0][:]
    type_true = lambda self, a: a[0][:]
    type_false = lambda self, a: a[0][:]


def fof_formula_transformer(formula):
    fomula_pharse = fof_parser.parse(formula)
    formula_tree = Transform().transform(fomula_pharse)
    return formula_tree

#A = fof_formula_transformer("fof(t162_funct_1, axiom, (v1_relat_1(sk262))&((v2_relat_1(sk262))&((v1_funct_1(sk262))))).")
#print(A[2][2])

