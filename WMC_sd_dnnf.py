
import itertools
import random
import re



'''
2.1 Logical Representations
The code uses custom classes for:

Var: Propositional variables

Not, And, Or: Logical operators

Const: Constants (⊤ for True, ⊥ for False)

A pretty-printing function pretty_print() is also provided for better readability of formulas.

This code cell is trying to convert the string input into the parse infix formula.
'''


def convert_logic_notation(input_string):
    converted_string = (input_string.replace('∧', '&')
                                    .replace('∨', '|')
                                    .replace('¬', '~')
                                    .replace('->', '->')
                                    .replace('→', '->')
                                    .replace('<->', '<->')
                                    .replace('↔', '<->')
                                    .replace('=', '<->'))  # Allow = as equiv
    return converted_string


def construct_formula(input_formula):
    converted_formula = convert_logic_notation(input_formula)
    print("Converted to PyEDA syntax:", converted_formula)
    formula = converted_formula
    return formula

def formula_to_string(formula):
    return str(formula)


class Var:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Not:
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"¬{self.child}"

    def __eq__(self, other):
        return isinstance(other, Not) and self.child == other.child

    def __hash__(self):
        return hash(('not', self.child))


class And:
    def __init__(self, children):
        self.children = children

    def __repr__(self):
        return f"({' ∧ '.join(map(str, self.children))})"

    def __eq__(self, other):
        return isinstance(other, And) and set(self.children) == set(other.children)

    def __hash__(self):
        return hash(('and', frozenset(self.children)))


class Or:
    def __init__(self, children):
        self.children = children

    def __repr__(self):
        return f"({' ∨ '.join(map(str, self.children))})"

    def __eq__(self, other):
        return isinstance(other, Or) and set(self.children) == set(other.children)

    def __hash__(self):
        return hash(('or', frozenset(self.children)))


class Implies:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} → {self.right})"

    def __eq__(self, other):
        return isinstance(other, Implies) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('implies', self.left, self.right))


class Iff:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} ↔ {self.right})"

    def __eq__(self, other):
        return isinstance(other, Iff) and (
            (self.left == other.left and self.right == other.right) or
            (self.left == other.right and self.right == other.left)
        )

    def __hash__(self):
        return hash(('iff', frozenset([self.left, self.right])))

def parse_infix_formula(s):
    # Tokenize input including implication/equivalence
    tokens = re.findall(r'<->|->|[~&|()]|\w+', s)

    def precedence(op):
        return {
            '~': 4,
            '&': 3,
            '|': 2,
            '->': 1,
            '<->': 0
        }.get(op, -1)

    def is_right_associative(op):
        return op in ('->', '<->')

    def to_ast(output):
        token = output.pop()
        if isinstance(token, str) and token == '~':
            return Not(to_ast(output))
        elif token == '&':
            right = to_ast(output)
            left = to_ast(output)
            return And([left, right])
        elif token == '|':
            right = to_ast(output)
            left = to_ast(output)
            return Or([left, right])
        elif token == '->':
            right = to_ast(output)
            left = to_ast(output)
            return Implies(left, right)
        elif token == '<->':
            right = to_ast(output)
            left = to_ast(output)
            return Iff(left, right)
        elif isinstance(token, Var):
            return token
        else:
            return token

    # Shunting Yard Algorithm
    output = []
    stack = []

    for tok in tokens:
        if re.match(r'\w+', tok):
            output.append(Var(tok))
        elif tok == '~':
            stack.append(tok)
        elif tok in ('&', '|', '->', '<->'):
            while (stack and stack[-1] not in ('(', ')') and
                   (precedence(stack[-1]) > precedence(tok) or
                   (precedence(stack[-1]) == precedence(tok) and not is_right_associative(tok)))):
                output.append(stack.pop())
            stack.append(tok)
        elif tok == '(':
            stack.append(tok)
        elif tok == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('

    while stack:
        output.append(stack.pop())

    return to_ast(output)


def assign_weights(expr1, mode="default", default_value=0.5):
    """
    Assign weights to literals in the formula.
    mode = "default" -> use default_value for all
    mode = "random" -> use random weights
    mode = "manual" -> prompt user for each variable
    """
    import random

    variables = sorted(list(get_variables(expr1)))
    weights = {}

    for var in variables:
        if mode == "manual":
            while True:
                try:
                    w = float(input(f"Enter weight for {var} (probability of {var} being True): "))
                    if 0 <= w <= 1:
                        break
                    print("Weight must be between 0 and 1.")
                except ValueError:
                    print("Invalid number.")
        elif mode == "random":
            w = round(random.uniform(0.1, 0.9), 2)
        else:  # default
            w = default_value

        weights[var] = w

    return weights




# Update your parse_infix_formula to parse '→' and '↔' as Imply and Equiv respectively.
# Then in eliminate_implications function:

def eliminate_implications(node):
    if isinstance(node, Var):
        return node

    elif isinstance(node, Not):
        return Not(eliminate_implications(node.child))

    elif isinstance(node, And):
        return And([eliminate_implications(c) for c in node.children])

    elif isinstance(node, Or):
        return Or([eliminate_implications(c) for c in node.children])

    elif isinstance(node, Implies):
        # A → B ≡ ¬A ∨ B
        left = eliminate_implications(node.left)
        right = eliminate_implications(node.right)
        return Or([Not(left), right])

    elif isinstance(node, Iff):
        # A ↔ B ≡ (¬A ∨ B) ∧ (¬B ∨ A)
        left = eliminate_implications(node.left)
        right = eliminate_implications(node.right)
        left_to_right = Or([Not(left), right])
        right_to_left = Or([Not(right), left])
        return And([left_to_right, right_to_left])

    else:
        raise ValueError(f"Unknown node type: {node}")



def push_negations(expr1):
    # Push negations inside so that negation applies only to variables
    if isinstance(expr1, Var):
        return expr1
    if isinstance(expr1, Not):
        child = expr1.child
        if isinstance(child, Var):
            return expr1  # negation on var is OK in NNF
        if isinstance(child, Not):
            return push_negations(child.child)  # ¬¬A = A
        if isinstance(child, And):
            # ¬(A ∧ B) = ¬A ∨ ¬B
            return push_negations(Or([Not(c) for c in child.children]))
        if isinstance(child, Or):
            # ¬(A ∨ B) = ¬A ∧ ¬B
            return push_negations(And([Not(c) for c in child.children]))
    if isinstance(expr1, And):
        return And([push_negations(c) for c in expr1.children])
    if isinstance(expr1, Or):
        return Or([push_negations(c) for c in expr1.children])

    return expr1

import itertools
import random
import re

class Const:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return "⊤" if self.value else "⊥"
    def __eq__(self, other):
        return isinstance(other, Const) and self.value == other.value
    def __hash__(self):
        return hash(('const', self.value))


def pretty_print(expr1):
    if isinstance(expr1, Var):
        return expr1.name
    elif isinstance(expr1, Not):
        return f"¬{pretty_print(expr1.child)}"
    elif isinstance(expr1, And):
        return "(" + " ∧ ".join(pretty_print(c) for c in expr1.children) + ")"
    elif isinstance(expr1, Or):
        return "(" + " ∨ ".join(pretty_print(c) for c in expr1.children) + ")"
    elif expr1 is True:
        return "⊤"
    elif expr1 is False:
        return "⊥"
    else:
        return str(expr1)

'''
3.1 Function: simplify(expr)
This function recursively simplifies formulas by:
Flattening nested AND/OR structures
Removing redundant constants (True/False)
Deduplicating identical sub-expressions
Short-circuiting when an absorbing element is found (False for AND, True for OR)
Enhancement:
It adds a seen set to eliminate duplicate operands in both conjunctions and disjunctions.
'''

def simplify(expr1):
    if isinstance(expr1, And):
        new_children = []
        seen = set()
        for c in expr1.children:
            sc = simplify(c)
            if sc is False:
                return False  # Short-circuit
            elif sc is not True:
                key = repr(sc)
                if key not in seen:
                    seen.add(key)
                    new_children.append(sc)
        if not new_children:
            return True
        elif len(new_children) == 1:
            return new_children[0]
        return And(new_children)

    elif isinstance(expr1, Or):
        new_children = []
        seen = set()
        for c in expr1.children:
            sc = simplify(c)
            if sc is True:
                return True  # Short-circuit
            elif sc is not False:
                key = repr(sc)
                if key not in seen:
                    seen.add(key)
                    new_children.append(sc)
        if not new_children:
            return False
        elif len(new_children) == 1:
            return new_children[0]
        return Or(new_children)

    elif isinstance(expr1, Not):
        child = simplify(expr1.child)
        if child is True:
            return False
        elif child is False:
            return True
        return Not(child)

    else:
        return expr1  # Variables or already simplified atoms




#Checks if the formula is satisfiable by brute-force evaluation
#over all possible truth assignments (based on Cartesian product of [True, False] values).

def is_satisfiable(expr1):
    vars = list(get_variables(expr1))
    for vals in itertools.product([False, True], repeat=len(vars)):
        assignment = dict(zip(vars, vals))
        if eval_formula(expr1, assignment):
            #print("True")
            return True
    return False




#Returns a set of all propositional variables present in a formula.

def get_variables(node):
    if isinstance(node, Var):
        return {node.name}
    if isinstance(node, Not):
        return get_variables(node.child)
    if isinstance(node, And) or isinstance(node, Or):
        vars_set = set()
        for c in node.children:
            vars_set |= get_variables(c)
        return vars_set
    return set()



#Applies variable substitution (var = value) to the entire formula,
#simplifying where applicable.

def restrict(expr1, var, value):
    """Substitute variable with value: var=True or var=False"""
    if isinstance(expr1, Var):
        return expr1 if expr1.name != var else (True if value else False)
    elif isinstance(expr1, Not):
        child = restrict(expr1.child, var, value)
        return Not(child) if child not in [True, False] else not child
    elif isinstance(expr1, And):
        new_children = [restrict(c, var, value) for c in expr1.children]
        new_children = [c for c in new_children if c is not True]
        if any(c is False for c in new_children):
            return False
        return And(new_children) if new_children else True
    elif isinstance(expr1, Or):
        new_children = [restrict(c, var, value) for c in expr1.children]
        new_children = [c for c in new_children if c is not False]
        if any(c is True for c in new_children):
            return True
        return Or(new_children) if new_children else False
    return expr1


'''
Ensures smoothness in disjunctions. That is, both sides of the OR operation
must mention the same set of variables. Missing variables are padded using
tautologies like (p ∨ ¬p) via a helper add_tautology().'''

def smooth_or(phi, psi):
    phi_vars = get_variables(phi)
    psi_vars = get_variables(psi)

    all_vars = phi_vars | psi_vars

    def add_tautology(expr1, vars_to_add):
        if not vars_to_add:
            return expr1
        tautologies = [Or([Var(v), Not(Var(v))]) for v in vars_to_add]
        return And([expr1] + tautologies)

    phi = add_tautology(phi, all_vars - phi_vars)
    psi = add_tautology(psi, all_vars - psi_vars)

    return Or([phi, psi])


'''The main function for converting a formula to SD-DNNF recursively'''


def to_sd_dnnf(expr1):
    if isinstance(expr1, Var) or (isinstance(expr1, Not) and isinstance(expr1.child, Var)):
        return expr1

    elif isinstance(expr1, And):
        children = [to_sd_dnnf(c) for c in expr1.children]
        # Check decomposability
        seen_vars = set()
        for c in children:
            vars_c = get_variables(c)
            if seen_vars & vars_c:
                shared = list(seen_vars & vars_c)[0]
                p = shared
                pos = restrict(expr1, p, True)
                neg = restrict(expr1, p, False)
                return simplify(smooth_or(
                    simplify(And([Var(p), to_sd_dnnf(pos)])),
                    simplify(And([Not(Var(p)), to_sd_dnnf(neg)]))
                ))
            seen_vars |= vars_c
        return simplify(And(children))

    elif isinstance(expr1, Or):
        children = [to_sd_dnnf(c) for c in expr1.children]
        # Enforce determinism (pairwise disjoint)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                both = And([children[i], children[j]])
                if is_satisfiable(both):
                    shared_vars = get_variables(children[i]) & get_variables(children[j])
                    if shared_vars:
                        shared = list(shared_vars)[0]
                        p = shared
                        # Restrict only the conflicting subformula (the two children)
                        conflict = Or([children[i], children[j]])
                        pos = restrict(conflict, p, True)
                        neg = restrict(conflict, p, False)
                        return simplify(smooth_or(
                            simplify(And([Var(p), to_sd_dnnf(pos)])),
                            simplify(And([Not(Var(p)), to_sd_dnnf(neg)]))
                        ))
                    elif not shared_vars:
                        # Get shared vars
                        vars_i = get_variables(children[i])
                        vars_j = get_variables(children[j])

                        # Pick a pivot variable that exists in either child
                        p = list(vars_i | vars_j)[0]
                        conflict = Or([children[i], children[j]])
                        pos = restrict(conflict, p, True)
                        neg = restrict(conflict, p, False)
                        return simplify(smooth_or(
                            simplify(And([Var(p), to_sd_dnnf(pos)])),
                            simplify(And([Not(Var(p)), to_sd_dnnf(neg)]))
                        ))


        # If no conflicts found, just return the OR of children
        return simplify(Or(children))

        '''# Apply smoothing
        result = children[0]
        for c in children[1:]:
            result = smooth_or(result, c)
        return result'''

    return expr1


# --- Weighted Model Counting ---
def wmc(node, weights):
    if isinstance(node, Var):
        w = weights.get(node.name, 0.5)
        return w
    if isinstance(node, Not):
        w = 1 - wmc(node.child, weights)
        return w
    if isinstance(node, And):
        results = [wmc(c, weights) for c in node.children]
        return eval_product(results)
    if isinstance(node, Or):
        results = [wmc(c, weights) for c in node.children]
        return sum(results)


def wmc_(node, weights):
    if isinstance(node, Var):
        # Directly check if there's a weight for 'a'
        return weights.get(node.name, 0.5)

    if isinstance(node, Not):
        child = node.child
        if isinstance(child, Var):
            # Look up weight of the negated literal, e.g. '¬a'
            negated_name = f"¬{child.name}"
            return weights.get(negated_name, 0.5)
        else:
            # If it's a full expression (not just a literal), recursively compute
            return 1 - wmc(child, weights)

    if isinstance(node, And):
        results = [wmc(c, weights) for c in node.children]
        return eval_product(results)

    if isinstance(node, Or):
        results = [wmc(c, weights) for c in node.children]
        return sum(results)

    raise ValueError(f"Unknown node type: {type(node)}")


def eval_product(lst):
    result = 1.0
    for v in lst: result *= v
    return result

# --- Truth Table WMC ---
def extract_vars(node):
    if isinstance(node, Var): return {node.name}
    if isinstance(node, Not): return extract_vars(node.child)
    if isinstance(node, And) or isinstance(node, Or):
        result = set()
        for c in node.children:
            result |= extract_vars(c)
        return result
    return set()

def eval_formula(expr, assignment):
    if isinstance(expr, Var): return assignment[expr.name]
    if isinstance(expr, Not): return not eval_formula(expr.child, assignment)
    if isinstance(expr, And): return all(eval_formula(c, assignment) for c in expr.children)
    if isinstance(expr, Or): return any(eval_formula(c, assignment) for c in expr.children)


def exact_wmc_truth_table(expr, weights):
    vars = sorted(list(extract_vars(expr)))
    total = 0.0
    for vals in itertools.product([False, True], repeat=len(vars)):
        assignment = dict(zip(vars, vals))
        prob = 1.0
        for v in vars:
            prob *= weights[v] if assignment[v] else (1 - weights[v])
        if eval_formula(expr, assignment):
            total += prob
    return total




# --- SampleSAT Estimator ---
def sample_sat(expr1, weights, samples=10000):
    vars = sorted(list(extract_vars(expr1)))
    count = 0
    for _ in range(samples):
        assignment = {v: random.random() < weights[v] for v in vars}
        if eval_formula(expr1, assignment):
            count += 1
    return count / samples


from WMC_cnf import evaluate_CNF_formula_wmc



# --- Run it all ---
if __name__ == "__main__":
    # Get user input
    input_formula = input("Enter a logical formula (e.g., (A → B) ∧ (B → (C ∨ D))): ")
    print('Original Formula:', input_formula)

    # Processing steps
    converted = convert_logic_notation(input_formula)
    formula = construct_formula(converted)
    Parsed_formula = parse_infix_formula(formula)
    formula = Parsed_formula

    # Eliminate implications and push negations
    expr_no_implies = eliminate_implications(Parsed_formula)
    print("Without Implications:", expr_no_implies)

    expr_nnf = push_negations(expr_no_implies)
    print("Parsed Formula in NNF:", expr_nnf)

    # Assign weights
    weights = assign_weights(expr_nnf, mode="random", default_value=0.6)
    print("Weights:", weights)

    # Compile to sd-DNNF
    sd_dnnf_formula = to_sd_dnnf(expr_nnf)
    print("sd-DNNF (structurally):", sd_dnnf_formula)
    print("sd-DNNF (prettified):", pretty_print(sd_dnnf_formula))

    # Perform weighted model counting
    wmc_val = wmc_(sd_dnnf_formula, weights)
    print("WMC (sd-DNNF):", wmc_val)

    exact_val = exact_wmc_truth_table(sd_dnnf_formula, weights)
    print("WMC (Truth Table):", exact_val)

    estimated = sample_sat(sd_dnnf_formula, weights, samples=10000)
    print("WMC (SampleSAT):", estimated)

    evaluate_CNF_formula_wmc(input_formula, weights)






