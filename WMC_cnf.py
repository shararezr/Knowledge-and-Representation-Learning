from sympy import symbols, to_cnf
from sympy.logic.boolalg import Or, And, Not
from pysat.formula import CNF
from pysat.solvers import Solver
from sympy.logic.boolalg import Or, And, Not, Implies, Equivalent



import re

# --- Helper Functions ---

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

def convert_to_sympy_syntax(formula_str):
    formula_str = convert_logic_notation(formula_str)

    # Replace <-> (equivalence)
    formula_str = re.sub(r'(\w+)\s*<->\s*(\w+)', r'Equivalent(\1, \2)', formula_str)

    # Replace -> (implication), including nested expressions
    formula_str = re.sub(r'(\w+)\s*->\s*\(([^()]+|(?:\([^()]*\)))+\)', r'Implies(\1, \2)', formula_str)
    formula_str = re.sub(r'(\w+)\s*->\s*(\w+)', r'Implies(\1, \2)', formula_str)

    # Logical ops
    formula_str = formula_str.replace('∧', '&').replace('∨', '|').replace('¬', '~')

    return formula_str



def extract_variables(formula_str):
    """Extract unique variable names from the formula string."""
    return sorted(set(re.findall(r'\b[a-zA-Z_]\w*', formula_str)))






def sympy_to_pysat_clauses(expr, var_map):
    """Convert sympy CNF expression to PySAT-compatible list of clauses."""
    def to_lit(sym):
        sign = 1
        if isinstance(sym, Not):
            sym = sym.args[0]
            sign = -1
        return sign * var_map[str(sym)]

    clauses = []

    if isinstance(expr, And):
        for arg in expr.args:
            if isinstance(arg, Or):
                clauses.append([to_lit(x) for x in arg.args])
            else:
                clauses.append([to_lit(arg)])
    elif isinstance(expr, Or):
        clauses.append([to_lit(x) for x in expr.args])
    else:
        clauses.append([to_lit(expr)])

    return clauses

def compute_wmc_via_models(cnf, weights, var_map):
    """Compute Weighted Model Count using PySAT."""
    inv_var_map = {v: k for k, v in var_map.items()}
    wmc = 0.0

    with Solver(bootstrap_with=cnf.clauses) as solver:
        while solver.solve():
            model = solver.get_model()
            assignment = set(model)
            prob = 1.0

            for var_id in inv_var_map:
                name = inv_var_map[var_id]
                p = weights.get(name, 0.5)
                prob *= p if var_id in assignment else (1 - p)

            wmc += prob
            solver.add_clause([-lit for lit in model])  # Block current model

    return wmc

# --- Main Function ---

def evaluate_CNF_formula_wmc(input_formula, weights=None):
    """
    Given a logical formula string and optional weights, print CNF, clauses, and WMC.
    """
    print("Original Formula:", input_formula)

    input_formula = convert_logic_notation(input_formula)
    variables = extract_variables(input_formula)
    var_map = {v: i + 1 for i, v in enumerate(variables)}

    converted = convert_to_sympy_syntax(input_formula)
    symbols_dict = {v: symbols(v) for v in variables}
    print("Converted for eval:", converted)
    sympy_expr = eval(converted, {"Equivalent": Equivalent, "Implies": Implies, **symbols_dict})

    cnf_expr = to_cnf(sympy_expr, simplify=True)

    print("CNF (SymPy):", cnf_expr)

    clauses = sympy_to_pysat_clauses(cnf_expr, var_map)
    cnf = CNF()
    cnf.extend(clauses)

    print("PySAT CNF Clauses:", clauses)
    print("Variable Map:", var_map)

    if weights is None:
        weights = {v: 0.5 for v in variables}

    wmc = compute_wmc_via_models(cnf, weights, var_map)

    print("Weights:", weights)
    print("WMC:", wmc)
    return wmc



