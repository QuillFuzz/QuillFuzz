import ast
import logging

from .ast_ops import is_qubit_guppy, get_array_size_guppy, CircuitRenamer, QiskitMainTransformer

def _validate_qiskit_if_test_usage(tree):
    issues = []

    class MainVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_main = False

        def visit_FunctionDef(self, node):
            if node.name == 'main':
                previous_state = self.in_main
                self.in_main = True
                self.generic_visit(node)
                self.in_main = previous_state
                return
            self.generic_visit(node)

        def visit_Call(self, node):
            if self.in_main and isinstance(node.func, ast.Attribute) and node.func.attr == 'if_test':
                bad_keywords = [kw.arg for kw in node.keywords if kw.arg in {'body', 'true_body', 'qubits', 'clbits'}]
                if len(node.args) > 1 or bad_keywords:
                    issues.append(
                        'legacy positional-body if_test usage is unsupported; use the context-manager form `with qc.if_test((clbit, val)):` instead'
                    )
            self.generic_visit(node)

    MainVisitor().visit(tree)
    if issues:
        raise ValueError('; '.join(dict.fromkeys(issues)))

def assemble_qiskit(files, output_path, unique_index=0):
    all_imports = []
    renamed_bodies = []
    main_call_blocks = []
    main_resource_reqs = []

    # Add required import for diff testing
    diff_test_import = ast.ImportFrom(
        module='src.utils.diff_testing',
        names=[ast.alias(name='qiskitTesting', asname=None)],
        level=0
    )
    all_imports.append(diff_test_import)

    # Always add required qiskit classes used by assembled master main
    required_qiskit_import = ast.ImportFrom(
        module='qiskit',
        names=[
            ast.alias(name='QuantumCircuit', asname=None),
            ast.alias(name='QuantumRegister', asname=None),
            ast.alias(name='ClassicalRegister', asname=None),
        ],
        level=0,
    )
    all_imports.append(required_qiskit_import)
    
    # Global tracking for max resources
    global_max_qubits = 1
    global_max_clbits = 1

    # Track seen imports to avoid duplication
    seen_imports = {ast.unparse(required_qiskit_import)}

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r") as f:
                source = f.read()
            
            tree = ast.parse(source)
            prefix = f"c{i}_"
            file_global_funcs = set()
            
            # Transform main function to use shared resources
            transformer = QiskitMainTransformer()
            tree = transformer.visit(tree)

            # Reject legacy if_test branch bodies so assembled output only uses
            # the context-manager style that matches the official Qiskit docs.
            _validate_qiskit_if_test_usage(tree)
            
            # Update global max requirements
            global_max_qubits = max(global_max_qubits, transformer.max_qubits)
            global_max_clbits = max(global_max_clbits, transformer.max_clbits)
            main_resource_reqs.append((transformer.max_qubits, transformer.max_clbits))
            
            # First pass: collect global function names
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    file_global_funcs.add(node.name)
            
            top_level_nodes = []
            
            # Second pass: process nodes and imports
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # If it's an import from qiskit, ensure we have the required classes
                    if isinstance(node, ast.ImportFrom) and node.module == 'qiskit':
                        existing_names = {alias.name for alias in node.names}
                        for required in ['QuantumCircuit', 'QuantumRegister', 'ClassicalRegister']:
                            if required not in existing_names:
                                node.names.append(ast.alias(name=required, asname=None))
                    
                    code_str = ast.unparse(node)
                    if code_str not in seen_imports:
                        seen_imports.add(code_str)
                        all_imports.append(node)
                    continue
                
                # Verify passed code doesn't have if __name__ == "__main__" blocks that might run
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                     if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                         continue

                # Filter out top-level calls to main() since we will invoke it manually with appropriate arguments
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                     if isinstance(node.value.func, ast.Name) and node.value.func.id == "main":
                         continue

                top_level_nodes.append(node)
                
            # Rename
            renamer = CircuitRenamer(prefix, file_global_funcs)
            new_nodes = []
            for node in top_level_nodes:
                new_node = renamer.visit(node)
                new_nodes.append(new_node)
                
            renamed_bodies.extend(new_nodes)
            
            # Store the call to the renamed main function
            req_qubits, req_clbits = main_resource_reqs[-1]
            local_qr_name = f"{prefix}qr"
            local_cr_name = f"{prefix}cr"
            local_qc_name = f"{prefix}qc"

            local_qr_init = ast.Assign(
                targets=[ast.Name(id=local_qr_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='QuantumRegister', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=max(1, req_qubits)),
                        ast.Constant(value=f"{prefix}q"),
                    ],
                    keywords=[]
                )
            )

            local_cr_init = ast.Assign(
                targets=[ast.Name(id=local_cr_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='ClassicalRegister', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=max(1, req_clbits)),
                        ast.Constant(value=f"{prefix}c"),
                    ],
                    keywords=[]
                )
            )

            local_qc_init = ast.Assign(
                targets=[ast.Name(id=local_qc_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='QuantumCircuit', ctx=ast.Load()),
                    args=[
                        ast.Name(id=local_qr_name, ctx=ast.Load()),
                        ast.Name(id=local_cr_name, ctx=ast.Load()),
                    ],
                    keywords=[]
                )
            )

            local_main_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=f"{prefix}main", ctx=ast.Load()),
                    args=[
                        ast.Name(id=local_qc_name, ctx=ast.Load()),
                        ast.Subscript(
                            value=ast.Name(id=local_qr_name, ctx=ast.Load()),
                            slice=ast.Slice(
                                lower=None,
                                upper=ast.Constant(value=max(1, req_qubits)),
                                step=None,
                            ),
                            ctx=ast.Load(),
                        ),
                        ast.Subscript(
                            value=ast.Name(id=local_cr_name, ctx=ast.Load()),
                            slice=ast.Slice(
                                lower=None,
                                upper=ast.Constant(value=max(1, req_clbits)),
                                step=None,
                            ),
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[]
                )
            )

            compose_local = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='qc', ctx=ast.Load()),
                        attr='compose',
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=local_qc_name, ctx=ast.Load())],
                    keywords=[
                        ast.keyword(
                            arg='qubits',
                            value=ast.Subscript(
                                value=ast.Name(id='qr', ctx=ast.Load()),
                                slice=ast.Slice(
                                    lower=None,
                                    upper=ast.Constant(value=max(1, req_qubits)),
                                    step=None,
                                ),
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.keyword(
                            arg='clbits',
                            value=ast.Subscript(
                                value=ast.Name(id='cr', ctx=ast.Load()),
                                slice=ast.Slice(
                                    lower=None,
                                    upper=ast.Constant(value=max(1, req_clbits)),
                                    step=None,
                                ),
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.keyword(arg='inplace', value=ast.Constant(value=True)),
                    ],
                )
            )

            main_call_blocks.append([
                local_qr_init,
                local_cr_init,
                local_qc_init,
                local_main_call,
                compose_local,
            ])
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    # Construct output module
    new_module = ast.Module(body=[], type_ignores=[])
    

    # Add collected imports
    new_module.body.extend(all_imports)
    
    # Add all renamed bodies
    new_module.body.extend(renamed_bodies)
    
    # Create master main function
    
    # Setup global resources
    # qr = QuantumRegister(global_max_qubits, 'q')
    qr_init = ast.Assign(
        targets=[ast.Name(id='qr', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id='QuantumRegister', ctx=ast.Load()),
            args=[
                ast.Constant(value=global_max_qubits),
                ast.Constant(value='q')
            ],
            keywords=[]
        )
    )
    
    # cr = ClassicalRegister(global_max_clbits, 'c')
    cr_init = ast.Assign(
        targets=[ast.Name(id='cr', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id='ClassicalRegister', ctx=ast.Load()),
            args=[
                ast.Constant(value=global_max_clbits),
                ast.Constant(value='c')
            ],
            keywords=[]
        )
    )
    
    # qc = QuantumCircuit(qr, cr)
    qc_init = ast.Assign(
        targets=[ast.Name(id='qc', ctx=ast.Store())],
        value=ast.Call(
            func=ast.Name(id='QuantumCircuit', ctx=ast.Load()),
            args=[
                ast.Name(id='qr', ctx=ast.Load()),
                ast.Name(id='cr', ctx=ast.Load())
            ],
            keywords=[]
        )
    )

    # Return qc at the end
    return_qc = ast.Return(value=ast.Name(id='qc', ctx=ast.Load()))

    master_body = [qr_init, cr_init, qc_init]
    
    if main_call_blocks:
        for block in main_call_blocks:
            master_body.extend(block)
    else:
        master_body.append(ast.Pass())

    master_body.append(return_qc)

    master_main = ast.FunctionDef(
        name='main',
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=master_body,
        decorator_list=[]
    )
    
    new_module.body.append(master_main)

    # Write to file
    try:
        ast.fix_missing_locations(new_module)
        with open(output_path, "w") as f:
            f.write(ast.unparse(new_module))
        return True
    except Exception as e:
        logging.error(f"Error writing output to {output_path}: {e}")
        return False

def assemble_guppy(files, output_path, unique_index=0):
    all_imports = []
    renamed_bodies = []
    main_funcs = []

    # Track seen imports to avoid duplication
    # We store the unparsed string as key
    seen_imports = set(["from src.utils.diff_testing import guppyTesting"]) # Pre-add our required import

    # Add required import for diff testing
    diff_test_import = ast.ImportFrom(
        module='src.utils.diff_testing',
        names=[ast.alias(name='guppyTesting', asname=None)],
        level=0
    )
    all_imports.append(diff_test_import)

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r") as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            prefix = f"c{i}_"
            file_global_funcs = set()
            
            # First pass: collect global function names
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    file_global_funcs.add(node.name)
            
            top_level_nodes = []
            
            # Second pass: process nodes
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    code_str = ast.unparse(node)
                    if code_str not in seen_imports:
                        seen_imports.add(code_str)
                        all_imports.append(node)
                    continue
                    
                top_level_nodes.append(node)
                
            # Rename
            renamer = CircuitRenamer(prefix, file_global_funcs)
            new_nodes = []
            for node in top_level_nodes:
                # Filter out top-level executions like 'main.compile()' 
                # or 'enable_experimental_features()' which we handle globally
                if isinstance(node, ast.Expr):
                    code_str = ast.unparse(node)
                    if "compile" in code_str or "enable_experimental_features" in code_str:
                        continue
                
                new_node = renamer.visit(node)
                new_nodes.append(new_node)
                
            renamed_bodies.extend(new_nodes)
            
            # Find the renamed main function definition to extract arguments
            for node in new_nodes:
                if isinstance(node, ast.FunctionDef) and node.name == prefix + "main":
                    main_funcs.append(node)
                    break
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    # Construct output module
    new_module = ast.Module(body=[], type_ignores=[])
    
    # Add collected imports
    new_module.body.extend(all_imports)
    
    # Add guppylang.enable_experimental_features() if likely needed
    # (We check if 'guppylang' is in imports)
    has_guppy = any("guppylang" in imp for imp in seen_imports)
    if has_guppy:
        enable_expr = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='guppylang', ctx=ast.Load()),
                    attr='enable_experimental_features',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )
        )
        new_module.body.append(enable_expr)
        
    # Add all renamed bodies
    new_module.body.extend(renamed_bodies)
    
    # Create master main
    main_body = []
    
    if main_funcs:
        for main_node in main_funcs:
            m_name = main_node.name
            call_args = []
            result_stmts = []
            
            for arg in main_node.args.args:
                arg_name = arg.arg
                ann = arg.annotation
                if not ann:
                    continue
                
                # Create unique local variable name
                local_var = f"{m_name}_{arg_name}"
                
                setup_expr = None
                measure_call = None
                
                if is_qubit_guppy(ann):
                    # var = qubit()
                    setup_expr = ast.Call(
                        func=ast.Name(id='qubit', ctx=ast.Load()),
                        args=[], keywords=[]
                    )
                    # measure(var)
                    measure_call = ast.Call(
                        func=ast.Name(id='measure', ctx=ast.Load()),
                        args=[ast.Name(id=local_var, ctx=ast.Load())],
                        keywords=[]
                    )
                    
                else:
                    arr_size = get_array_size_guppy(ann)
                    if arr_size is not None:
                        # var = array(qubit() for _ in range(size))
                        range_call = ast.Call(
                            func=ast.Name(id='range', ctx=ast.Load()),
                            args=[ast.Constant(value=arr_size)],
                            keywords=[]
                        )
                        
                        qubit_call = ast.Call(
                            func=ast.Name(id='qubit', ctx=ast.Load()),
                            args=[], keywords=[]
                        )
                        
                        gen_exp = ast.GeneratorExp(
                            elt=qubit_call,
                            generators=[
                                ast.comprehension(
                                    target=ast.Name(id='_', ctx=ast.Store()),
                                    iter=range_call,
                                    ifs=[],
                                    is_async=0
                                )
                            ]
                        )
                        
                        setup_expr = ast.Call(
                            func=ast.Name(id='array', ctx=ast.Load()),
                            args=[gen_exp],
                            keywords=[]
                        )
                        
                        # measure_array(var)
                        measure_call = ast.Call(
                            func=ast.Name(id='measure_array', ctx=ast.Load()),
                            args=[ast.Name(id=local_var, ctx=ast.Load())],
                            keywords=[]
                        )

                if setup_expr:
                    # Assign setup
                    main_body.append(ast.Assign(
                        targets=[ast.Name(id=local_var, ctx=ast.Store())],
                        value=setup_expr
                    ))
                    
                    call_args.append(ast.Name(id=local_var, ctx=ast.Load()))
                    
                    if measure_call:
                        # result("m_name.arg_name", measure_call)
                        result_key = f"{m_name}.{arg_name}"
                        result_stmts.append(ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='result', ctx=ast.Load()),
                                args=[
                                    ast.Constant(value=result_key),
                                    measure_call
                                ],
                                keywords=[]
                            )
                        ))

            # Call the main function: m_name(args...)
            main_body.append(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id=m_name, ctx=ast.Load()),
                        args=call_args,
                        keywords=[]
                    )
                )
            )
            
            # Append measurement results AFTER the function call
            main_body.extend(result_stmts)

    else:
        main_body.append(ast.Pass())


    main_def = ast.FunctionDef(
        name='main',
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=main_body,
        decorator_list=[ast.Name(id='guppy', ctx=ast.Load())],
        returns=ast.Constant(value=None, kind=None), 
        lineno=0
    )
    
    new_module.body.append(main_def)
    
    ast.fix_missing_locations(new_module)
    output_code = ast.unparse(new_module)
    
    with open(output_path, "w") as f:
        f.write(output_code)
    # print(f"Assembled {len(files)} files into {output_path}")


def assemble_pytket(files, output_path, unique_index=0):
    all_imports = []
    renamed_bodies = []
    main_builders = []

    required_import = ast.ImportFrom(
        module='pytket',
        names=[ast.alias(name='Circuit', asname=None)],
        level=0,
    )
    all_imports.append(required_import)
    seen_imports = {ast.unparse(required_import)}

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r") as f:
                source = f.read()

            tree = ast.parse(source)
            prefix = f"c{i}_"
            file_global_funcs = set()

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    file_global_funcs.add(node.name)

            top_level_nodes = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    code_str = ast.unparse(node)
                    if code_str not in seen_imports:
                        seen_imports.add(code_str)
                        all_imports.append(node)
                    continue

                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                        continue

                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == "main":
                        continue

                top_level_nodes.append(node)

            renamer = CircuitRenamer(prefix, file_global_funcs)
            new_nodes = []
            for node in top_level_nodes:
                new_node = renamer.visit(node)
                new_nodes.append(new_node)

            renamed_bodies.extend(new_nodes)

            renamed_main = prefix + "main"
            if any(isinstance(node, ast.FunctionDef) and node.name == renamed_main for node in new_nodes):
                main_builders.append(renamed_main)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    new_module = ast.Module(body=[], type_ignores=[])
    new_module.body.extend(all_imports)
    new_module.body.extend(renamed_bodies)

    builder_list = ", ".join(main_builders)
    master_main_src = f"""
def main():
    assembled = None
    builders = [{builder_list}]

    for builder in builders:
        try:
            sub_circuit = builder()
        except Exception:
            continue

        if assembled is None:
            assembled = sub_circuit
            continue

        try:
            args = list(assembled.qubits[:sub_circuit.n_qubits]) + list(assembled.bits[:sub_circuit.n_bits])
            assembled.add_circuit(sub_circuit, args)
        except Exception:
            continue

    if assembled is None:
        assembled = Circuit(1, 1)

    return assembled
"""
    new_module.body.append(ast.parse(master_main_src).body[0])

    try:
        ast.fix_missing_locations(new_module)
        with open(output_path, "w") as f:
            f.write(ast.unparse(new_module))
        return True
    except Exception as e:
        logging.error(f"Error writing output to {output_path}: {e}")
        return False


def assemble_pennylane(files, output_path, unique_index=0):
    all_imports = []
    renamed_bodies = []
    main_builders = []

    required_import = ast.parse("import pennylane as qml").body[0]
    all_imports.append(required_import)
    seen_imports = {ast.unparse(required_import)}

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r") as f:
                source = f.read()

            tree = ast.parse(source)
            prefix = f"c{i}_"
            file_global_funcs = set()

            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    file_global_funcs.add(node.name)

            top_level_nodes = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    code_str = ast.unparse(node)
                    if code_str not in seen_imports:
                        seen_imports.add(code_str)
                        all_imports.append(node)
                    continue

                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                        continue

                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == "main":
                        continue

                top_level_nodes.append(node)

            renamer = CircuitRenamer(prefix, file_global_funcs)
            new_nodes = []
            for node in top_level_nodes:
                new_node = renamer.visit(node)
                new_nodes.append(new_node)

            renamed_bodies.extend(new_nodes)

            renamed_main = prefix + "main"
            if any(isinstance(node, ast.FunctionDef) and node.name == renamed_main for node in new_nodes):
                main_builders.append(renamed_main)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    new_module = ast.Module(body=[], type_ignores=[])
    new_module.body.extend(all_imports)
    new_module.body.extend(renamed_bodies)

    builder_list = ", ".join(main_builders)
    master_main_src = f"""
def main():
    assembled = None
    builders = [{builder_list}]

    for builder in builders:
        try:
            candidate = builder()
        except Exception:
            continue

        if assembled is None:
            assembled = candidate
            continue

        if callable(candidate):
            assembled = candidate

    return assembled
"""
    new_module.body.append(ast.parse(master_main_src).body[0])

    try:
        ast.fix_missing_locations(new_module)
        with open(output_path, "w") as f:
            f.write(ast.unparse(new_module))
        return True
    except Exception as e:
        logging.error(f"Error writing output to {output_path}: {e}")
        return False


def assemble_cirq(files, output_path, unique_index=0):
    import cirq
    all_imports = []
    renamed_bodies = []
    main_builders = []

    required_import = ast.Import(names=[ast.alias(name='cirq', asname=None)])
    all_imports.append(required_import)
    seen_imports = {ast.unparse(required_import)}

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r") as f:
                source = f.read()

            tree = ast.parse(source)
            prefix = f"c{i}_"
            file_global_funcs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

            top_level_nodes = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    code_str = ast.unparse(node)
                    if code_str not in seen_imports:
                        seen_imports.add(code_str)
                        all_imports.append(node)
                    continue

                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                        continue

                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == "main":
                        continue

                top_level_nodes.append(node)

            renamer = CircuitRenamer(prefix, file_global_funcs)
            new_nodes = []
            for node in top_level_nodes:
                new_node = renamer.visit(node)
                new_nodes.append(new_node)

            renamed_bodies.extend(new_nodes)

            renamed_main = prefix + "main"
            if any(isinstance(node, ast.FunctionDef) and node.name == renamed_main for node in new_nodes):
                main_builders.append(renamed_main)

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    new_module = ast.Module(body=[], type_ignores=[])
    new_module.body.extend(all_imports)
    new_module.body.extend(renamed_bodies)

    builder_list = ", ".join(main_builders)
    master_main_src = f"""
def main():
    assembled = cirq.Circuit()
    builders = [{builder_list}]

    for builder in builders:
        try:
            candidate = builder()
        except Exception:
            continue

        if candidate is None:
            continue

        try:
            if isinstance(candidate, cirq.Circuit):
                circuit_to_add = candidate
            else:
                circuit_to_add = cirq.Circuit(candidate)
            # Remove any measurements from the candidate to avoid key conflicts
            ops_no_measure = [op for op in circuit_to_add.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)]
            clean_circuit = cirq.Circuit(ops_no_measure)
            assembled += clean_circuit
        except Exception:
            continue

    # Add a single consolidated measurement at the end
    if assembled.all_qubits():
        assembled.append(cirq.measure(*sorted(assembled.all_qubits()), key='result'))
    return assembled
"""
    new_module.body.append(ast.parse(master_main_src).body[0])

    try:
        ast.fix_missing_locations(new_module)
        with open(output_path, "w") as f:
            f.write(ast.unparse(new_module))
        return True
    except Exception as e:
        logging.error(f"Error writing output to {output_path}: {e}")
        return False

def assemble(files, output_path, unique_index=0, language='guppy'):
    if language == 'qiskit':
        return assemble_qiskit(files, output_path, unique_index)
    if language == 'cirq':
        return assemble_cirq(files, output_path, unique_index)
    if language == 'pytket':
        return assemble_pytket(files, output_path, unique_index)
    if language == 'pennylane':
        return assemble_pennylane(files, output_path, unique_index)
    
    return assemble_guppy(files, output_path, unique_index)
