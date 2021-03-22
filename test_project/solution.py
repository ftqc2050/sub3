import numpy as np
import cirq


def _to_id(num_qubits, ops):
    if not ops:
        return [cirq.I(q) for q in cirq.LineQubit.range(num_qubits)]
    return ops


def matrix_to_operations(matrix: np.ndarray, precise:bool) -> cirq.OP_TREE:
    num_qubits = int(np.log2(len(matrix)))
    qs = cirq.LineQubit.range(num_qubits)

    if len(matrix) == 2:
        res = [g(*qs) for g in
               cirq.single_qubit_matrix_to_gates(matrix, tolerance=1e-8)]
    elif len(matrix) == 4:
        res = cirq.two_qubit_matrix_to_operations(qs[0], qs[1], matrix, allow_partial_czs=False)
    elif len(matrix) == 8:
        qs = cirq.LineQubit.range(num_qubits)
        res = cirq.three_qubit_matrix_to_operations(qs[0], qs[1], qs[2], matrix, atol=1e-8)
    else:
        res = []
    return _to_id(num_qubits, res)
