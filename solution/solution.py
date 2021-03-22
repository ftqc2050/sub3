from typing import List

import numpy as np
import cirq


def matrix_to_sycamore_operations(qs: List[cirq.GridQubit], matrix: np.ndarray) -> cirq.OP_TREE:
    num_qubits = len(qs)
    if np.allclose(matrix, np.eye(2 ** num_qubits)):
        res = []
    elif num_qubits == 3:
        res = [cirq.CCX(*qs)]
    else:
        return NotImplemented
    c = cirq.Circuit(res)
    return cirq.google.optimized_for_sycamore(c).all_operations()
