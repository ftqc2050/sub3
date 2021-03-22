# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
from typing import List, Optional

import cirq
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from attr import dataclass
from cirq.testing import assert_allclose_up_to_global_phase


@dataclass
class JudgeLogEntry:
    max_score: int
    actual_score: int
    task: str
    msgs: str = ""

    def __str__(self):
        pad = "-" * int((100 - len(self.task)) / 2)
        return f"""{pad}{self.task}{pad}
{self.msgs}
Result: {self.actual_score} / {self.max_score}"""


@dataclass
class JudgeLog:
    entries: List[JudgeLogEntry] = []

    def results(self):
        total = sum(e.actual_score for e in self.entries)
        total_max = sum(e.max_score for e in self.entries)
        lines = "\n".join(str(e) for e in self.entries)
        return f"""
{lines}
{"=" * 100}
Total score: {total:.2f} / {total_max} points!
"""


@pytest.fixture(scope="session")
def judge_log():
    log = JudgeLog()
    yield log
    print(log.results())


def test_simple_identity(judge_log):
    result = JudgeLogEntry(max_score=2, actual_score=0, task="Simple identity check.")
    _score_and_log(np.eye(2), judge_log, 1, result, n_qubits=1,  min_two_qubit=0)


@pytest.mark.parametrize('gate', [
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.H,
    cirq.S,
    cirq.T,
])
def test_single_qubit_gates(judge_log, gate):
    multiplier = 2
    result = JudgeLogEntry(max_score=multiplier * 2, actual_score=0,
                           task=f"Single qubit gate {gate}")
    qs = cirq.LineQubit.range(1)
    input = cirq.unitary(gate(*qs))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=1, min_two_qubit=0)


@pytest.mark.parametrize('gate', [
    cirq.CZ,
    cirq.CX,
    cirq.XX,
    cirq.YY,
    cirq.ZZ,
    cirq.IdentityGate(num_qubits=2),
])
def test_two_qubit_gates(judge_log, gate):
    multiplier = 4
    result = JudgeLogEntry(max_score=multiplier * 2, actual_score=0,
                           task=f"Two-qubit gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(2))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=2,  min_two_qubit=1)


@pytest.mark.parametrize('gate', [
    cirq.CCX,
    cirq.CSWAP,
    cirq.CCZ,
    cirq.IdentityGate(num_qubits=3),
])
def test_three_qubit_gates(judge_log, gate):
    multiplier = 8
    result = JudgeLogEntry(max_score=multiplier * 2, actual_score=0,
                           task=f"Three-qubit gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(3))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits=3)


@pytest.mark.parametrize("n_qubits", [n for n in range(4, 10)])
def test_identities(judge_log, n_qubits):
    multiplier = 1
    gate = cirq.IdentityGate(num_qubits=n_qubits)
    result = JudgeLogEntry(max_score=multiplier * 2, actual_score=0,
                           task=f"{n_qubits}-qubit identity gate {gate}")

    input = cirq.unitary(gate(*(cirq.LineQubit.range(n_qubits))))

    _score_and_log(input, judge_log, multiplier, result, n_qubits, min_two_qubit=0)


@pytest.mark.parametrize("n_qubits", [n for n in range(1, 10)])
def test_randos(judge_log, n_qubits):
    multiplier = 2 ** n_qubits * 2
    s = np.random.RandomState(seed=1231242)

    result = JudgeLogEntry(max_score=multiplier * 2, actual_score=0,
                           task=f"{n_qubits}-qubit random unitary")

    input = cirq.testing.random_unitary(2**n_qubits, random_state=s)

    _score_and_log(input, judge_log, multiplier, result, n_qubits)


def _score_and_log(input, judge_log, multiplier, result, n_qubits, min_two_qubit=np.inf):
    try:
        _score_input(input, result, multiplier, n_qubits, min_two_qubit)
    except Skipped:
        result.msgs += "skipped\n"
    except BaseException as ex:
        result.msgs += (f"✘\n {type(ex)}: {str(ex)}"
                        f"\n {traceback.format_exc()}")
    finally:
        judge_log.entries.append(result)


def _score_input(input: np.ndarray, result: JudgeLogEntry, multiplier: int, n_qubits: int, min_two_qubit):
    from test_project import matrix_to_operations
    # see Shende et al.
    theoretical_lower_bound = 1/4 * (4 ** n_qubits - 3 * n_qubits - 1)
    lower_bound = min(theoretical_lower_bound, min_two_qubit)
    # max double points for the two-qubit gate count
    max_two_qubit_gate_count_score = result.max_score
    result.max_score += 2 * max_two_qubit_gate_count_score

    # TODO do scoring for locality
    # cirq.GridQubit.rect()
    # cirq.testing.ValidatingTestDevice(qs)

    for task, precise in [("equal up to global phase", False), ("precise equality", True)]:
        result.msgs += f"\n{task}: "
        response = matrix_to_operations(input, precise=precise)
        if response == NotImplemented:
            pytest.skip()
        response_circuit = cirq.Circuit(response)
        response_unitary = response_circuit.unitary()
        assert_allclose_up_to_global_phase(response_unitary, input, atol=1e-8)
        more_than_two_qubit_gates = len([op for op in response_circuit.all_operations() if cirq.num_qubits(op) > 2])
        assert more_than_two_qubit_gates == 0, f"There are {more_than_two_qubit_gates} gates that are bigger than 2 qubit gates!"
        num_two_qubit_gates = len([op for op in response_circuit.all_operations() if cirq.num_qubits(op) == 2])

        if num_two_qubit_gates == 0:
            result.actual_score += max_two_qubit_gate_count_score
        else:
            result.actual_score += lower_bound / num_two_qubit_gates * max_two_qubit_gate_count_score
        result.actual_score += multiplier
        result.msgs += f"✔ Two-qubit gates: {num_two_qubit_gates} Lower bound: {lower_bound}"

