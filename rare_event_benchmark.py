#!/usr/bin/env python3
"""First-passage benchmarks for Extremely Rare Events.

--------
python rare_event_benchmark.py --D 5 --T 7 --boundary reflect
python rare_event_benchmark.py --D 5 --T 7 --quantum --shots 20000
python rare_event_benchmark.py --sweep 3 5 7 --slack 2 --quantum --csv results.csv
python rare_event_benchmark.py --D 1074 --T 1075 --boundary unrestricted --shots 0
python rare_event_benchmark.py --self-test
python rare_event_benchmark.py --self-test --quantum

Models
------
unrestricted: negative positions allowed; stop at D.
reflect: at zero, both coin outcomes lead to one; stop at D.
stay: a backward attempt at zero stays at zero; stop at D.
Coin bit 1 means forward, with probability p. X_0 = 0; D >= 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import unittest
from decimal import Decimal, localcontext
from itertools import product
from pathlib import Path

BOUNDARIES = ('unrestricted', 'reflect', 'stay')


def validate(D, T, p, boundary):
    if type(D) is not int or D < 1 or type(T) is not int or T < 0:
        raise ValueError('D must be a positive integer; T a nonnegative integer.')
    prob = Decimal(str(p))
    if not prob.is_finite() or not 0 <= prob <= 1:
        raise ValueError('p must be finite and between 0 and 1.')
    if boundary not in BOUNDARIES:
        raise ValueError(f'boundary must be one of {BOUNDARIES}')
    return prob


def advance(x, bit, D, boundary):
    """Single transition, including absorption; bit is 0 or 1."""
    if x == D:
        return D
    if x == 0 and boundary == 'reflect':
        return 1
    if x == 0 and boundary == 'stay' and not bit:
        return 0
    return x + (1 if bit else -1)


def first_passage_distribution(D, T, p='0.5', boundary='reflect', precision=80):
    """Return [P(tau=0), ..., P(tau=T)] without float underflow."""
    prob = validate(D, T, p, boundary)
    if type(precision) is not int or precision < 16:
        raise ValueError('precision must be an integer >= 16.')
    with localcontext() as ctx:
        ctx.prec = precision
        q = 1 - prob
        mass = {0: Decimal(1)}
        first = [Decimal(0)] * (T + 1)
        for t in range(1, T + 1):
            following = {}
            for x, weight in mass.items():
                if x + (T - t + 1) < D:
                    continue
                for bit, step_prob in ((0, q), (1, prob)):
                    if not step_prob:
                        continue
                    y = advance(x, bit, D, boundary)
                    contribution = weight * step_prob
                    if y == D:
                        first[t] += contribution
                    else:
                        following[y] = following.get(y, Decimal(0)) + contribution
            mass = following
        return first


def first_passage_probability(D, T, p='0.5', boundary='reflect', precision=80):
    with localcontext() as ctx:
        ctx.prec = precision
        return sum(first_passage_distribution(D, T, p, boundary, precision), Decimal(0))


def ballot_probability(D, T, p='0.5', precision=80):
    """Independent formula ONLY for the unrestricted classical walk."""
    prob = validate(D, T, p, 'unrestricted')
    with localcontext() as ctx:
        ctx.prec = precision
        if T < D or prob == 0:
            return Decimal(0)
        if prob == 1:
            return Decimal(1)
        return sum((Decimal(D) / t * math.comb(t, (t + D) // 2)
                    * prob ** ((t + D) // 2)
                    * (1 - prob) ** ((t - D) // 2)
                    for t in range(D, T + 1, 2)), Decimal(0))


def path_hits(bits, D, boundary):
    x = 0
    for bit in bits:
        x = advance(x, bit, D, boundary)
        if x == D:
            return True
    return False


def wilson_interval(hits, shots):
    """Approximate two-sided 95% binomial interval; valid at zero hits too."""
    z = 1.959963984540054
    estimate = hits / shots
    denominator = 1 + z*z / shots
    center = (estimate + z*z / (2*shots)) / denominator
    radius = z * math.sqrt(estimate*(1-estimate)/shots + z*z/(4*shots*shots)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def monte_carlo(D, T, p='0.5', boundary='reflect', shots=20000, seed=42):
    prob = float(validate(D, T, p, boundary))
    if type(shots) is not int or shots <= 0:
        raise ValueError('shots must be a positive integer.')
    rng = random.Random(seed)
    hits = 0
    for _ in range(shots):
        x = 0
        for _ in range(T):
            x = advance(x, rng.random() < prob, D, boundary)
            if x == D:
                hits += 1
                break
    return {'hits': hits, 'shots': shots, 'estimate': hits/shots,
            'wilson_95': wilson_interval(hits, shots)}


def build_path_circuit(D, T, p='0.5', boundary='reflect'):
    """Return a unitary Qiskit path-preparation + hit-oracle circuit.

    Qubits 0..T-1 are chronological coin choices; qubit T is the flag.
    Qiskit displays bit strings in reverse qubit order. No measurements or
    resets are included, so the circuit has an inverse for future QAE work.
    """
    prob = float(validate(D, T, p, boundary))
    if T > 10:
        raise ValueError('Enumerated quantum oracle is limited to T <= 10.')
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(T + 1, name='classical_path_hit')
    angle = 2 * math.asin(math.sqrt(prob))
    for i in range(T):
        circuit.ry(angle, i)
    marked = 0
    for bits in product((0, 1), repeat=T):
        if not path_hits(bits, D, boundary):
            continue
        marked += 1
        zeros = [i for i, bit in enumerate(bits) if not bit]
        for i in zeros:
            circuit.x(i)
        if T == 1:
            circuit.cx(0, T)
        else:
            circuit.mcx(list(range(T)), T)
        for i in zeros:
            circuit.x(i)
    circuit.metadata = {'marked_strings': marked, 'boundary': boundary}
    return circuit


def quantum_benchmark(D, T, p='0.5', boundary='reflect', shots=20000, seed=42):
    from qiskit import transpile, __version__ as qiskit_version
    from qiskit.quantum_info import Statevector
    circuit = build_path_circuit(D, T, p, boundary)
    state = Statevector.from_instruction(circuit)
    flagged = float(state.probabilities(qargs=[T])[1])
    compiled = transpile(circuit, basis_gates=['u', 'cx'],
                         optimization_level=1, seed_transpiler=seed)
    result = {'qiskit_version': qiskit_version,
              'gate_basis': ['u', 'cx'], 'optimization_level': 1,
              'statevector_probability': flagged,
              'qubits': circuit.num_qubits,
              'marked_strings': circuit.metadata['marked_strings'],
              'logical_depth': circuit.depth(),
              'compiled_depth': compiled.depth(),
              'compiled_gate_count': compiled.size(),
              'compiled_cx_count': compiled.count_ops().get('cx', 0)}
    if shots:
        state.seed(seed)
        counts = state.sample_counts(shots, qargs=[T])
        hits = int(counts.get('1', 0))
        result['simulated_shots'] = {'hits': hits, 'shots': shots,
                                    'estimate': hits/shots,
                                    'wilson_95': wilson_interval(hits, shots)}
    return result


def benchmark(D, T, p, boundary, shots, seed, precision, quantum):
    exact = first_passage_probability(D, T, p, boundary, precision)
    with localcontext() as ctx:
        ctx.prec = precision
        log10 = str(exact.log10()) if exact else '-Infinity'
    row = {'D': D, 'T': T, 'p': str(p), 'boundary': boundary,
           'seed': seed, 'decimal_precision': precision,
           'reference_probability': str(exact), 'log10_probability': log10}
    if boundary == 'unrestricted':
        row['ballot_reference'] = str(ballot_probability(D, T, p, precision))
    if shots:
        row['classical_mc'] = monte_carlo(D, T, p, boundary, shots, seed)
    if quantum:
        row['quantum'] = quantum_benchmark(D, T, p, boundary, shots, seed)
        with localcontext() as ctx:
            ctx.prec = precision
            row['quantum']['absolute_error_vs_reference'] = str(abs(
                Decimal(str(row['quantum']['statevector_probability'])) - exact))
    return row


def run_tests(include_quantum=False):
    """Analytic and exhaustive checks independent of the DP recurrence."""
    class Validation(unittest.TestCase):
        def test_unrestricted_formula(self):
            for D in range(1, 6):
                for T in range(9):
                    for p in ('0', '0.2', '0.5', '1'):
                        a = first_passage_probability(D, T, p, 'unrestricted')
                        b = ballot_probability(D, T, p)
                        self.assertLessEqual(abs(a-b), Decimal('1e-70'))

        def test_boundary_path_enumeration(self):
            p = Decimal('0.3')
            for boundary in BOUNDARIES:
                for D in (1, 2, 4):
                    for T in range(7):
                        total = Decimal(0)
                        for bits in product((0, 1), repeat=T):
                            x, hit = 0, False
                            weight = Decimal(1)
                            for bit in bits:
                                weight *= p if bit else 1-p
                                if not hit:
                                    proposed = x + (1 if bit else -1)
                                    x = (abs(proposed) if boundary == 'reflect'
                                         else max(0, proposed) if boundary == 'stay'
                                         else proposed)
                                    hit = x == D
                            if hit:
                                total += weight
                        self.assertEqual(total, first_passage_probability(D, T, p, boundary))

        def test_short_deadline_reflection(self):
            p = Decimal('0.3')
            for D in range(2, 9):
                expected = p**(D-1) * (1+(1-p)*(1+(D-2)*p))
                self.assertEqual(expected, first_passage_probability(D, D+2, p))
            self.assertEqual(first_passage_probability(1, 1, '0', 'reflect'), 1)
            self.assertEqual(first_passage_probability(1, 5, '0', 'stay'), 0)

        def test_extreme_tail(self):
            value = first_passage_probability(1100, 1100, '0.5', 'unrestricted')
            self.assertGreater(value, 0)
            self.assertEqual(float(value), 0.0)
            reference = ballot_probability(1100, 1100, '0.5')
            self.assertLess(abs(value/reference - 1), Decimal('1e-25'))

        def test_invalid_inputs(self):
            for args in ((0, 1, '.5', 'reflect'), (1, -1, '.5', 'stay'),
                         (1, 2, 'NaN', 'reflect'), (1, 2, '1.1', 'stay')):
                with self.assertRaises(ValueError):
                    first_passage_probability(*args)

        @unittest.skipUnless(include_quantum, 'optional quantum validation')
        def test_quantum_and_inverse(self):
            from qiskit.quantum_info import Statevector
            for boundary in BOUNDARIES:
                for D, T, p in ((1, 0, '.5'), (1, 3, '.2'), (2, 4, '.3'),
                                (3, 5, '.5'), (2, 3, '0'), (2, 3, '1')):
                    qc = build_path_circuit(D, T, p, boundary)
                    state = Statevector.from_instruction(qc)
                    expected = float(first_passage_probability(D, T, p, boundary))
                    self.assertAlmostEqual(float(state.probabilities([T])[1]), expected, places=12)
                    recovered = state.evolve(qc.inverse())
                    self.assertAlmostEqual(float(recovered.probabilities()[0]), 1.0, places=12)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Validation))
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--D', type=int, default=5)
    parser.add_argument('--T', type=int, default=7)
    parser.add_argument('--p', default='0.5')
    parser.add_argument('--boundary', choices=BOUNDARIES, default='reflect')
    parser.add_argument('--shots', type=int, default=20000, help='0 skips sampling')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=int, default=80)
    parser.add_argument('--quantum', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--sweep', nargs='+', type=int, help='D values; T = D + slack')
    parser.add_argument('--slack', type=int, default=2)
    parser.add_argument('--csv', type=Path, help='write rows; nested metrics are JSON strings')
    args = parser.parse_args()
    try:
        if args.self_test:
            return 0 if run_tests(args.quantum) else 1
        if args.shots < 0:
            raise ValueError('shots cannot be negative.')
        pairs = [(D, D+args.slack) for D in args.sweep] if args.sweep else [(args.D, args.T)]
        rows = [benchmark(D, T, args.p, args.boundary, args.shots, args.seed,
                          args.precision, args.quantum) for D, T in pairs]
        if args.csv:
            with args.csv.open('w', newline='') as handle:
                fields = list(dict.fromkeys(k for row in rows for k in row))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v
                                     for k, v in row.items()})
        print(json.dumps(rows, indent=2))
        return 0
    except ImportError as exc:
        parser.error(f'{exc}. Install optional quantum dependency: pip install qiskit')
    except (ValueError, ArithmeticError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
