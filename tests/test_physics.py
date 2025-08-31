import unittest
from src.physics.quantum_field import QuantumFieldTheoryEngine, Hadronizer

class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.qft = QuantumFieldTheoryEngine(grid_size=(5,5))
        self.hadronizer = Hadronizer()

    def test_inject_energy(self):
        self.qft.inject_energy((2,2), 10, 'electron')
        self.assertEqual(self.qft.fields['electron'][2,2], 10)

    def test_hadronizer_combinations(self):
        quarks = [Hadronizer.Quark(c) for c in ['R','G','B']]
        stable = self.hadronizer.find_stable_combinations(quarks)
        self.assertEqual(len(stable), 1)
