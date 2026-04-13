from pytket.circuit import Circuit

from .base import Base


class cirqTesting(Base):
    def __init__(self):
        super().__init__()

    def run_circ(self, circuit: Circuit):
        '''
        Runs circuit on cirq simulator and returns counts
        '''
        pass
