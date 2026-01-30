from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
import numpy as np
from qiskit.algorithms.optimizers import COBYLA

def run_qaoa_allocation():
    
    algorithm_globals.random_seed = 42

   
    qp = QuadraticProgram(name="Disaster_Resource_Allocation")

    qp.binary_var('A')
    qp.binary_var('B')
    qp.binary_var('C')

    qp.maximize(linear=[12, 10, 2])

 
    qp.linear_constraint([1, 1, 1], sense='LE', rhs=2, name='max_regions')

    sampler = Sampler()
    
    optimizer = COBYLA(maxiter=100)
    qaoa = QAOA(optimizer=optimizer, sampler=sampler, reps=1)

    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)

   
    allocation = result.variables_dict
    total_score = result.fval

    print("\n🚨 QAOA Resource Allocation Result:")
    print("Optimal Allocation:", allocation)
    print("Total Allocation Score:", total_score)

    return allocation, total_score



if __name__ == "__main__":
    run_qaoa_allocation()
