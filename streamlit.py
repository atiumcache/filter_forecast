import streamlit as st

import sys
sys.path.append('../')
import numpy as np
import scipy

from Implementations.algorithms.TimeDependentBeta import TimeDependentAlgo
from Implementations.resamplers.resamplers import PoissonResample,NBinomResample
from Implementations.solvers.DeterministicSolvers import LSODACalvettiSolver,LSODASolver,LSODASolverSEIARHD
from Implementations.perturbers.perturbers import MultivariatePerturbations
from utilities.Utils import Context,ESTIMATION
from functools import partial


def run_algo():        
    algo = TimeDependentAlgo(integrator = LSODASolver(),
                            perturb = MultivariatePerturbations(hyper_params={"h":0.5,"sigma1":0.05,"sigma2":0.1}),
                            resampler = NBinomResample(),
                            ctx=Context(sankey_indices = [],
                                        population=int(pop_input),
                                        state_size = 4,
                                        weights = np.ones(num_of_particles),
                                        seed_loc=[1],
                                        seed_size=0.005,
                                        forward_estimation=1,
                                        rng=np.random.default_rng(),
                                        particle_count=num_of_particles,
                                        run_sankey=False))

    algo.initialize(params={
    "beta":ESTIMATION.VARIABLE,
    "gamma":0.06,
    "mu":0.004,
    "q":0.1,
    "eta":1/7,
    "std":10,
    "R":50,
    "hosp":ESTIMATION.STATIC,
    "L":90,
    "D":10,
    }
    ,priors={"beta":partial(algo.ctx.rng.uniform,0.1,0.6), 
            "gamma":partial(algo.ctx.rng.uniform,0,1/7),
            "eta":partial(algo.ctx.rng.uniform,1/15,1/3),
            "hosp":partial(algo.ctx.rng.uniform,5,20),
            "D":partial(algo.ctx.rng.uniform,0,20)
            })


    #algo.print_particles()
    algo.run(csv_file, time_steps)



st.header("Particle Filter")

num_of_particles = st.select_slider(
    'How many particles?',
    options = [10, 50, 100, 500, 1000, 5000, 10000]
)

pop_input = st.text_input("State population:")

time_steps = st.slider("Time Steps:", min_value=1, max_value=365)

csv_file = st.file_uploader("Choose data file.")

if st.button("Run Filter", type="primary"):
    run_algo()

    
