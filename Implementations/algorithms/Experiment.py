
from Abstract.Algorithm import Algorithm
from Abstract.Integrator import Integrator
from Abstract.Perturb import Perturb
from Abstract.Resampler import Resampler
from utilities.Utils import *
from typing import Dict,Callable
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from utilities.Utils import Context

class ExperimentAlgo(Algorithm): 
    '''Main particle filtering algorithm as described in Calvetti et. al. '''
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler,ctx:Context) -> None:
        '''Constructor passes back to the parent, nothing fancy here'''
        super().__init__(integrator, perturb, resampler,ctx)
        
    def initialize(self,params:Dict[str,int],priors:Dict[str,Callable]) -> None:
        '''Initialize the parameters and their flags for estimation'''
        for _,(key,val) in enumerate(params.items()):
            if(val == ESTIMATION.STATIC): 
                self.ctx.estimated_params[key] = ESTIMATION.STATIC
            elif(val == ESTIMATION.VARIABLE): 
                self.ctx.estimated_params[key] = ESTIMATION.VARIABLE
            elif(val == ESTIMATION.STATIC_PER_LOCATION):
                self.ctx.estimated_params[key] = ESTIMATION.STATIC_PER_LOCATION

        for _ in range(self.ctx.particle_count): 
            '''Setup the particles at t = 0'''
            p_params = params.copy()
            for _,(key,val) in enumerate(self.ctx.estimated_params.items()):
                if((p_params[key] == ESTIMATION.STATIC) or (p_params[key] == ESTIMATION.VARIABLE)): 
                    p_params[key] = priors[key]()
                elif (p_params[key] == ESTIMATION.STATIC_PER_LOCATION):
                    p_params[key] = np.zeros(self.ctx.state_size)
                    for i in range(self.ctx.state_size):
                        p_params[key][i] = priors[key]()


            seeds = []
            for _ in range(len(self.ctx.seed_loc)):
                seeds.append(self.ctx.rng.uniform(0,self.ctx.seed_size))

            state = np.zeros(self.ctx.state_size)
            for i in range(len(seeds)):
                state[self.ctx.seed_loc[i]] += seeds[i]

            self.particles.append(Particle(param=p_params,state=state.copy(),observation=np.array([0 for _ in range(self.ctx.state_size)])))    

    def forward_propagator(): 
        '''This function simulates the 7 days data to be '''


    #@timing
    def run(self,data_path:str,runtime:int) ->None:
        '''The algorithms main run method, takes the time series data as a parameter and returns an output object encapsulating parameter and state values'''

        data1 = pd.read_csv(data_path).to_numpy()

        '''Arrays to hold all the output data'''
        r = []
        k = []
        state = []
        mu = []

        for t in range(runtime): 

            #one step propagation 
            self.integrator.propagate(self.particles,self.ctx)
        
            obv = data1[t, :]
            self.ctx.weights = self.resampler.compute_weights(self.ctx,obv,self.particles)
            self.particles = self.resampler.resample(self.ctx,self.particles)
            self.particles = self.perturb.randomly_perturb(self.ctx,self.particles) 

            print(f"Iteration: {t}")

            r_unavg = []
            k_unavg = []
            state_unavg = []
            mu_unavg = []

            for particle in self.particles:
                r_unavg.append(particle.param["r"])
                k_unavg.append(particle.param["k"])
                state_unavg.append(particle.state)
                mu_unavg.append(particle.param["mu"])

            r.append(np.average(np.array(r_unavg), weights=np.exp(self.ctx.weights)))
            k.append(np.average(np.array(k_unavg), axis=0 , weights=np.exp(self.ctx.weights)))
            state.append(np.average(np.array(state_unavg), axis=0, weights=np.exp(self.ctx.weights)))
            mu.append(np.average(np.array(mu_unavg), weights=np.exp(self.ctx.weights)))

            print(np.average(np.array(r_unavg), weights=np.exp(self.ctx.weights)))

        

        pd.DataFrame(r).to_csv('../datasets/r_quantiles.csv')
        pd.DataFrame(k).to_csv('../datasets/k_quantiles.csv')
        pd.DataFrame(state).to_csv('../datasets/state_quantiles.csv')
        pd.DataFrame(mu).to_csv('../datasets/mu_quantiles.csv')