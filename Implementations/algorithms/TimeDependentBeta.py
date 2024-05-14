
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

class TimeDependentAlgo(Algorithm): 
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

        for _ in range(self.ctx.particle_count): 
            '''Setup the particles at t = 0'''
            p_params = params.copy()
            for _,(key,val) in enumerate(self.ctx.estimated_params.items()):
                if((p_params[key] == ESTIMATION.STATIC) or (p_params[key] == ESTIMATION.VARIABLE)): 
                    p_params[key] = priors[key]()

            seeds = []
            for _ in range(len(self.ctx.seed_loc)):
                seeds.append(self.ctx.rng.uniform(0,self.ctx.seed_size*self.ctx.population))

            state = np.concatenate((np.array([self.ctx.population],dtype=np.float_),[0 for _ in range(self.ctx.state_size-1)])) 
            for i in range(len(seeds)):

                state[self.ctx.seed_loc[i]] += seeds[i]
                state[0] -= seeds[i]

            self.particles.append(Particle(param=p_params,state=state.copy(),observation=np.array([0 for _ in range(self.ctx.forward_estimation)])))    

    def forward_propagator(): 
        '''This function simulates the 7 days data to be '''


    @timing
    def run(self,data_path:str,runtime:int) ->None:
        '''The algorithms main run method, takes the time series data as a parameter and returns an output object encapsulating parameter and state values'''

        data1 = pd.read_csv(data_path).to_numpy()
        data1 = np.delete(data1,0,1)



        '''Arrays to hold all the output data'''
        eta_quantiles = []

        state = []
        LL = []
        ESS = []
        gamma_quantiles = []

        state_quantiles = []
        beta_quantiles = []
        beta = []
        eta = []
        observations = []
        gamma = []

        while(self.ctx.clock.time < runtime): 

            #one step propagation 
            self.integrator.propagate(self.particles,self.ctx)
        
            obv = data1[self.ctx.clock.time, :]
            self.ctx.weights = self.resampler.compute_weights(self.ctx,obv,self.particles)
            self.particles = self.resampler.resample(self.ctx,self.particles)
            self.particles = self.perturb.randomly_perturb(self.ctx,self.particles) 
            
            

            print(f"Iteration: {self.ctx.clock.time}")
            self.ctx.clock.tick()

        pd.DataFrame(beta).to_csv('../datasets/average_beta.csv')
        pd.DataFrame(eta).to_csv('../datasets/average_eta.csv')
        pd.DataFrame(gamma).to_csv('../datasets/average_gamma.csv')

        pd.DataFrame(beta_quantiles).to_csv('../datasets/beta_quantiles.csv')
        pd.DataFrame(eta_quantiles).to_csv('../datasets/eta_quantiles.csv')
        pd.DataFrame(gamma_quantiles).to_csv('../datasets/gamma_quantiles.csv')
        pd.DataFrame(observations).to_csv('../datasets/particle_observation.csv')

        pd.DataFrame(state).to_csv('../datasets/ESTIMATED_STATE.csv')            

        state_quantiles = np.array(state_quantiles)
        beta_quantiles = np.array(beta_quantiles)
        eta_quantiles = np.array(eta_quantiles)
        gamma_quantiles = np.array(gamma_quantiles)

        colors = cm.plasma(np.linspace(0, 1, 12)) # type: ignore



        