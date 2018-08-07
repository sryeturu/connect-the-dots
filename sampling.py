import numpy as np

class Sampler:
    
    def __init__(self, vals, distribtion):
        self.vals = vals
        self.distribtion = distribtion
    
    
    def get_sample(self):
        return np.random.choice(a=self.vals, p=self.distribtion)