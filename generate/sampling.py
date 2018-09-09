import numpy as np

class Sampler:
    
    def __init__(self, vals, distribtion=None):
        '''
        distribution is either a array like that sums to 1 with distribtion[2] representing the probability
        of sampling the 3rd image. Defaults to random if none. 
        '''
        
        if len(vals) == 0:
            raise ValueError('must have > 0 values') 
            
        self.vals = vals
        self.distribtion = distribtion 
    
    def get_sample(self):
        if self.distribtion == None:
            return self.vals[np.random.randint(0, len(self.vals))]
        else:
            return np.random.choice(a=self.vals, p=self.distribtion)