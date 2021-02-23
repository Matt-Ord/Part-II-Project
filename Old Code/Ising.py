import numpy as np
from scipy.constants import k

class IsingSimulation():
    pass
class IsingState():

    def getNextState(self):
        pass
def array_map(x):
    return np.array(list(map(f, x)))

class IsingState2D(IsingState):
    def __init__(self, state, temperature):
        self.currentState = np.array(state)
        self.xDim = self.currentState.size
        self.yDim = self.currentState[0].size
        self.temperature = 0
        self.exchangeEnergy = 0

    def getState(self, coordinate):
        return self.currentState[x % self.xDim,y % self.yDim]

    def getNeighbouringStates(self,coordinate):
        x, y = coordinate
        return np.array([self.getState((x + 1 ,y)),
                            self.getState((x - 1 ,y)),
                            self.getState((x ,y + 1)),
                            self.getState((x ,y - 1))], dtype = bool)

    def _generateEnergyDictionary(self, current, neighbours):
        energyDict = {}
        for state in [True, False]:
            for numberOfPositiveNeighbours in range(4 + 1):
                energyDict[(state, numberOfPositiveNeighbours)] = calculateEnergy(self, currentState, neighbouringStates)
        exchangeEnergy = 1
        positiveNeighbours = np.sum(neighbours)
        isingEnergy = - exchangeEnergy #* (1 if  else -1) * (positiveNeighbours - neighbours.size)
        return isingEnergy
    
    def _calculateEnergy(self, currentState, numberOfPositiveNeighbours):
        numberOfNeighbours = 4
        localEnergy = - self.exchangeEnergy * (1 if currentState else -1) * (numberOfPositiveNeighbours - numberOfNeighbours)
        return localEnergy
    
    def getEnergy(self, state, neighbours):
        positiveNeighbours = np.sum(neighbouringStates, axis = -1)
        return self.calculateEnergy(state, positiveNeighbours)

    def getProbabilityOfTransition(self, coordinate):
        #If flipping decreases energy then flip
        #Else flip with a probability exp(-E/KT)
        neighbours = self.getNeighbouringStates(x)
        state = self.getState(coordinate)

        energy  = self.getEnergy(state, neighbours)
        
        ##If flipping would cause a reduction in energy, flip
        if(energy > 0):
            return 1        
        
        return np.exp( - self.getEnergy(coordinate) / (self.temperature * k))

    def getNextState(self, coordinate):
        pOfTransition = self.getProbabilityOfTransition(coordinates)
        hasFlipped = np.greater(np.random.rand(), pOfTransition)
        nextState  = np.logical_xor(hasFlipped, self.getState(coordinate))
        return nextState

print(np.array([1,2,3])[3 % 3])
if __name__ == '__main__':
    state = IsingState2D([[True,True,False],[True,True,False],[True,True,False]], 1)
    #print(state.getEnergy((1,1)))
    
