import numpy as np
import time
import matplotlib.pyplot as plt

class IsingSimulation():
    def __init__(self, initialIsingState):
        self.firstIsingState = initialIsingState

    def getStateNumber(self, N):
        currentState = self.firstIsingState
        for x in range(N):
            currentState = currentState.getNextState()
        return currentState

    def getFirstNMagnetisations(self, N):
        currentState = self.firstIsingState
        magnetisations = []
        
        for x in range(N):
            magnetisations.append(currentState.getMagnetisation())
            currentState = currentState.getNextState()

        return magnetisations        
    
class NearestNeighbourIsingSimulation(IsingSimulation):
    
    def __init__(self, initialIsingState):
        if not isinstance(initialIsingState, NearestNeighbourIsingState):
            raise Exception('Not nearest neighbour ising state')

        super().__init__(initialIsingState)

class NearestNeighbourIsingSimulation2D(NearestNeighbourIsingSimulation):

    def __init__(self, temperature, xDim, yDim, initialState = None):
        initialState = np.ones((xDim, yDim), dtype=bool) if initialState == None else initialState

        initialIsingState = NearestNeighbourIsingState2D.generateFromArray(initialState, temperature)
        
        super().__init__(initialIsingState)

class IsingSimulationPlotter():

    def __init__(self, isingSimulation):
        self.simulation = isingSimulation

    def plotFirstNMagnetisations(self, N):
        magnetisations = self.simulation.getFirstNMagnetisations(N)
        stateNumbers   = range(N)
        plt.figure()
        plt.plot(stateNumbers, magnetisations)
        #plt.show()

class IsingState():
    def __init__(self, state, spinStateSolver, previousState = None):
        self.currentState = state
        self.spinStateSolver = spinStateSolver
        self.nextState = None
        self.previousState = previousState
    
    def getDimensions(self):
        return self.currentState.shape

    def getNumberOfStates(self):
        return self.currentState.size

    def getNumberOfSpinUp(self):
        return np.count_nonzero(self.currentState)
    
    def getMagnetisation(self):
        return 2 * self.getNumberOfSpinUp() - self.getNumberOfStates()
    
    def getPreviousState(self):
        return self.previousState

    def getNextState(self):
        if(self.nextState != None):
            return self.nextState
        coordinateToUpdate = self.getRandomCoordinate()

        nextSpin = self._calculateNextSpin(coordinateToUpdate)

        nextStateAsArray = np.array(self.currentState)
        nextStateAsArray[coordinateToUpdate] = nextSpin

        self._setNextStateFromArray(nextStateAsArray)
        return self.nextState
    
class VariableTrackedIsingState(IsingState):
    #Provides additional speed saving when plotting magnetisation by tracking the number of spin up/ down
    def __init__(self, magnetisation):
        self.magnetisation = magnetisation

    def getMagnetisation(self):
        return self.magnetisation

class NearestNeighbourIsingState(IsingState):
    def __init__(self, state, spinStateSolver, dimension = 2,  previousState = None):
        super().__init__(state, spinStateSolver, previousState)

        if not isinstance(spinStateSolver, NearestNeighbourSpinStateSolver):
            raise Exception('Spin solver not nearest neighbours')
        if spinStateSolver.getDimension() != dimension:
            raise Exception('Spin solver has wrong dimension: ' + str(spinStateSolver.getDimension()) + ' required: ' + str(dimension))
    
    def _getLocalState(self, coordinate):
        coordinateSpin = self.getState(coordinate)
        neibouringSpin = self.getNeighbouringStates(coordinate)
        return (coordinateSpin, neibouringSpin)

    def _calculateNextSpin(self, coordinate):
        localState = self._getLocalState(coordinate)
        return self.spinStateSolver.getNextSpinState(localState)

    def OLDgetNextState(self):
        coordinateToUpdate = (np.random.randint(0, self.xDim), np.random.randint(0, self.yDim))
        coordinateSpin = self.getState(coordinateToUpdate)
        neibouringSpin = self.getNeighbouringStates(coordinateToUpdate)

        
        nextValue = self.spinStateSolver.getNextSpinState((coordinateSpin, neibouringSpin))
        
        nextStateAsArray = np.array(self.currentState)
        nextStateAsArray[coordinateToUpdate] = nextValue

        self.nextState = IsingState2D(nextStateAsArray, self.spinStateSolver.temperature)
        return self.nextState

    def __str__(self):
        return self.currentState.__str__()

class NearestNeighbourIsingState2D(NearestNeighbourIsingState):
    #Spin solver with additional speedup for the 2D case
    def __init__(self, state, spinStateSolver, previousState = None):
        super().__init__(state, spinStateSolver, 2, previousState)

    def getState(self, coordinate):
        (x,y) = coordinate
        return self.currentState[x % self.getDimensions()[0], y % self.getDimensions()[1]]

    def getNeighbouringStates(self, coordinate):
        x, y = coordinate
        return np.array([self.getState((x + 1 ,y)),
                            self.getState((x - 1 ,y)),
                            self.getState((x ,y + 1)),
                            self.getState((x ,y - 1))])

    def getRandomCoordinate(self):
        return (np.random.randint(0, self.getDimensions()[0]) , np.random.randint(0, self.getDimensions()[1]))

    def generateFromArray(array, temperature):
        solver = LazyNearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        return NearestNeighbourIsingState2D(array, solver)
    
    def _setNextStateFromArray(self, array):
        self.nextState = NearestNeighbourIsingState2D(array, self.spinStateSolver, previousState = self)

class TrackedNearestNeighbourIsingState2D(VariableTrackedIsingState, NearestNeighbourIsingState2D):
    def __init__(self, state, spinStateSolver, previousState = None, magnetisation = None):
        super(NearestNeighbourIsingState2D).__init__(state, spinStateSolver, 2, previousState)
        super(VariableTrackedIsingState).__init__(magnetisation)

    def generateFromArray(array, temperature):
        solver = LazyNearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        magnetisation = 0
        return TrackedNearestNeighbourIsingState2D(array, solver)

    def _setNextStateFromArray(self, array):
        self.nextState = NearestNeighbourIsingState2D(array, self.spinStateSolver, previousState = self)    

class SpinStateSolver():
    #When given a spin state, and the state of its nearest neighbours the solver returns
    #the next spin state
    def __init__(self, temperature):
        self.temperature = temperature
    
    def getTemperature(self):
        return self.temperature
    
    def getProbabilityOfTransition(self, localState):
        #If flipping decreases energy then flip
        #Else flip with a probability exp(-E/KT)

        energyToFlip  = self._calculateEnergyToFlip(localState)
        
        ##If flipping would cause a reduction in energy, flip
        if(energyToFlip < 0):
            return 1
        
        return np.exp( - energyToFlip / (self.getTemperature()))
    
    def getNextSpinState(self, localState):
        if not self._isValidState(localState):
            raise Exception('Invalid Local State ' + str(localState))

        return self._getNextSpinState(localState)

class NearestNeighbourSpinStateSolver(SpinStateSolver):
    
    def __init__(self, temperature, dimension = 2):
        super().__init__(temperature)
        #N dimension, 2*N sides to the cube
        self.numberOfNeighbours = 2 * dimension
        self.exchangeEnergy = 1

    def _isValidState(self, localState):
        (spin, neighbours) = localState
        return (neighbours.size == self.numberOfNeighbours)

    def _calculateEnergyToFlip(self, localState):
        (currentState, neighbours) = localState
        numberOfPositiveNeighbours = np.count_nonzero(neighbours, axis = -1)

        localEnergy = - self.exchangeEnergy * (1 if currentState else -1) * (numberOfPositiveNeighbours - (self.numberOfNeighbours / 2))
        energyToFlip = - 2 * localEnergy
        return energyToFlip

    def _getNextSpinState(self, localState):
        (coordinateSpin, neibouringSpin) = localState
        pOfTransition = self.getProbabilityOfTransition(localState)
        hasFlipped = np.random.rand() > pOfTransition
        nextState  = np.logical_xor(hasFlipped, coordinateSpin)
        return nextState

    def getDimension(self):
        return int(self.numberOfNeighbours / 2)

class LazyNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver):

    def __init__(self, temperature, dimension = 2):
        super().__init__(temperature, dimension)
        self._setupSpinProbabilityDict()

    def _setupSpinProbabilityDict(self):
        probabilityDict = {}
        for spin in [True,False]:
            for positiveNeighbours in range(self.numberOfNeighbours + 1):
                neighbouringSpin = [True for x in range(positiveNeighbours)]
                localState = (spin, neighbouringSpin)
                probabilityDict[(spin, positiveNeighbours)] = super().getProbabilityOfTransition(localState)
        self.probabilityDict = probabilityDict

    def getProbabilityOfTransition(self, localState):
        (spin, neighbours) = localState
        numberOfPositiveNeighbours = np.count_nonzero(neighbours, axis = -1)
        return self.probabilityDict[(spin, numberOfPositiveNeighbours)]
    
class VectorisedSpinStateSolver(NearestNeighbourSpinStateSolver):

    def __init__(self, temperature, dimension = 2):
        super().__init__(temperature, dimension)
        self._setupSpinProbabilityDict()
        self._setupSpinTransitionDict()

    def _setupSpinProbabilityDict(self):
        probabilityDict = {}
        for spin in [True,False]:
            for positiveNeighbours in range(self.numberOfNeighbours + 1):
                neighbouringSpin = [True for x in range(positiveNeighbours)]
                localState = (spin, neighbouringSpin)
                probabilityDict[(spin, positiveNeighbours)] = super().getProbabilityOfTransition(localState)
        self.probabilityDict = probabilityDict

    def _generateNewSpinTransitions(self, spin, positiveNeighbours):
        numberToGenerate = 10000
        randomNumbers = np.random.rand(numberToGenerate)
        hasFlipped    = randomNumbers > self.probabilityDict[(spin, positiveNeighbours)]
        self.spinTransitionDict[(spin, positiveNeighbours)] = np.logical_xor(hasFlipped, spin)

    def _setupSpinTransitionDict(self):
        self.spinTransitionDict = {}
        for spin in [True,False]:
            for positiveNeighbours in range(self.numberOfNeighbours + 1):
                self._generateNewSpinTransitions(spin, positiveNeighbours)

    def getProbabilityOfTransition(self, localState):
        (spin, neighbours) = localState
        numberOfPositiveNeighbours = np.count_nonzero(neighbours, axis = -1)
        return self.probabilityDict[(spin, numberOfPositiveNeighbours)]

    def _getNextSpinState(self, localState):
        (spin, neighbours) = localState
        positiveNeighbours = np.count_nonzero(neighbours, axis = -1)
        
        nextStates = self.spinTransitionDict[(spin, positiveNeighbours)]
        
        if nextStates.size == 0:
            self._generateNewSpinTransitions(spin, positiveNeighbours)
            nextStates = self.spinTransitionDict[(spin, positiveNeighbours)]

        nextState, nextStates = nextStates[-1], nextStates[:-1]
        self.spinTransitionDict[(spin, positiveNeighbours)] = nextStates
        return nextState


if __name__ == '__main__':


    sim = NearestNeighbourIsingSimulation2D(100, 10, 10)
    plotter = IsingSimulationPlotter(sim)
    sim.getStateNumber(300000)
    start = time.time()
    
    plotter.plotFirstNMagnetisations(300000)
    
    end = time.time()
    print(end-start)
    start = time.time()
    sim.getStateNumber(60000)
    end = time.time()
    print(end-start)
