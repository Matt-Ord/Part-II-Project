import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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

class IsingSimulation2D(IsingSimulation):

    def getFirstNStatesAsAnimation(self, N, resolution = 200):        
        statesToAnimate = np.linspace(0,N, num = resolution, dtype = int)

        #images = [(self.getStateNumber(x).getCurrentStateAsImage(),) for x in statesToAnimate]
        fig = plt.figure()
        images = []
        for x in statesToAnimate:
            images.append((self.getStateNumber(x).getCurrentStateAsImage(),))
      
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                
        im_ani = animation.ArtistAnimation(fig, images, interval=10, repeat_delay=0,
                                   blit=True)
        im_ani.save('im.mp4', writer=writer)
        plt.close('all')
        return fig, im_ani
        
    
class NearestNeighbourIsingSimulation(IsingSimulation):
    
    def __init__(self, initialIsingState):
        if not isinstance(initialIsingState, NearestNeighbourIsingState):
            raise Exception('Not nearest neighbour ising state')

        super().__init__(initialIsingState)

class NearestNeighbourIsingSimulation2D(IsingSimulation2D, NearestNeighbourIsingSimulation):

    def __init__(self, temperature, xDim, yDim, initialState = None):
        initialState = np.ones((xDim, yDim), dtype=bool) if initialState == None else initialState

        initialIsingState = NearestNeighbourIsingState2D.generateFromArray(initialState, temperature)

        NearestNeighbourIsingSimulation.__init__(self, initialIsingState)
        

class IsingSimulationPlotter():

    def __init__(self, isingSimulation):
        self.simulation = isingSimulation

    def plotFirstNMagnetisations(self, N):
        magnetisations = self.simulation.getFirstNMagnetisations(N)
        stateNumbers   = range(N)
        fig1 = plt.figure()
        plt.plot(stateNumbers, magnetisations)
        plt.savefig('MagnetisationPlot.pdf')
        plt.close(fig1.number)
        #fig1.show()
        
    def plotFirstNMagnetisationsAtTemperatures(self, N, Temperatures):
        magnetisations = self.simulation.getFirstNMagnetisations(N)
        stateNumbers   = range(N)
        fig1 = plt.figure()
        plt.plot(stateNumbers, magnetisations)
        plt.savefig('MagnetisationPlot.pdf')
        plt.close(fig1.number)
        #fig1.show()

class IsingSimulationPlotter2D(IsingSimulationPlotter):
    def __init__(self, isingSimulation):
        super().__init__(isingSimulation)

    def animateFirstNStates(self, N):
        fig, ani = self.simulation.getFirstNStatesAsAnimation(N)


'''-----------------------------------------------------------------------'''

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

    def getCurrentStateAsArray(self):
        return np.array(self.currentState)

class IsingState2D(IsingState):

    def getDimension(self):
        return 2

    def getState(self, coordinate):
        (x,y) = coordinate
        return self.currentState[x % self.getDimensions()[0], y % self.getDimensions()[1]]

    def getNeighbouringStates(self, coordinate):
        x, y = coordinate
        return np.array([self.getState((x + 1 ,y)),
                            self.getState((x - 1 ,y)),
                            self.getState((x ,y + 1)),
                            self.getState((x ,y - 1))])
    def countAllPositiveNeighbours(self):
        #Find the neighbouring elements
        left   = np.roll(self.currentState, 1, axis = 0)
        right  = np.roll(self.currentState,-1, axis = 0)
        top    = np.roll(self.currentState, 1, axis = 1)
        bottom = np.roll(self.currentState,-1, axis = 1)
        #find the total spin
        return np.add(np.add(left, right, dtype = int), np.add(top, bottom, dtype = int))
                      
    def getAllLocalStates(self):
        numberOfPositiveNeighbours = self.countAllPositiveNeighbours()
        return np.stack((self.currentState, numberOfPositiveNeighbours), axis=-1)

    def getRandomCoordinate(self):
        return (np.random.randint(0, self.getDimensions()[0]) , np.random.randint(0, self.getDimensions()[1]))

    def getCurrentStateAsImage(self):
        return plt.imshow(self.currentState, aspect='equal', cmap=plt.cm.gray, interpolation='nearest', animated = True, norm=plt.Normalize(0, 1))
        ##plt.pcolor(range(self.getDimensions()[0]), range(self.getDimensions()[1]), self.currentState, norm=plt.Normalize(0, 1))
    
class RandomUpdateIsingState(IsingState):
    
    def getNextState(self):
        #Generates the next state by updating a random coordinate
        if(self.nextState != None):
            return self.nextState
        
        coordinateToUpdate = self.getRandomCoordinate()
        
        localState = self._getLocalState(coordinateToUpdate)
        (isFlipped, nextSpin) = self.spinStateSolver.getNextSpinState(localState)
        
        self._setNextState(coordinateToUpdate, isFlipped, nextSpin)
        return self.nextState
    
class VariableTrackedIsingState(RandomUpdateIsingState):
    #Provides additional speed saving when plotting magnetisation by tracking the number of spin up/ down
    def __init__(self, magnetisation):
        self.magnetisation = magnetisation

    def getMagnetisation(self):
        return self.magnetisation
    
class SystematicUpdateIsingState(IsingState):
    
    def getNextState(self):
        #Generates teh next state by updating all coordinates
        if(self.nextState != None):
            return self.nextState
        
        localStates = self.getAllLocalStates()

        nextSpins = self.spinStateSolver.getNextSpinStates(localStates)
        
        self._setNextState(nextSpins)
        return self.nextState
    

class NearestNeighbourIsingState(IsingState):
    def __init__(self, state, spinStateSolver,  previousState = None):
        super().__init__(state, spinStateSolver, previousState)

        if not isinstance(spinStateSolver, NearestNeighbourSpinStateSolver):
            raise Exception('Spin solver not nearest neighbours')
        if spinStateSolver.getDimension() != self.getDimension():
            raise Exception('Spin solver has wrong dimension: ' + str(spinStateSolver.getDimension()) + ' required: ' + str(self.getDimension()))
    
    def _getLocalState(self, coordinate):
        coordinateSpin = self.getState(coordinate)
        positiveNeibouringSpins = np.count_nonzero(self.getNeighbouringStates(coordinate))
        return (coordinateSpin, positiveNeibouringSpins)
    
    def _getAllLocalStates(self):
        return np.vectorize(self._getLocalState)(self._getAllCoordinates())
        
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

class NearestNeighbourIsingState2D(IsingState2D, NearestNeighbourIsingState, SystematicUpdateIsingState):
    #Spin solver with additional speedup for the 2D case
    def __init__(self, state, spinStateSolver, previousState = None):
        NearestNeighbourIsingState.__init__(self, state, spinStateSolver, previousState = previousState)

    def generateFromArray(array, temperature):
        solver = NearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        return NearestNeighbourIsingState2D(state, solver)
    
    def _setNextState(self, coordinateFlipped, hasFlipped, nextSpin):
        if(not hasFlipped):
            self.nextState = NearestNeighbourIsingState2D(self.currentState, self.spinStateSolver, previousState = self)
        else:
            nextStateAsArray = np.array(self.currentState)
            nextStateAsArray[coordinateFlipped] = nextSpin
            self.nextState = NearestNeighbourIsingState2D(nextStateAsArray, self.spinStateSolver, previousState = self)

class TrackedNearestNeighbourIsingState2D(VariableTrackedIsingState, NearestNeighbourIsingState2D):
    def __init__(self, state, spinStateSolver, magnetisation, previousState = None):
        NearestNeighbourIsingState2D.__init__(self, state, spinStateSolver, previousState)
        VariableTrackedIsingState.__init__(self, magnetisation)

    def generateFromArray(array, temperature):
        solver = NearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        magnetisation = 2 * np.count_nonzero(state) - state.size
        return TrackedNearestNeighbourIsingState2D(array, solver, magnetisation)

    def _setNextState(self, coordinateFlipped, hasFlipped, nextSpin):
        if(not hasFlipped):
            self.nextState = TrackedNearestNeighbourIsingState2D(self.currentState, self.spinStateSolver, self.getMagnetisation(), previousState = self)
        else:
            nextStateAsArray = np.array(self.currentState)
            nextStateAsArray[coordinateFlipped] = nextSpin
            newMagnetisation = (self.getMagnetisation() + 2 if nextSpin else self.getMagnetisation() - 2)
            self.nextState = TrackedNearestNeighbourIsingState2D(nextStateAsArray, self.spinStateSolver, magnetisation = newMagnetisation , previousState = self)

'''---------------------------------------------------------------------------------------------'''

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
    
    def getNextSpinStates(self, localStates):
        if not np.all(self._areValidStates(localStates)):
            raise Exception('Invalid Local State ' + str(localStates))

        return self._getNextSpinStates(localStates)
    
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
        self._setupSpinProbabilityDict()
        #self._getNextSpinStates = np.vectorize(self._getNextSpinState)

    def _isValidState(self, localState):
        print("L", localState)
        (_, positiveNeighbours) = localState
        return positiveNeighbours <= self.numberOfNeighbours
    
    def _areValidStates(self, localStates):
        return np.greater_equal(self.numberOfNeighbours, localStates[...,1])

    def _calculateEnergyToFlip(self, localState):
        print(localState)
        (currentState, numberOfPositiveNeighbours) = localState

        localEnergy = - self.exchangeEnergy * (1 if currentState else -1) * (numberOfPositiveNeighbours - (self.numberOfNeighbours / 2))
        energyToFlip = - 2 * localEnergy
        return energyToFlip

    def getDimension(self):
        return int(self.numberOfNeighbours / 2)

    def _setupSpinProbabilityDict(self):
        probabilityDict = {}
        for spin in [True,False]:
            for positiveNeighbours in range(self.numberOfNeighbours + 1):
                localState = (spin, positiveNeighbours)
                probabilityDict[(spin, positiveNeighbours)] = super().getProbabilityOfTransition(localState)
        self.probabilityDict = probabilityDict
        self.getProbabilityOfTransition  = probabilityDict.get
        print(self.probabilityDict)
        def f(a,b):
            return probabilityDict.get((a==1,b))
        self.getProbabilityOfTransitions = np.vectorize(f)
        #self.getProbabilityOfTransitions = np.vectorize(lambda a,b : probabilityDict.get((a,b)))
        
    def _getNextSpinState(self, localState):
        (coordinateSpin, _) = localState
        pOfTransition = self.getProbabilityOfTransition(localState)
        hasFlipped = np.random.rand() > pOfTransition
        nextState  = np.logical_xor(hasFlipped, coordinateSpin)
        return (hasFlipped, nextState)
        
    def _getNextSpinStates(self, localStates):
        coordinateSpins = localStates[...,0]
        pOfTransition = self.getProbabilityOfTransitions(localStates[...,0],localStates[...,1])
        print(" " ,localStates, self.getProbabilityOfTransitions(localStates[...,0],localStates[...,1]))
        #print(np.random.rand(10,10))
        hasFlipped = np.greater(np.random.rand(*coordinateSpins.shape), pOfTransition)
        nextState  = np.logical_xor(hasFlipped, coordinateSpins)
        return (hasFlipped, nextState)
    
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
        self.vectorisedProbabilityDict = np.vectorize(probabilityDict.get)

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

def plotNMagnetisationsAgainstTemperature():
    pass


if __name__ == '__main__':


    sim = NearestNeighbourIsingSimulation2D(100, 10, 10)
    plotter = IsingSimulationPlotter2D(sim)
    sim.getStateNumber(30000)
    start = time.time()
    #sim.getStateNumber(300000)
    #plotter.animateFirstNStates(200000)
    
    end = time.time()
    plotter.plotFirstNMagnetisations(10000)
    print(end-start)
    start = time.time()
    sim.getStateNumber(60000)
    end = time.time()
    print(end-start)
