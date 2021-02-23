import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


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

    def getFirstNStatesAsAnimation(self, N, resolution = 50):        
        statesToAnimate = np.linspace(0,N, num = resolution, dtype = int)

        #images = [(self.getStateNumber(x).getCurrentStateAsImage(),) for x in statesToAnimate]
        fig = plt.figure()
        images = []
        for x in statesToAnimate:
            images.append((self.getStateNumber(x).getStateAsImage(),))
      
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
        initialState = np.zeros((xDim, yDim), dtype=bool) if initialState == None else initialState

        initialIsingState = SystematicNearestNeighbourIsingState2D.generateFromArray(initialState, temperature)

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
        self.initialState = state                   #Array representing the true ising state
        self.currentState = np.array(state)         #Array on which any intermediate calculation is done
        self.spinStateSolver = spinStateSolver
        self.nextState = None
        self.previousState = previousState
    
    def getDimensions(self):
        return self.initialState.shape

    def getNumberOfStates(self):
        return self.initialState.size

    def getNumberOfSpinUp(self):
        return np.count_nonzero(self.initialState)
    
    def getMagnetisation(self):
        return 2 * self.getNumberOfSpinUp() - self.getNumberOfStates()
    
    def getPreviousState(self):
        return self.previousState
    
    def getAllCoordinates(self):
        points = [range(x) for x in self.getDimensions()]
        pointsMesh = np.meshgrid(*points, indexing = "ij")
        return np.stack(pointsMesh, axis = -1)

    def getStateAsArray(self):
        return np.array(self.initialState)

class IsingState2D(IsingState):

    def getDimension(self):
        return 2

    def getState(self, coordinate):
        (x,y) = coordinate
        return self.initialState[x % self.getDimensions()[0], y % self.getDimensions()[1]]

    def getNeighbouringStates(self, coordinate):
        x, y = coordinate
        return np.array([self.getState((x + 1 ,y)),
                            self.getState((x - 1 ,y)),
                            self.getState((x ,y + 1)),
                            self.getState((x ,y - 1))])
                      
    def getAllInitialLocalStates(self):
        numberOfPositiveNeighbours = self.countAllInitialPositiveNeighbours()
        return np.stack((self.initialState, numberOfPositiveNeighbours), axis=-1)
        
    def countAllInitialPositiveNeighbours(self):
        #Find the neighbouring elements
        left   = np.roll(self.initialState, 1, axis = 0)
        right  = np.roll(self.initialState,-1, axis = 0)
        top    = np.roll(self.initialState, 1, axis = 1)
        bottom = np.roll(self.initialState,-1, axis = 1)
        #find the total spin
        return np.add(np.add(left, right, dtype = int), np.add(top, bottom, dtype = int))
        
    def getAllCurrentLocalStates(self):
        numberOfPositiveNeighbours = self.countAllCurrentPositiveNeighbours()
        return np.stack((self.currentState, numberOfPositiveNeighbours), axis=-1)
    
    def countAllCurrentPositiveNeighbours(self):
        #Find the neighbouring elements
        left   = np.roll(self.currentState, 1, axis = 0)
        right  = np.roll(self.currentState,-1, axis = 0)
        top    = np.roll(self.currentState, 1, axis = 1)
        bottom = np.roll(self.currentState,-1, axis = 1)
        #find the total spin
        return np.add(np.add(left, right, dtype = int), np.add(top, bottom, dtype = int))

    def getRandomCoordinate(self):
        return (np.random.randint(0, self.getDimensions()[0]) , np.random.randint(0, self.getDimensions()[1]))

    def getStateAsImage(self):
        return plt.imshow(self.initialState, aspect='equal', cmap=plt.cm.gray, interpolation='nearest', animated = True, norm=plt.Normalize(0, 1))
        ##plt.pcolor(range(self.getDimensions()[0]), range(self.getDimensions()[1]), self.currentState, norm=plt.Normalize(0, 1))
    
class RandomIsingState(IsingState):
    def __init__(self, magnetisation):
        self.magnetisation = magnetisation

    def getMagnetisation(self):
        return self.magnetisation
    
    def getNextState(self):
        #Generates the next state by updating a random coordinate
        if(self.nextState != None):
            return self.nextState
        
        coordinateToUpdate = self.getRandomCoordinate()
        
        localState = self._getLocalState(coordinateToUpdate)
        isFlipped, nextSpin = self.spinStateSolver.getNextSpinStates(np.array(localState))
        
        self._setNextState(coordinateToUpdate, isFlipped, nextSpin)
        return self.nextState
    
class SystematicIsingState(IsingState):
    #Updates all elements, however this does not take into account the fact that
    #the local state also changes each time a state is flipped.
    neighbourhoodSlices = [np.s_[...]]
    
    #Classes Implementing the method should provide a list conataining
    #the required slicing rules of each 'neighbourhood'- default just np.sl_[...]
    #by only updating Neighbourhoods which don't share nearest neighbours the effect
    #of sudden jumps is reduced and oscillations are prevented almost entirely
    
    
    def getNextState(self):
        #Generates the next state by updating all coordinates
        if(self.nextState != None):
            return self.nextState
        #Suffles neighbourhood to reduce order in the final ising state
        random.shuffle(self.neighbourhoodSlices)
        for s in self.neighbourhoodSlices:
            neighbourhoodLocalStates = self.getAllCurrentLocalStates()[s] #[s] - only neighbourhood slice is returned
            _, neighbourhoodNextSpins = self.spinStateSolver.getNextSpinStates(neighbourhoodLocalStates)
            self.currentState[s] = neighbourhoodNextSpins                 #[s] - only neighbourhood slice is updated
        
        self._setNextState()
        return self.nextState
    
class ImprovedSystematicIsingState(SystematicIsingState):
    #Improves the systematic approach by only reverting to piecewise method
    #if a large jump is detected
    def _updateAllStatesIndividually(self):
        coordsToUpdate = self.getAllCoordinates()
        np.apply_along_axis(self._updateSingleState, -1, coordsToUpdate)

    def _updateSingleState(self, coordinateToUpdate):
        localState = self._getLocalState(coordinateToUpdate)
        _, nextSpin = self.spinStateSolver.getNextSpinStates(np.array(localState))
        self.currentState[tuple(coordinateToUpdate)] = nextSpin
    
    def getNextState(self):
        #Generates the next state by updating all coordinates
        if(self.nextState != None):
            return self.nextState
        
        random.shuffle(self.neighbourhoodSlices)
        for s in self.neighbourhoodSlices:
            neighbourhoodLocalStates = self.getAllCurrentLocalStates()[s] #[s] - only neighbourhood slice is returned
            haveFlipped, neighbourhoodNextSpins = self.spinStateSolver.getNextSpinStates(neighbourhoodLocalStates)
            
            if np.count_nonzero(haveFlipped) > (self.getNumberOfStates() / (len(self.neighbourhoodSlices) * 2)):
                #print(np.count_nonzero(haveFlipped))
                #print(haveFlipped)
                #Revert to updating individual states to prevent unnatural jumps
                self.currentState = self.getStateAsArray()
                self._updateAllStatesIndividually()
                self._setNextState()
                return self.nextState
            
            self.currentState[s] = neighbourhoodNextSpins                 #[s] - only neighbourhood slice is updated
        
        
        self._setNextState()
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
        
        nextStateAsArray = np.array(self.initialState)
        nextStateAsArray[coordinateToUpdate] = nextValue

        self.nextState = IsingState2D(nextStateAsArray, self.spinStateSolver.temperature)
        return self.nextState

    def __str__(self):
        return self.initialState.__str__()


class SystematicNearestNeighbourIsingState2D(IsingState2D, ImprovedSystematicIsingState, NearestNeighbourIsingState):
    
    neighbourhoodSlices = [np.s_[0::2,0::2], np.s_[1::2,0::2], 
                           np.s_[0::2,1::2], np.s_[1::2,1::2],]
    '''
    neighbourhoodSlices = [np.s_[0::2,0::2], np.s_[1::2,0::2],
                           np.s_[0::2,1::2], np.s_[1::2,1::2],]
    '''
    '''
    neighbourhoodSlices = [np.s_[0::4,0::4], np.s_[1::4,0::4], np.s_[2::4,0::4], np.s_[3::4,0::4],
                           np.s_[0::4,1::4], np.s_[1::4,1::4], np.s_[2::4,1::4], np.s_[3::4,1::4],
                           np.s_[0::4,2::4], np.s_[1::4,2::4], np.s_[2::4,2::4], np.s_[3::4,2::4],
                           np.s_[0::4,3::4], np.s_[1::4,3::4], np.s_[2::4,3::4], np.s_[3::4,3::4],]
    '''
    #Spin solver with additional speedup for the 2D case
    def __init__(self, state, spinStateSolver, previousState = None):
        NearestNeighbourIsingState.__init__(self, state, spinStateSolver, previousState = previousState)
        #self._generateSlices()

    def generateFromArray(array, temperature):
        solver = NearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        return SystematicNearestNeighbourIsingState2D(state, solver)
    
    def _generateSlices(self):
        xDim, yDim = self.getDimensions()
        neighbourhoodSlices = []
        neighbourhoodSlices += [np.s_[i,:] for i in range(xDim)]
        neighbourhoodSlices += [np.s_[:,i] for i in range(yDim)]
        self.neighbourhoodSlices = neighbourhoodSlices
    
    def _setNextState(self):
        self.nextState = SystematicNearestNeighbourIsingState2D(self.currentState, self.spinStateSolver, previousState = self)


class RandomNearestNeighbourIsingState2D(IsingState2D, RandomIsingState, NearestNeighbourIsingState):
    def __init__(self, state, spinStateSolver, magnetisation, previousState = None):
        NearestNeighbourIsingState.__init__(self, state, spinStateSolver, previousState = previousState)
        RandomIsingState.__init__(self, magnetisation)

    def generateFromArray(array, temperature):
        solver = NearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        magnetisation = 2 * np.count_nonzero(state) - state.size
        return RandomNearestNeighbourIsingState2D(array, solver, magnetisation)

    def _setNextState(self, coordinateFlipped, hasFlipped, nextSpin):
        if(not hasFlipped):
            self.nextState = RandomNearestNeighbourIsingState2D(self.initialState, self.spinStateSolver, self.getMagnetisation(), previousState = self)
        else:
            nextStateAsArray = np.array(self.initialState)
            nextStateAsArray[coordinateFlipped] = nextSpin
            newMagnetisation = (self.getMagnetisation() + 2 if nextSpin else self.getMagnetisation() - 2)
            self.nextState = RandomNearestNeighbourIsingState2D(nextStateAsArray, self.spinStateSolver, magnetisation = newMagnetisation , previousState = self)
            
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
        if not self._isValidState(localStates):
            raise Exception('Invalid Local State ' + str(localStates))

        return self._getNextSpinStates(localStates)

class NearestNeighbourSpinStateSolver(SpinStateSolver):
    
    def __init__(self, temperature, dimension = 2):
        super().__init__(temperature)
        #N dimension, 2*N sides to the cube
        self.numberOfNeighbours = 2 * dimension
        self.exchangeEnergy = 1
        self._setupSpinProbabilityDict()
        #self._getNextSpinStates = np.vectorize(self._getNextSpinState)
        
    def getDimension(self):
        return int(self.numberOfNeighbours / 2)
    
    def _isValidState(self, localStates):
        numberOfPositiveNeighbours = localStates[...,1]
        return np.all(np.greater_equal(self.numberOfNeighbours, numberOfPositiveNeighbours))

    def _calculateEnergyToFlip(self, localState):
        (currentState, numberOfPositiveNeighbours) = localState

        localEnergy = - self.exchangeEnergy * (1 if currentState else -1) * (numberOfPositiveNeighbours - (self.numberOfNeighbours / 2))
        energyToFlip = - 2 * localEnergy
        return energyToFlip

    def _setupSpinProbabilityDict(self):
        probabilityDict = {}
        for spin in [True,False]:
            for positiveNeighbours in range(self.numberOfNeighbours + 1):
                localState = (spin, positiveNeighbours)
                probabilityDict[(spin, positiveNeighbours)] = super().getProbabilityOfTransition(localState)
        self.getProbabilityOfTransition  = probabilityDict.get
        self.getProbabilityOfTransitions = np.vectorize(lambda a,b : probabilityDict.get((a,b)))        
        
    def _getNextSpinStates(self, localStates):
        coordinateSpins = localStates[...,0]
        pOfTransition = self.getProbabilityOfTransitions(localStates[...,0],localStates[...,1])
        haveFlipped = np.greater(np.random.rand(*coordinateSpins.shape), pOfTransition)
        
        nextStates  = np.logical_xor(haveFlipped, coordinateSpins)
        #print(nextStates)
        return (haveFlipped, nextStates)


def plotNMagnetisationsAgainstTemperature():
    pass


if __name__ == '__main__':


    sim = NearestNeighbourIsingSimulation2D(10000, 100, 100)
    plotter = IsingSimulationPlotter2D(sim)
    #sim.getStateNumber(30000)
    start = time.time()
    #sim.getStateNumber(300000)
    sim.getStateNumber(500)
    
    
    end = time.time()
    print(end-start)
    plotter.animateFirstNStates(50)
    plotter.plotFirstNMagnetisations(500)
    start = time.time()
    #sim.getStateNumber(60000)
    end = time.time()
    print(end-start)
