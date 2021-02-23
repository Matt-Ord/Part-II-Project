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
    
    def getTemperature(self):
        return self.firstIsingState.getTemperature()
    
    def _applyToStates(self, function, indexOfStates):
        #Equivalent to but faster than [function(self.getStateNumber(N)) for N in indexOfStates]
        np.sort(indexOfStates)
        lastState = indexOfStates[-1]
        
        vals = []
        currentState = self.firstIsingState
        for x in range(lastState):
            if(x == indexOfStates[0]):
                indexOfStates = indexOfStates[1:]
                vals.append(function(currentState))
            currentState = currentState.getNextState()
        vals.append(function(currentState))
        
        return vals
    
    def _getFirstNMagnetisations(self, N, resolution = None):
        statesToSample = (range(N) if resolution == None else np.linspace(0,N, num=min(resolution,N), dtype = int))
        return self._applyToStates(IsingState.getMagnetisation, statesToSample)
    
    def _getFirstNStates(self, N, resolution = None):
        statesToSample = (range(N) if resolution == None else np.linspace(0,N, num=min(resolution,N), dtype = int))

        return self._applyToStates(IsingState.getStateAsArray, statesToSample)

    def _getFirstNFourierTransforms(self, N, resolution = None):
        statesToSample = (range(N) if resolution == None else np.linspace(0,N, num=min(resolution,N), dtype = int))

        return self._applyToStates(IsingState.getStateAsDFT, statesToSample)

    def plotFirstNMagnetisations(self, N, resolution = 1000, ax = plt.axis(), label = ""):
        magnetisations = self._getFirstNMagnetisations(N, resolution = resolution)
        ax.plot(np.linspace(0,N, num=min(resolution,N), dtype = int), magnetisations, label = "")
        return ax
    
    def averageFirstNMagnetisations(self, N, resolution = None):
        magnetisations = self._getFirstNMagnetisations(N, resolution = resolution)
        return np.average(magnetisations, axis = 0)
    
    def averageFirstNStates(self, N, resolution = None):
        states = self._getFirstNStates(N, resolution = resolution)
        return np.average(states, axis = 0)
    
    def averageFirstNFourierTransforms(self, N, resolution = None):
        average = np.average(np.absolute(self._getFirstNFourierTransforms(N, resolution = resolution)), axis = 0)
        return average

class IsingSimulation2D(IsingSimulation):
    def _getFirstNImages(self, N, resolution):
        statesToSample = (range(N) if resolution == None else np.linspace(0,N, num=min(resolution,N), dtype = int))
        
        return self._applyToStates(IsingState2D.getStateAsImage, statesToSample)

    def animateFirstNStates(self, N, resolution = 100):
        fig = plt.figure()
        images = []
        for x in self._getFirstNImages(N, resolution):
            images.append((x,))
      
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                
        im_ani = animation.ArtistAnimation(fig, images, interval=10, repeat_delay=0,
                                   blit=True)
        im_ani.save('im.mp4', writer=writer)
        plt.close('all')
        return fig, im_ani
    
    def animateFirstNFourierTransforms(self, N, resolution = 100):
        fig = plt.figure()
        images = []
        for x in self._getFirstNFourierTransforms(N, resolution):
            im = plt.imshow(np.absolute(x), aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            images.append((im,))
            
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                
        im_ani = animation.ArtistAnimation(fig, images, interval=10, repeat_delay=0,
                                   blit=True)
        im_ani.save('fourierTransformAnimation.mp4', writer=writer)
        plt.close('all')
        return fig, im_ani
    
    def plotAverageFirstNFourierTransforms(self, N, resolution = 1000, ax = None, label = ""):
        ax = plt.axis() if ax == None else ax
        averageTransform = self.averageFirstNFourierTransforms(N, resolution = resolution)
        im = plt.imshow(averageTransform, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
        ax.add_image(im)
        ax.set_title(label)
        
        return ax
    
    def plotAverageFirstNStates(self, N, resolution = None, ax = plt.axis(), label = ""):
        ax = plt.axis() if ax == None else ax
        averageStates = self.averageFirstNStates(N, resolution = resolution)

        im = plt.imshow(averageStates, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
        ax.add_image(im)
        ax.set_title(label)
        
        return ax
        
    
class NearestNeighbourIsingSimulation2D(IsingSimulation2D):
    
    def __init__(self, initialIsingState):
        if not isinstance(initialIsingState, NearestNeighbourIsingState):
            raise Exception('Not nearest neighbour ising state')

        super().__init__(initialIsingState)
        
    def generateRandomNearestNeighbourSim(temperature, xDim, yDim):
        initialState = np.ones((xDim, yDim), dtype=bool)
        initialIsingState = RandomNearestNeighbourIsingState2D.generateFromArray(initialState, temperature)

        return NearestNeighbourIsingSimulation2D(initialIsingState)
        
    def generateSystematicNearestNeighbourSim(temperature, xDim, yDim):
        initialState = np.ones((xDim, yDim), dtype=bool)
        initialIsingState = SystematicNearestNeighbourIsingState2D.generateFromArray(initialState, temperature)

        return NearestNeighbourIsingSimulation2D(initialIsingState)

        

class IsingSimulationPlotter():

    def plotFirstNMagnetisations(N, simulations, sameAxis = False, simulationLables = None):
        numberOfSimulations = len(simulations)
        sameAxis |= (numberOfSimulations == 1)
        
        fig, axs = plt.subplots(1 if sameAxis else numberOfSimulations, 1, sharex=True)
        for i in range(numberOfSimulations):
            simulations[i].plotFirstNMagnetisations(N, ax = (axs if sameAxis else axs[i]), label = ("" if simulationLables == None else simulationLables[i]))
        
        fig.savefig('MagnetisationPlot.pdf')
        return fig, axs
    
class IsingSimulationPlotter2D(IsingSimulationPlotter):

    def animateFirstNStates(N, simulation):
        fig, ani = simulation.getFirstNStatesAsAnimation(N)
        
    def plotAverageFirstNFourierTransforms(N, simulations, simulationLables = None):
        numberOfSimulations = len(simulations)
        
        fig, axs = plt.subplots(numberOfSimulations, 1, sharex=True)
        
        for i in range(numberOfSimulations):
            simulations[i].plotAverageFirstNFourierTransforms(N, ax = (axs if numberOfSimulations == 1 else axs[i]), label = ("" if simulationLables == None else simulationLables[i]))
        
        fig.savefig('AverageFourierTransformPlot.pdf')
        return fig, axs
    
    def plotAverageFirstNStates(N, simulations, simulationLables = None):
        numberOfSimulations = len(simulations)
        
        fig, axs = plt.subplots(numberOfSimulations, 1, sharex=True)
        
        for i in range(numberOfSimulations):
            simulations[i].plotAverageFirstNStates(N, ax = (axs if numberOfSimulations == 1 else axs[i]), label = ("" if simulationLables == None else simulationLables[i]))
        
        fig.savefig('AverageStatesPlot.pdf')
        return fig, axs
    
    def animateFirstNFourierTransforms(N, simulation):
        fig, ani = simulation.animateFirstNFourierTransforms(N)
    
    def plotFirstNMagnetisationsAtTemperatures(N, temperatures, xDim, yDim, sameAxis = False):
        simulations = [NearestNeighbourIsingSimulation2D.generateRandomNearestNeighbourSim(T, xDim, yDim) for T in temperatures]
        fig, axs = IsingSimulationPlotter2D.plotFirstNMagnetisations(N, simulations, sameAxis, simulationLables = map(str, temperatures))
        if sameAxis:
            axs.legend()
        fig.savefig('MagnetisationPlot2.pdf')
        return fig, axs
    
    def plotMeanMagnetisationAgainstTemperature(N, simulations, ax = None, label = ""):
        ax = (plt.axes() if ax == None else ax)
        averageMagnetisations = [sim.averageFirstNMagnetisations(N) for sim in simulations]
        temperatures =          [sim.getTemperature()               for sim in simulations]

        
        ax.plot(temperatures, averageMagnetisations, label = label)
        
        return ax
        
    def plotMeanMagnetisationAgainsyTemperatureForDifferentMethods():
        N = 800
        xDim = 40
        yDim = 40
        temperatures = np.linspace(0.5,1.7,20)
        systematicSimulations = [NearestNeighbourIsingSimulation2D.generateSystematicNearestNeighbourSim(T, xDim, yDim) for T in temperatures]
        randomSimulations     = [NearestNeighbourIsingSimulation2D.generateSystematicNearestNeighbourSim(T, xDim, yDim) for T in temperatures]
        fig, ax = plt.subplots()
        IsingSimulationPlotter2D.plotMeanMagnetisationAgainstTemperature(N, systematicSimulations, ax=ax, label = "systematic")
        IsingSimulationPlotter2D.plotMeanMagnetisationAgainstTemperature(N, randomSimulations    , ax=ax, label = "random")
        fig.savefig("MagnetisationTemperaturePlot.pdf")

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
    
    def getDimension(self):
        return self.initialState.shape

    def getNumberOfStates(self):
        return self.initialState.size

    def getNumberOfSpinUp(self):
        return np.count_nonzero(self.initialState)
    
    def getMagnetisation(self):
        #Spin up - spin down
        return 2 * self.getNumberOfSpinUp() - self.getNumberOfStates()
    
    def getTemperature(self):
        return self.spinStateSolver.getTemperature()
    
    def getPreviousState(self):
        return self.previousState
    
    def getAllCoordinates(self):
        return np.stack(np.indices(self.getDimensions()), axis = -1)
    
    def getRandomCoordinate(self):
        return np.random.choice(self.getAllCoordinates)

    def getStateAsArray(self):
        return np.array(self.initialState)
    
    def getStateAsDFT(self):
        #print(np.fft.fftn(self.initialState, norm=None).shape)
        ft = np.fft.fftn(self.initialState, norm=None)
        np.put(ft, 0, 0)
        return np.fft.fftshift(ft)
    
    def _setNextState(self):
        self.nextState = type(self)(self.currentState, self.spinStateSolver, previousState = self)

class IsingState2D(IsingState):

    def getDimension(self):
        return 2

    def getInitialState(self, coordinate):
        (x,y) = coordinate
        return self.initialState[x % self.getDimensions()[0], y % self.getDimensions()[1]]
    
    def getCurrentState(self, coordinate):
        (x,y) = coordinate
        return self.currentState[x % self.getDimensions()[0], y % self.getDimensions()[1]]

    def getCurrentNeighbouringStates(self, coordinate):
        x, y = coordinate
        return np.array([self.getCurrentState((x + 1 ,y)),
                            self.getCurrentState((x - 1 ,y)),
                            self.getCurrentState((x ,y + 1)),
                            self.getCurrentState((x ,y - 1))])
    
    def getInitialNeighbouringStates(self, coordinate):
        x, y = coordinate
        return np.array([self.getInitialState((x + 1 ,y)),
                            self.getInitialState((x - 1 ,y)),
                            self.getInitialState((x ,y + 1)),
                            self.getInitialState((x ,y - 1))])
                      
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
        currentStates = np.stack((self.currentState, numberOfPositiveNeighbours), axis=-1)
        #assert(np.all(currentStates == self._getAllCurrentLocalStates()))
        return currentStates
    
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
    
    def getNextState(self):
        #Generates the next state by updating a random coordinate
        if(self.nextState != None):
            return self.nextState
        
        coordinateToUpdate = self.getRandomCoordinate()
        
        localState = self.getInitialLocalState(coordinateToUpdate)
        isFlipped, nextSpin = self.spinStateSolver.getNextSpinStates(np.array(localState))
        self.currentState[coordinateToUpdate] = nextSpin
        self._setNextState()
        return self.nextState
    
class SystematicIsingState(IsingState):
    #Updates all elements, however by default this does not take into account the fact that
    #the local state also changes each time a state is flipped.
    neighbourhoodSlices = [np.s_[...]]
    
    #Classes Implementing the method should provide a list conataining
    #the required slicing rules of each 'neighbourhood'- default just np.sl_[...]
    #to only update Neighbourhoods which don't share nearest neighbours.
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
    numberOfSplits = 128
    
    def getNextState(self):
        #Generates the next state by updating all coordinates
        if(self.nextState != None):
            return self.nextState
        #Suffles neighbourhood to reduce order in the final ising state
        flatternCoordinates = np.arange(self.getNumberOfStates(), dtype = int)
        random.shuffle(flatternCoordinates)
        
        neighbourhoods = np.array_split(flatternCoordinates, self.numberOfSplits)
        for s in neighbourhoods:
            allLocalStates = self.getAllCurrentLocalStates()
            allLocalSpins        = allLocalStates[...,0]
            allNeighbouringSpins = allLocalStates[...,1]
            neighbourhoodLocalStatesSplit = (np.take(allLocalSpins, s), np.take(allNeighbouringSpins, s))
            neighbourhoodLocalStates = np.stack(neighbourhoodLocalStatesSplit, axis=-1) 
            #neighbourhoodLocalStates = np.apply_along_axis(lambda x: np.take(x, s), axis = -1, allLocalStates)
            _, neighbourhoodNextSpins = self.spinStateSolver.getNextSpinStates(neighbourhoodLocalStates)
            np.put(self.currentState, s, neighbourhoodNextSpins)
        
        self._setNextState()
        #print("Pass")
        return self.nextState
    

class NearestNeighbourIsingState(IsingState):
    def __init__(self, state, spinStateSolver,  previousState = None):
        super().__init__(state, spinStateSolver, previousState)

        if not isinstance(spinStateSolver, NearestNeighbourSpinStateSolver):
            raise Exception('Spin solver not nearest neighbours')
        if spinStateSolver.getDimension() != self.getDimension():
            raise Exception('Spin solver has wrong dimension: ' + str(spinStateSolver.getDimension()) + ' required: ' + str(self.getDimension()))
    
    def getInitialLocalState(self, coordinate):
        coordinateSpin = self.getInitialState(coordinate)
        positiveNeibouringSpins = np.count_nonzero(self.getInitialNeighbouringStates(coordinate))
        return (coordinateSpin, positiveNeibouringSpins)
    
    def getCurrentLocalState(self, coordinate):
        coordinateSpin = self.getCurrentState(coordinate)
        positiveNeibouringSpins = np.count_nonzero(self.getCurrentNeighbouringStates(coordinate))
        return (coordinateSpin, positiveNeibouringSpins)
    
    def _getAllInitialLocalStates(self):
        return np.apply_along_axis(self._getInitialLocalState, -1, self.getAllCoordinates())
    
    def _getAllCurrentLocalStates(self):
        return np.apply_along_axis(self.getCurrentLocalState, -1, self.getAllCoordinates())

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
        #self._setupNeighbourhoodSlices()

    def generateFromArray(array, temperature):
        solver = NearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        return SystematicNearestNeighbourIsingState2D(state, solver)
    
    def _setupNeighbourhoodSlices(self):
        xDim, yDim = self.getDimensions()
        neighbourhoodSlices = []
        neighbourhoodSlices += [np.s_[i,0::2] for i in range(xDim)]
        neighbourhoodSlices += [np.s_[i,1::2] for i in range(xDim)]
        neighbourhoodSlices += [np.s_[0::2,i] for i in range(yDim)]
        neighbourhoodSlices += [np.s_[1::2,i] for i in range(yDim)]
        return neighbourhoodSlices
    
 #   def _setNextState(self):
        #print(self.currentState[0,0])
  #      self.nextState = SystematicNearestNeighbourIsingState2D(self.currentState, self.spinStateSolver, previousState = self)


class RandomNearestNeighbourIsingState2D(IsingState2D, RandomIsingState, NearestNeighbourIsingState):
    def __init__(self, state, spinStateSolver, previousState = None):
        NearestNeighbourIsingState.__init__(self, state, spinStateSolver, previousState = previousState)

    def generateFromArray(array, temperature):
        solver = NearestNeighbourSpinStateSolver(temperature, dimension = 2)
        state = np.array(array)
        return RandomNearestNeighbourIsingState2D(state, solver)

 #   def _setNextState(self):
  #      self.nextState = RandomNearestNeighbourIsingState2D(self.currentState, self.spinStateSolver, previousState = self)
            
'''---------------------------------------------------------------------------------------------'''

class SpinStateSolver():
    #When given a spin state, and the state of its nearest neighbours the solver returns
    #the next spin state
    def __init__(self, temperature):
        self.temperature = temperature
    
    def getTemperature(self):
        return self.temperature
    
    def getProbabilityOfTransitions(self, localState):
        return np.vectorize(lambda a,b: self.getProbabilityOfTransition(a,b))
    
    def getProbabilityOfTransition(self, localState):
        #If flipping decreases energy then flip
        #Else flip with a probability exp(-E/KT)

        energyToFlip  = self._calculateEnergyToFlip(localState)
        
        #If flipping would cause a reduction in energy, flip
        if(energyToFlip < 0):
            return 1
        return np.exp( - energyToFlip / (self.getTemperature()))
    
    def _getNextSpinStates(self, localStates):
        coordinateSpins = localStates[...,0]
        pOfTransition = self.getProbabilityOfTransitions(localStates[...,0],localStates[...,1])
        haveFlipped = np.greater(pOfTransition, np.random.rand(*coordinateSpins.shape))
        nextStates  = np.logical_xor(haveFlipped, coordinateSpins)
        #print(nextStates)
        return (haveFlipped, nextStates)
    
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


def plotNMagnetisationsAgainstTemperature():
    pass


if __name__ == '__main__':
    #Lowest temp 1


    sim = NearestNeighbourIsingSimulation2D.generateSystematicNearestNeighbourSim(10, 40, 40)
    #sim.getStateNumber(30000)
    start = time.time()
    #sim.getStateNumber(300000)
    sim.getStateNumber(100)
    end = time.time()
    print(end-start)
    
    IsingSimulationPlotter2D.plotFirstNMagnetisations(100, [sim])
    print("pass")
    #IsingSimulationPlotter2D.animateFirstNFourierTransforms(100, sim)
    print("pass")
    #IsingSimulationPlotter2D.plotAverageFirstNStates(100, [sim])
    #IsingSimulationPlotter2D.animateFirstNStates(50)
    
    start = time.time()
    #sim.getStateNumber(60000)
    IsingSimulationPlotter2D.plotMeanMagnetisationAgainsyTemperatureForDifferentMethods()
   
    end = time.time()
    print(end-start)
