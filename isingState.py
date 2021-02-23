# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:58:38 2020

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from spinStateSolver import NearestNeighbourSpinStateSolver, NextNearestNeighbourSpinStateSolver, FieldDependantNearestNeighbourSpinStateSolver, TemperatureDependantNearestNeighbourSpinStateSolver
import itertools

class IsingState():
    dimension  = None
    solverType = None
    def __init__(self, state, spinStateSolver, previousState = None):
        self.initialSpins = state                                      #Array representing the true ising state
        self.currentSpins = np.array(self.getAllInitialSpins())        #Array to store the next states spins
        self.spinStateSolver = spinStateSolver
        self.nextState = None
        self.previousState = previousState
        
        if not isinstance(spinStateSolver, self.solverType):
            raise Exception('Spin solver not {}'.format(self.solverType))
        if spinStateSolver.getDimensions() != self.getDimensions():
            raise Exception('Spin solver has wrong dimension: {} required: {}'.format(spinStateSolver.getDimensions(), self.getDimensions()))
        
    def getDimensions(self):
        return self.getAllInitialSpins().shape
    
    def getDimension(self):
        return self.dimension

    def getNumberOfSpins(self):
        return self.getAllInitialSpins().size

    def getNumberOfSpinUp(self):
        return np.count_nonzero(self.getAllInitialSpins())
    
    def getNumberOfSpinDown(self):
        return self.getNumberOfSpins() - self.getNumberOfSpinUp()
    
    def getMagnetisation(self):
        # Spin up - spin down, saves calling np.count_Nonzero twice
        return (2 * self.getNumberOfSpinUp() - self.getNumberOfSpins()) / self.getNumberOfSpins()
    
    def getAllInitialSpins(self):
        return self.initialSpins
    
    def getAllCurrentSpins(self):
        return self.currentSpins
    
    def getAllInitialParameters(self):
        #Get parameter for all positions in the lattuice
        return np.tile(False, reps = self.getDimensions())
    
    def getAllCurrentParameters(self):
        return self.getAllInitialParameters()
    
    def getAllEnergies(self):
        return self.spinStateSolver.getEnergyOf(self.getAllInitialLocalStates())
    
    def getAllInitialPositiveNeighbours(self):
        # Find the neighbouring elements
        neighbours = []
        for axis in range(self.getDimension()):
            neighbours.append(np.roll(self.getAllInitialSpins(), +1, axis = axis))
            neighbours.append(np.roll(self.getAllInitialSpins(), -1, axis = axis))
        # Find the total spin
        return np.sum(neighbours, axis=0, dtype=int)
    
    def getAllCurrentPositiveNeighbours(self):
        # Find the neighbouring elements
        neighbours = []
        for axis in range(self.getDimension()):
            neighbours.append(np.roll(self.getAllCurrentSpins(), +1, axis = axis))
            neighbours.append(np.roll(self.getAllCurrentSpins(), -1, axis = axis))
        # Find the total spin
        return np.sum(neighbours, axis=0, dtype=int)
    
    def getAllInitialPositiveNextNeighbours(self):
        nextNeighbours = []
        for axis in range(self.getDimension()):
            nextNeighbours.append(np.roll(self.getAllCurrentSpins(), +2, axis = axis))
            nextNeighbours.append(np.roll(self.getAllCurrentSpins(), -2, axis = axis))
        #Find diagonal neighbours 
        for pair in itertools.combinations(range(self.getDimension()),2):
            nextNeighbours.append(np.roll(self.getAllCurrentSpins(), (+1, +1), axis = pair))
            nextNeighbours.append(np.roll(self.getAllCurrentSpins(), (+1, -1), axis = pair))
            nextNeighbours.append(np.roll(self.getAllCurrentSpins(), (-1, +1), axis = pair))
            nextNeighbours.append(np.roll(self.getAllCurrentSpins(), (-1, -1), axis = pair))
        return np.sum(nextNeighbours, axis=0, dtype=int)
    
    def getAllCurrentPositiveNextNeighbours(self):
        nextNeighbours = []
        for axis in range(self.getDimension()):
            nextNeighbours.append(np.roll(self.getAllInitialSpins(), +2, axis = axis))
            nextNeighbours.append(np.roll(self.getAllInitialSpins(), -2, axis = axis))
        #Find diagonal neighbours
        for pair in itertools.combinations(range(self.getDimension()),2):
            nextNeighbours.append(np.roll(self.getAllInitialSpins(), (+1, +1), axis = pair))
            nextNeighbours.append(np.roll(self.getAllInitialSpins(), (+1, -1), axis = pair))
            nextNeighbours.append(np.roll(self.getAllInitialSpins(), (-1, +1), axis = pair))
            nextNeighbours.append(np.roll(self.getAllInitialSpins(), (-1, -1), axis = pair))
        return np.sum(nextNeighbours, axis=0, dtype=int)
    
    def getInitialSpin(self, indices):
        return np.take(self.getAllInitialSpins(), indices)
    
    def getCurrentSpin(self, indices):
        return np.take(self.getAllCurrentSpins(), indices)
    
    gIP = np.vectorize(lambda x: False)
    def getInitialParameter(self, indices):
        return IsingState.gIP(indices)

    def getCurrentParameter(self, indices):
        return self.getInitialParameter(indices)
    
    def getEnergy(self, indices):
        return np.take(self.getAllEnergies(), indices)
        
    def getInitialPositiveNeighbours(self, indices):
        indexOfNeighbours = self.getIndexOfNeighbours(indices)
        spinOfNeighbours = self.getInitialSpin(indexOfNeighbours)
        return np.sum(spinOfNeighbours, axis = -1)
    
    def getCurrentPositiveNeighbours(self, indices):
        indexOfNeighbours = self.getIndexOfNeighbours(indices)
        spinOfNeighbours = self.getCurrentSpin(indexOfNeighbours)
        return np.sum(spinOfNeighbours, axis = -1)
    
    def getInitialPositiveNextNeighbours(self, indices):
        indexOfNextNeighbours = self.getIndexOfNextNeighbours(indices)
        spinOfNextNeighbours = self.getInitialSpin(indexOfNextNeighbours)
        return np.sum(spinOfNextNeighbours, axis = -1)
    
    def getCurrentPositiveNextNeighbours(self, indices):
        indexOfNextNeighbours = self.getIndexOfNextNeighbours(indices)
        spinOfNextNeighbours = self.getCurrentSpin(indexOfNextNeighbours)
        return np.sum(spinOfNextNeighbours, axis = -1)
    
    def getIndexOfCoordinate(self, coordinates):
        return np.ravel_multi_index(coordinates,
                                    dims = self.getDimensions(),
                                    mode = "wrap")
    
    def getCoordinateOfIndex(self, indices):
        return np.stack(np.unravel_index(indices, shape = self.getDimensions()), axis = -1)
    
    def getIndexOfNeighbours(self, indices):
        return self.spinStateSolver.getIndexOfNeighbours(indices)
    
    def getIndexOfNextNeighbours(self, indices):
        return self.spinStateSolver.getIndexOfNextNeighbours(indices)
    
    def getEnergyOf(self, indices):
        return self.spinStateSolver.getEnergyOf(self.getInitialLocalState(indices))
    
    def getTotalEnergy(self):
        return np.sum(self.getAllEnergies())
    
    def getTemperature(self):
        return self.spinStateSolver.getTemperature()
    
    def getField(self):
        return self.spinStateSolver.getField()
    
    def getPreviousState(self):
        return self.previousState
    
    def getAllIndex(self):
        return np.arange(0, self.getNumberOfSpins())
    
    def getRandomIndex(self, size = None):
        return np.random.randint(0, self.getNumberOfSpins(), size = size)
    
    def getAllCoordinates(self):
        return np.argwhere(self.getAllInitialSpins())
    
    def getRandomCoordinate(self, size = None):
        return np.unravel_index(self.getRandomIndex(size = size),
                                self.getDimensions())

    def getStateAsArray(self):
        return np.array(self.getAllInitialSpins())
    
    def getStateAsDFT(self, removeCentre = True):
        ft = np.fft.fftn(self.getAllInitialSpins(), norm=None)
        if removeCentre:
            np.put(ft, 0, 0)
        return np.fft.fftshift(ft)
        
    def _seperateNeighbouringIndicies(self, indices):
        #The neighbourhood surrounding an index is
        #all points which will be effected if index is updated
        splits = []
        isInCurrentRegion = np.zeros(self.getNumberOfSpins())
        for pos in range(indices.size):
            index = indices[pos]
            if(isInCurrentRegion[index]):
                #Must split array here
                splits.append(pos)
                isInCurrentRegion = np.zeros(self.getNumberOfSpins())
            np.put(isInCurrentRegion,index, True)
            np.put(isInCurrentRegion,self.getIndexInNeighbourhood(index), True)
        return np.split(indices, splits)
    
    def _getNextState(self):
        if(self.nextState is None):
            self._calculateNextState()
        return self.nextState
    
    def _calculateNextState(self):        
        regions = self._getRegionsToUpdate()
        for regionIndices in regions:
            self._updateSpinRegion(regionIndices)
        self._setNextState()
        
    def _updateSpinRegion(self, regionIndices):
        regionLocalStates = self.getCurrentLocalState(regionIndices)
        regionNextSpins = self.spinStateSolver.getNextSpinStates(regionLocalStates)
        np.put(self.currentSpins, regionIndices, regionNextSpins) 
    
    def _getNextKwargs(self):
        return {'state' : self.currentSpins,
                'spinStateSolver' : self.spinStateSolver,
                'previousState' : self,}
    
    def _setNextState(self):
        self.nextState = type(self)(**self._getNextKwargs())
    
    @classmethod
    def generateFromArray(cls, array, temperature, field = 0, solverKwargs = {}, **kwargs):
        if(cls.dimension == None):
            raise Exception("Class has no definate dimension")
        if(cls.solverType == None):
            raise Exception("Class has no definate solver type")
        if(len(array.shape) != cls.dimension):
            raise Exception("Array has incorrect dimensions")
        
        state = np.array(array)
        solver = cls.solverType(temperature = temperature,
                                field = field,
                                dimensions = array.shape,
                                **solverKwargs)

        return cls(state = state,
                   spinStateSolver = solver,
                   previousState = None,
                   **kwargs)
    
    @classmethod
    def generateRandomState(cls, dimensions, bias = 0, **kwargs):
        initialSpins = np.random.choice([True, False], p = (.5 + bias, .5 - bias), size = dimensions)
        return cls.generateFromArray(array = initialSpins, **kwargs)
        
    @classmethod
    def generateOrderedState(cls, initialSpinUp = True, **kwargs):
        return cls.generateRandomState(bias = -0.5 + initialSpinUp, **kwargs)
    
    def __str__(self):
        return self.getAllInitialSpins().__str__()
    
class ForgetfulIsingState(IsingState):
    
    def _getNextState(self):
        nextState = super()._getNextState()
        self.nextState = None
        # Forgets about the next state
        return nextState
    
    def _getNextKwargs(self):
        kwargs = super()._getNextKwargs()
        # Forgets about the previous state
        kwargs['previousState'] = None
        return kwargs

class IsingState1D(IsingState):
    dimension = 1

class IsingState2D(IsingState):
    dimension = 2
    
class IsingState3D(IsingState):
    dimension = 3
    
class InterpolatedIsingState():
    
    def __init__(self, baseState, interpolatedSpins, stepsFromBaseState):
        self.baseState = baseState
        self.interpolatedSpins = interpolatedSpins
        self.stepsFromBaseState = stepsFromBaseState #baseState has steps from base state == 0
        
    def __getattr__(self, name):
        return self.baseState.__getattribute__(name)
    
    def getAllInitialSpins(self):
        return self.interpolatedSpins
    
    def getStateNumber(self, N):
        if(N==0):
            return self
        return self.baseState.getStateNumber(N + self.stepsFromBaseState)

class RandomIsingState(IsingState):
    
    def getNumberOfUpdatedSpins(self):
        return 1
    
    def getStateNumber(self, N):
        currentState = self
        for _ in range(N):
                currentState = currentState._getNextState()
        return currentState
    
    def _getRegionsToUpdate(self):
        #Only updates a single random coordinate
        return [self.getRandomIndex()]
    
#Ising state where more than one spin is updated at each step
#By default N^2 spins are updated at each step
class VectorisedIsingState(IsingState):
    
    def getNumberOfUpdatedSpins(self):
        return self.getNumberOfSpins()
    
    def getStateNumber(self, N):
        if(N < self.getNumberOfUpdatedSpins()):
            return self.getIntermediateState(N)
        currentState = self
        for x in range(N // self.getNumberOfUpdatedSpins()):
            currentState = currentState._getNextState()
        return currentState.getStateNumber(N % self.getNumberOfUpdatedSpins())
    
    def getIntermediateState(self, N):
        if(N==0):
            return self
        interpolatedSpins = self._getInterpolatedSpins(N)
        return InterpolatedIsingState(self, interpolatedSpins, N)
    
    def getIntermediateStateAsArray(self, N):
        return self._getInterpolatedSpins(N)
        
    def _getInterpolatedSpins(self, N):
        #Attemps to find the intermediate states,
        #However does not take into account doubly flipped indices
        if(self.nextState is None): #If not already calculated
            self._calculateNextState()
        interpolatedSpins = np.copy(self.getAllCurrentSpins())
        indexToUpdate = self.orderOfCalculation[:self.getNumberOfSpins() - N]
        np.put(interpolatedSpins,indexToUpdate, np.take(self.getAllInitialSpins(), indexToUpdate))
        return interpolatedSpins
    
    def _getRegionsToUpdate(self):
        self._generateOrderOfCalculation()
        regions = self._seperateNeighbouringIndicies(self.orderOfCalculation)
        return regions
            
class VectorisedRandomIsingState(VectorisedIsingState, RandomIsingState):
    
    def _generateOrderOfCalculation(self):
        #Updates N^2 random coordinates
        randomCoordinateIndex = self.getRandomIndex(size = self.getNumberOfUpdatedSpins())
        self.orderOfCalculation = randomCoordinateIndex

class SystematicIsingState(VectorisedIsingState):
    
    def _generateOrderOfCalculation(self):
        #Updates all coordinates once
        allCoordinateIndex = self.getAllIndex()
        random.shuffle(allCoordinateIndex)
        self.orderOfCalculation = allCoordinateIndex #Used when interpolating between states
    
#Overrides _splitIndiciesIntoNeighbourhoods to split into a given number of sub arrays
#This process is much quicker than that of the SystematicIsingState but leaves the
#Possibility of having two neighbouring states in the same region
class FastSystematicIsingState(SystematicIsingState):
    
    def __init__(self, numberOfSplits = None):
        self.numberOfSplits = self.getDimensions()[0] if numberOfSplits is None else numberOfSplits
    
    def setNumberOfSplits(self, N):
        if(N > self.getNumberOfSpins()):
            raise Exception("number of splits cannot be greater than number of elements. Try = {}, for {} elements".format(N, self.getNumberOfSpins()))
        self.numberOfSplits = N
        
    def _seperateNeighbouringIndicies(self, indices):
        return np.array_split(indices, self.numberOfSplits)
    
    def _getNextKwargs(self):
        kwargs = super()._getNextKwargs()
        kwargs['numberOfSplits'] = self.numberOfSplits
        return kwargs
    
class TemperatureVaryingIsingState(IsingState):
    
    def __init__(self, position = 0):
        self.position = position
        
    def getTemperature(self):
        return self.spinStateSolver.getField(self.position)
    
    def getAllInitialParameters(self):
        return np.tile(self.spinStateSolver.getParameter(self.position),
                       reps = self.getDimensions())

    def getInitialParameter(self, indices):
        return np.vectorize(lambda _:self.spinStateSolver.getParameter(self.position))(indices)
    
    def _getNextKwargs(self):
        kwargs = super()._getNextKwargs()
        kwargs['position'] = self.position + 1
        return kwargs
    
    @classmethod
    def generateFromArray(cls, temperatures = [0], solverKwargs = {}, **kwargs):
        solverKwargs["possibleParameters"] = temperatures
        return super().generateFromArray(temperature = None, solverKwargs = solverKwargs, **kwargs)
    
    @classmethod
    def generateOscillatingStateFromArray(cls, startTemp, endTemp, numberOfSteps, **kwargs):
        if(startTemp < endTemp):
            temperatures = np.linspace(startTemp, endTemp, num = numberOfSteps)
        else:
            temperatures = np.linspace(endTemp, startTemp, num = numberOfSteps)[::-1]
        return cls.generateFromArray(temperatures = temperatures, **kwargs)
    
    @classmethod
    def generateRandomOscillatingState(cls, startTemp, endTemp, numberOfSteps, **kwargs):
        if(startTemp < endTemp):
            temperatures = np.linspace(startTemp, endTemp, num = numberOfSteps)
        else:
            temperatures = np.linspace(endTemp, startTemp, num = numberOfSteps)[::-1]
        return cls.generateRandomState(temperatures = temperatures, **kwargs)

    
class MagnetisedIsingState(IsingState):
    
    def __init__(self, position = 0):
        self.position = position
        
    def getField(self):
        return self.spinStateSolver.getField(self.position)
    
    def getAllInitialParameters(self):
        return np.tile(self.spinStateSolver.getParameter(self.position),
                       reps = self.getDimensions())
    
    def getInitialParameter(self, indices):
        return np.vectorize(lambda x:self.spinStateSolver.getParameter(self.position))(indices)

    def _getNextKwargs(self):
        kwargs = super()._getNextKwargs()
        kwargs['position'] = self.position + 1
        return kwargs
    
    @classmethod
    def generateFromArray(cls, fields = [0], solverKwargs = {}, **kwargs):
        solverKwargs["possibleParameters"] = fields
        return super().generateFromArray(solverKwargs = solverKwargs, **kwargs)
    
    @classmethod
    def generateOscillatingStateFromArray(cls, amplitude, numberOfSteps, **kwargs):
        fields    = np.array([amplitude  * np.cos(x * 2 * np.pi / numberOfSteps) for x in range(numberOfSteps)])
        return cls.generateFromArray(fields = fields, **kwargs) 
        
    @classmethod
    def generateRandomOscillatingState(cls, amplitude, numberOfSteps, **kwargs):
        fields    = np.array([amplitude  * np.cos(x * 2 * np.pi / numberOfSteps) for x in range(numberOfSteps)])
        return cls.generateRandomState(fields = fields, **kwargs)

class NearestNeighbourIsingState(IsingState):
    solverType = NearestNeighbourSpinStateSolver
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    def getInitialLocalState(self, indices):
        coordinateSpins = self.getInitialSpin(indices)
        parameters = self.getInitialParameter(indices)
        positiveNeibouringSpins = self.getInitialPositiveNeighbours(indices)
        
        return np.stack((coordinateSpins, parameters, positiveNeibouringSpins), axis=-1)
    
    def getCurrentLocalState(self, indices):
        coordinateSpins = self.getCurrentSpin(indices)
        parameters = self.getCurrentParameter(indices)
        positiveNeibouringSpins = self.getCurrentPositiveNeighbours(indices)

        return np.stack((coordinateSpins, parameters, positiveNeibouringSpins), axis=-1)
    
    def getAllInitialLocalStates(self):
        coordinateSpins = self.getAllInitialSpins()
        parameters = self.getAllInitialParameters()
        positiveNeibouringSpins = self.getAllInitialPositiveNeighbours()

        return np.stack((coordinateSpins, parameters, positiveNeibouringSpins), axis=-1)
        
    def getAllCurrentLocalStates(self):
        coordinateSpins = self.getAllCurrentSpins()
        parameters = self.getAllCurrentParameters()
        positiveNeibouringSpins = self.getAllCurrentPositiveNeighbours()

        return np.stack((coordinateSpins, parameters, positiveNeibouringSpins), axis=-1)
    
    def getIndexInNeighbourhood(self, indices):
        return self.getIndexOfNeighbours(indices)

class NextNearestNeighbourIsingState(NearestNeighbourIsingState):
    solver = NextNearestNeighbourSpinStateSolver
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def getAllInitialParameters(self):
        return self.getAllInitialPositiveNextNeighbours()
        
    def getAllCurrentParameters(self):
        return self.getAllCurrentPositiveNextNeighbours()
    
    def getIndexInNeighbourhood(self, indices):
        return np.concatenate((self.getIndexOfNeighbours(indices), self.getIndexOfNextNeighbours(indices)), axis = -1)

class RandomNearestNeighbourIsingState1D(IsingState1D, RandomIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)
    
class VectorisedRandomNearestNeighbourIsingState1D(IsingState1D, VectorisedRandomIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)    
        
class SystematicNearestNeighbourIsingState1D(IsingState1D, SystematicIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs) 

class RandomNearestNeighbourForgetfulIsingState1D(ForgetfulIsingState, RandomNearestNeighbourIsingState1D):
    
    def __init__(self, **kwargs):
        RandomNearestNeighbourIsingState1D.__init__(self, **kwargs)

class SystematicNearestNeighbourIsingState2D(IsingState2D, SystematicIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)

class FastSystematicNearestNeighbourIsingState2D(IsingState2D, FastSystematicIsingState, NearestNeighbourIsingState):
    def __init__(self, numberOfSplits = None, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)
        FastSystematicIsingState.__init__(self, numberOfSplits = numberOfSplits)
    
class RandomNearestNeighbourIsingState2D(IsingState2D, RandomIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)
        
class VectorisedRandomNearestNeighbourIsingState2D(IsingState2D, VectorisedRandomIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)
        
class RandomNearestNeighbourMagnetisedIsingState2D(MagnetisedIsingState, RandomNearestNeighbourIsingState2D):
    solverType = FieldDependantNearestNeighbourSpinStateSolver
    
    def __init__(self, position = 0, **kwargs):
        MagnetisedIsingState.__init__(self, position)
        RandomNearestNeighbourIsingState2D.__init__(self, **kwargs)
        
class VectorisedRandomNearestNeighbourMagnetisedIsingState2D(MagnetisedIsingState, VectorisedRandomNearestNeighbourIsingState2D):
    solverType = FieldDependantNearestNeighbourSpinStateSolver
    
    def __init__(self, position = 0, **kwargs):
        MagnetisedIsingState.__init__(self, position)
        VectorisedRandomNearestNeighbourIsingState2D.__init__(self, **kwargs)
    
class RandomNearestNeighbourMagnetisedForgetfulIsingState2D(ForgetfulIsingState, RandomNearestNeighbourMagnetisedIsingState2D):
    
    def __init__(self, **kwargs):
        RandomNearestNeighbourMagnetisedIsingState2D.__init__(self, **kwargs)
        
class RandomNearestNeighbourTemperatureVaryingIsingState2D(TemperatureVaryingIsingState, RandomNearestNeighbourIsingState2D):
    solverType = TemperatureDependantNearestNeighbourSpinStateSolver
    
    def __init__(self, position = 0, **kwargs):
        TemperatureVaryingIsingState.__init__(self, position)
        RandomNearestNeighbourIsingState2D.__init__(self, **kwargs)
        
class VectorisedRandomNearestNeighbourTemperatureVaryingIsingState2D(TemperatureVaryingIsingState, VectorisedRandomNearestNeighbourIsingState2D):
    solverType = TemperatureDependantNearestNeighbourSpinStateSolver
    
    def __init__(self, position = 0, **kwargs):
        TemperatureVaryingIsingState.__init__(self, position)
        VectorisedRandomNearestNeighbourIsingState2D.__init__(self, **kwargs)
        
class RandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D(ForgetfulIsingState, RandomNearestNeighbourTemperatureVaryingIsingState2D):
    
    def __init__(self, **kwargs):
        RandomNearestNeighbourTemperatureVaryingIsingState2D.__init__(self, **kwargs)
        
class VectorisedRandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D(ForgetfulIsingState, VectorisedRandomNearestNeighbourTemperatureVaryingIsingState2D):
    
    def __init__(self, **kwargs):
        VectorisedRandomNearestNeighbourTemperatureVaryingIsingState2D.__init__(self, **kwargs)
        
class SystematicNearestNeighbourTemperatureVaryingIsingState2D(TemperatureVaryingIsingState, SystematicNearestNeighbourIsingState2D):
    solverType = TemperatureDependantNearestNeighbourSpinStateSolver
    def __init__(self, position = 0, **kwargs):
        TemperatureVaryingIsingState.__init__(self, position)
        SystematicNearestNeighbourIsingState2D.__init__(self, **kwargs)

class RandomNearestNeighbourForgetfulIsingState2D(ForgetfulIsingState, RandomNearestNeighbourIsingState2D):
    
    def __init__(self, **kwargs):
        RandomNearestNeighbourIsingState2D.__init__(self, **kwargs)
        
class SystematicNearestNeighbourForgetfulIsingState2D(ForgetfulIsingState, SystematicNearestNeighbourIsingState2D):
    
    def __init__(self, **kwargs):
        SystematicNearestNeighbourIsingState2D.__init__(self, **kwargs)
        
class FastSystematicNearestNeighbourForgetfulIsingState2D(ForgetfulIsingState, FastSystematicNearestNeighbourIsingState2D):
    
    def __init__(self, **kwargs):
        FastSystematicNearestNeighbourIsingState2D.__init__(self, **kwargs)
    
class RandomNearestNeighbourIsingState3D(IsingState3D, RandomIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)
    
class RandomNearestNeighbourForgetfulIsingState3D(ForgetfulIsingState, RandomNearestNeighbourIsingState3D):
    
    def __init__(self, **kwargs):
        RandomNearestNeighbourIsingState3D.__init__(self, **kwargs)
        
class VectorisedRandomNearestNeighbourIsingState3D(IsingState3D, VectorisedRandomIsingState, NearestNeighbourIsingState):
    def __init__(self, **kwargs):
        NearestNeighbourIsingState.__init__(self, **kwargs)