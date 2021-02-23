# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:00:16 2020

@author: Matt
"""


import numpy as np
import itertools

class SpinStateSolver():
    #When given a spin state, and the state of its nearest neighbours the solver returns
    #the next spin state
    def __init__(self, temperature):
        self.temperature = temperature
        #Calculates the probability and energy of each state and setup a vectorised call to these values.
        self._setupSpinProbabilityDict()
        self._setupEnergyDict()
    
    def getTemperature(self):
        return self.temperature
    
    def _setupSpinProbabilityDict(self):
        probabilityDict = {}
        for localState in self.getAllPossibleLocalStates():
            probabilityDict[(localState[0], localState[1], localState[2])] = self._calculateProbabilityOfTransition(localState)
        #self._getProbabilityOfTransitions = lambda a : np.apply_along_axis(lambda x : probabilityDict.get(tuple(x)), -1, a)
        self._getProbabilityOfTransitions = np.vectorize(lambda a,b,c : probabilityDict.get((a,b,c)))
        
    def _setupEnergyDict(self):
        energyDict = {}
        for localState in self.getAllPossibleLocalStates():
            energyDict[(localState[0], localState[1], localState[2])] = self._calculateEnergyOf(localState)

        self._getEnergyOf = np.vectorize(lambda a,b,c : energyDict.get((a,b,c)))
    
    def _calculateProbabilityOfTransition(self, localState):
        energyToFlip  = self._calculateEnergyToFlip(localState)
        
        #If flipping would cause a reduction in energy, flip
        
        if(energyToFlip < 0):
            return 1
        
        #Else flip with a probability exp(-E/KT)
        return np.exp( - energyToFlip / (self.getTemperature()))
    
    def getNextSpinStates(self, localStates):
        pOfTransition = self.getProbabilityOfTransitions(localStates)
        
        coordinateSpins = localStates[...,0]
        isFlipped = np.greater(pOfTransition, np.random.rand(*coordinateSpins.shape))
        nextStates  = np.logical_xor(isFlipped, coordinateSpins)
        
        return nextStates
        
    def getEnergyOf(self, localStates):
        return self._getEnergyOf(localStates[...,0], localStates[...,1], localStates[...,2])
    
    def getProbabilityOfTransitions(self, localStates):
        return self._getProbabilityOfTransitions(localStates[...,0], localStates[...,1], localStates[...,2])

class NearestNeighbourSpinStateSolver(SpinStateSolver):
    
    def __init__(self, temperature, dimension, possibleParameters = [False]):
        self.numberOfNeighbours = 2 * dimension  # N dimension, 2*N sides to the cube
        self.exchangeEnergy = 1
        self.possibleParameters = possibleParameters
        
        super().__init__(temperature)
        
    def getDimension(self):
        return int(self.numberOfNeighbours / 2)
    
    def getAllPossibleLocalStates(self):
        possibleSpins = [True,False]
        possibleParameters = np.unique(self.possibleParameters)
        possibleNeighbours = range(self.numberOfNeighbours + 1)
        possibleStates = np.array(np.meshgrid(possibleSpins, possibleParameters, possibleNeighbours)).T.reshape(-1,3)
        return possibleStates
    
    def _calculateInteractionEnergy(self, localState):
        (coordinateSpin, _, positiveNeighbours) = localState
        negativeNeighbours = self.numberOfNeighbours - positiveNeighbours
        interactionEnergy = - 0.5 * self.exchangeEnergy * (1 if coordinateSpin else -1) * (positiveNeighbours - negativeNeighbours)
        return interactionEnergy
    
    def _calculateFieldEnergy(self, localState):
        return 0
    
    def _calculateEnergyOf(self, localState):
        interactionEnergy = self._calculateInteractionEnergy(localState)
        fieldEnergy = self._calculateFieldEnergy(localState)
        return interactionEnergy + fieldEnergy
    
    def _calculateEnergyToFlip(self, localState):
        interactionEnergy = self._calculateInteractionEnergy(localState)
        fieldEnergy = self._calculateFieldEnergy(localState)
        return - 2 * ((2 * interactionEnergy) + fieldEnergy)
    
    def getParameter(self, position):
        return self.possibleParameters[position % self.possibleParameters.size]
    
#Extends NNSSS by adding H dependance
class FieldDependantNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver):
    def __init__(self, temperature, dimension, possibleParameters = [0]):
        super().__init__(temperature = temperature, dimension = dimension, possibleParameters = possibleParameters)
    
    def _calculateFieldEnergy(self, localState):
        (coordinateSpin, localH, _) = localState
        fieldEnergy = - coordinateSpin * localH
        return fieldEnergy

    
#Extends NNSSS by adding temperature dependance
class TemperatureDependantNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver):
    
    def __init__(self, temperature, dimension, possibleParameters):
        super().__init__(temperature = temperature, dimension = dimension, possibleParameters = possibleParameters)
    
    def _calculateProbabilityOfTransition(self, localState):
        (coordinateSpin, localT, positiveNeighbours) = localState
        
        energyToFlip  = self._calculateEnergyToFlip(localState)
        #If flipping would cause a reduction in energy, flip
        if(energyToFlip < 0):
            return 1
        #Else flip with a probability exp(-E/KT)
        return np.exp( - energyToFlip / (localT))
    
class NextNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver):
    
    def __init__(self, temperature, dimension, nextExchangeEnergy = 0.5):
        self.numberOfNeighbours = 2 * dimension  # N dimension, 2*N sides to the cube
        self.numberOfNextNeighbours = len(list(itertools.combinations(range(dimension),2))) + 2 * dimension
        self.nextExchangeEnergy = nextExchangeEnergy
        super().__init__(temperature = temperature, dimension = dimension, possibleParameters = range(self.numberOfNextNeighbours + 1))

    def _calculateEnergyOf(self, localState):
        (coordinateSpin, positiveNextNeighbours, positiveNeighbours) = localState
        localEnergy = - self.exchangeEnergy * (1 if coordinateSpin else -1) * (positiveNeighbours - (self.numberOfNeighbours / 2))
        nextLocalEnergy = - self.nextExchangeEnergy * (1 if coordinateSpin else -1) * (positiveNextNeighbours - (self.numberOfNextNeighbours / 2))
        return localEnergy + nextLocalEnergy