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
    def __init__(self, temperature, dimensions, field = 0):
        self.temperature = temperature
        self.field = field
        self.dimensions = dimensions
        #Calculates the Neighbours of each state and setup a vectorised call to these values.
        self._calculateAllNeighbours()
        self._calculateAllNextNeighbours()
        
    def getDimensions(self):
        return self.dimensions
    
    def getDimension(self):
        return len(self.getDimensions())
    
    def getNumberOfNeighbours(self):
        return 2 * self.getDimension()
    
    def getNumberOfNextNeighbours(self):
        return self.getIndexOfNeighbours(0).size
        #return len(list(itertools.combinations(range(self.getDimension()),2))) + 2 * self.getDimension()
    
    def getTemperature(self):
        return self.temperature
    
    def getField(self):
        return self.field
    
    def getNextSpinStates(self, localStates):
        pOfTransition = self.getProbabilityOfTransition(localStates)
        
        coordinateSpins = localStates[...,0]
        isFlipped = np.greater(pOfTransition, np.random.rand(*coordinateSpins.shape))
        nextStates  = np.logical_xor(isFlipped, coordinateSpins)
        
        return nextStates
    
    def getProbabilityOfTransition(self, localState):
        energyToFlip  = self._calculateEnergyToFlip(localState)
        #Flip with a probability exp(-E/KT)
        return np.exp(-energyToFlip / (self.getTemperature()))
        
    def getEnergyOf(self, localStates):
        return self._calculateEnergyOf(localStates)
    
    def getIndexOfNeighbours(self, indices):
        return self.neighbours[indices]
    
    def getIndexOfNextNeighbours(self, indices):
        return self.nextNeighbours[indices]
    
    def _calculateAllNeighbours(self):
        allIndices = np.arange(0, np.prod(self.getDimensions())).reshape(self.getDimensions())
        neighbours = []
        for axis in range(self.getDimension()):
            neighbours.append(np.roll(allIndices, +1, axis = axis).flatten())
            neighbours.append(np.roll(allIndices, -1, axis = axis).flatten())

        self.neighbours = np.stack(neighbours, axis=-1)
        
    def _calculateAllNextNeighbours(self):
        allIndices = np.arange(0, np.prod(self.getDimensions())).reshape(self.getDimensions())
        nextNeighbours = []
        for axis in range(self.getDimension()):
            nextNeighbours.append(np.roll(allIndices, +2, axis = axis).flatten())
            nextNeighbours.append(np.roll(allIndices, -2, axis = axis).flatten())
        #Find diagonal neighbours 
        for pair in itertools.combinations(range(self.getDimension()),2):
            nextNeighbours.append(np.roll(allIndices, (+1, +1), axis = pair).flatten())
            nextNeighbours.append(np.roll(allIndices, (+1, -1), axis = pair).flatten())
            nextNeighbours.append(np.roll(allIndices, (-1, +1), axis = pair).flatten())
            nextNeighbours.append(np.roll(allIndices, (-1, -1), axis = pair).flatten())

        self.nextNeighbours = np.stack(nextNeighbours, axis=-1)

class NearestNeighbourSpinStateSolver(SpinStateSolver):
    exchangeEnergy = 1
    
    def __init__(self, *args, **kwargs):
        SpinStateSolver.__init__(self, *args, **kwargs)

    def _calculateInteractionEnergy(self, localState):
        coordinateSpin = localState[...,0]
        positiveNeighbours = localState[...,2]
        negativeNeighbours = self.getNumberOfNeighbours() - positiveNeighbours
        interactionEnergy = - 0.5 * self.exchangeEnergy * np.where(coordinateSpin, 1, -1) * (positiveNeighbours - negativeNeighbours)
        return interactionEnergy
    
    def _calculateFieldEnergy(self, localState):
        coordinateSpin = localState[...,0]
        fieldEnergy = - np.where(coordinateSpin, 1, -1) * self.getField()
        return fieldEnergy
    
    def _calculateEnergyOf(self, localState):
        interactionEnergy = self._calculateInteractionEnergy(localState)
        fieldEnergy = self._calculateFieldEnergy(localState)
        return interactionEnergy + fieldEnergy
    
    def _calculateEnergyToFlip(self, localState):
        interactionEnergy = self._calculateInteractionEnergy(localState)
        fieldEnergy = self._calculateFieldEnergy(localState)
        return - 2 * ((2 * interactionEnergy) + fieldEnergy)
    
class ParameterisedSpinStateSolver(SpinStateSolver):
    
    def __init__(self, possibleParameters = np.array(False)):
        self.possibleParameters = np.array(possibleParameters)
        
    def getParameter(self, position):
        return np.take(self.possibleParameters, position, mode = 'wrap')
    
#Extends NNSSS by adding H dependance
class FieldDependantNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver, ParameterisedSpinStateSolver):
    def __init__(self, field = None, possibleParameters = None, **kwargs):
        NearestNeighbourSpinStateSolver.__init__(self, **kwargs, field = None)
        ParameterisedSpinStateSolver.__init__(self, possibleParameters)
    
    def _calculateFieldEnergy(self, localState):
        coordinateSpin = localState[...,0]
        localH = localState[..., 1]
        fieldEnergy = - np.where(coordinateSpin, 1, -1) * localH
        return fieldEnergy
   
    def getField(self, position):
        return self.getParameter(position)

#Extends NNSSS by adding temperature dependance
class TemperatureDependantNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver, ParameterisedSpinStateSolver):
    
    def __init__(self, temperature = None, possibleParameters = None, **kwargs):
        NearestNeighbourSpinStateSolver.__init__(self, temperature = None, **kwargs)
        ParameterisedSpinStateSolver.__init__(self, possibleParameters)
    
    def getProbabilityOfTransition(self, localState):
        energyToFlip  = self._calculateEnergyToFlip(localState)
        #Else flip with a probability exp(-E/KT)
        localT = localState[...,1]
        return np.exp( - energyToFlip / (localT))
    
    def getTemperature(self, position):
        return self.getParameter(position)
    
#Extends NNSSS by adding Next neighbour interaction
class NextNearestNeighbourSpinStateSolver(NearestNeighbourSpinStateSolver):
    
    def __init__(self, nextExchangeEnergy = 0.5, **kwargs):
        self.nextExchangeEnergy = nextExchangeEnergy
        super().__init__(**kwargs)

    def _calculateInteractionEnergy(self, localState):
        coordinateSpin = localState[...,0]
        positiveNextNeighbours = localState[...,1]
        positiveNeighbours = localState[...,2]
        negativeNeighbours = self.getNumberOfNeighbours() - positiveNeighbours
        negativeNextNeighbours = self.getNumberOfNextNeighbours() - positiveNextNeighbours
        interactionEnergy = - 0.5 * self.exchangeEnergy * np.where(coordinateSpin, 1, -1) * (positiveNeighbours - negativeNeighbours)
        nextinteractionEnergy = - 0.5 * self.nextExchangeEnergy * np.where(coordinateSpin, 1, -1) * (positiveNextNeighbours - negativeNextNeighbours)
        return interactionEnergy + nextinteractionEnergy