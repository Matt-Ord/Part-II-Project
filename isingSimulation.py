# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:57:29 2020

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from isingState import SystematicIsingState, VectorisedRandomNearestNeighbourIsingState2D, SystematicNearestNeighbourIsingState2D, FastSystematicNearestNeighbourIsingState2D
from isingSimulationPlotter import *
import time
import math
import statsmodels.tsa.stattools
import itertools
import pandas as pd

class IsingSimulation():
    def __init__(self, initialIsingState):
        self.firstIsingState = initialIsingState
        
    def _getInitialState(self):
        return self.firstIsingState

    def getStateNumber(self, N):
        return self._getInitialState().getStateNumber(N)
    
    def getInitialTemperature(self):
        return self._getInitialState().getTemperature()

    def getDimensions(self):
        return self._getInitialState().getDimensions()

    def getDimension(self):
        return self._getInitialState().getDimension()
    
    def getTypeOfState(self):
        return type(self._getInitialState())

    def getNumberOfSpins(self):
        # Returns number of spins in each ising state
        return self._getInitialState().getNumberOfSpins()

    def _applyToStates(self, function, numberOfStates, distanceBetweenStates, statesToIgnore = 0):
        # Equivalent to but faster than [function(self.getStateNumber(N)) for N in indexOfStates]
        vals = []
        currentState = self._getInitialState()
        currentState = currentState.getStateNumber(statesToIgnore * distanceBetweenStates)
        for x in range(numberOfStates):
            vals.append(function(currentState))
            currentState = currentState.getStateNumber(distanceBetweenStates)
        return np.array(vals)

    def getMagnetisations(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getMagnetisation(), numberOfStates, distanceBetweenStates, statesToIgnore)

    def getStates(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getStateAsArray()  , numberOfStates, distanceBetweenStates, statesToIgnore)

    def getFourierTransforms(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getStateAsDFT()    , numberOfStates, distanceBetweenStates, statesToIgnore)
    
    def getEnergies(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getTotalEnergy()   , numberOfStates, distanceBetweenStates, statesToIgnore)
    
    def getFields(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getField()  , numberOfStates, distanceBetweenStates)
    
    def getTemperatures(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getTemperature()   , numberOfStates, distanceBetweenStates, statesToIgnore)

    def getSpinUp(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        return self._applyToStates(lambda x: x.getNumberOfSpinUp(), numberOfStates, distanceBetweenStates, statesToIgnore)
    
    def getPeriodAveragedMagnetisations(self, period, numberOfPeriods = 10):
        lengthOfSample = numberOfPeriods * period
        magnetisations = self.getMagnetisations(lengthOfSample)
        splitMagnetisations = np.split(magnetisations, numberOfPeriods)
        periodAverage = np.average(np.stack(splitMagnetisations, axis = -1), axis = -1)
        return periodAverage
    
    def getAreaOfHysteresis(self, period, numberOfPeriods = 10, fields = None):
        magnetisations = self.getPeriodAveragedMagnetisations(period, numberOfPeriods)
        fields = self.getAverageFields(period) if fields is None else fields
        def area(x,y):
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        areaOfHysteresis = area(fields, magnetisations)
        return areaOfHysteresis
    
    def getEquilibriumTime(self, lengthOfSample, distanceBetweenSamples = 1, tolerance = 0, equilibriumMagnetisation = None):
        magnetisations = self.getMagnetisations(lengthOfSample, distanceBetweenSamples)
        equilibriumMagnetisation = self.getAverageMagnetisation(lengthOfSample, distanceBetweenStates = distanceBetweenSamples) if equilibriumMagnetisation == None else equilibriumMagnetisation
        
        hasConverged = np.greater(tolerance, np.abs(magnetisations - equilibriumMagnetisation))

        #plt.plot(np.abs(np.gradient(np.polyval(fit, range(lengthOfSample)))))
        yetToConverge = itertools.takewhile(lambda x: not x, hasConverged)        
        equilibriumTime = sum(1 for _ in yetToConverge)
        return (equilibriumTime * distanceBetweenSamples) / self.getNumberOfSpins()
                       
    def getAutocorrelation(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0, width = None):
        width = int(numberOfStates - statesToIgnore) if width == None else width
        magnetisations = self.getMagnetisations(numberOfStates, distanceBetweenStates = distanceBetweenStates, statesToIgnore = statesToIgnore)
        autocorrelation = statsmodels.tsa.stattools.acf(magnetisations, nlags = width, fft=True)
        return autocorrelation

    def getAutocorrelationLagTime(self, numberOfStates, distanceBetweenSamples = 1, statesToIgnore = 0):
        #The lag time is the time taken for the autocorrelation to fall by e
        autocorrelation = self.getAutocorrelation(numberOfStates, distanceBetweenSamples = distanceBetweenSamples, statesToIgnore = statesToIgnore)
        isAboveE     = np.greater(autocorrelation, 1/np.e)
        aboveE = itertools.takewhile(lambda x: x, isAboveE)        
        lagTime = sum(1 for _ in aboveE)
        return lagTime * distanceBetweenSamples / self.getNumberOfSpins()
    
    def getStandardDeviationEnergyFluctuaton(self, numberOfStates, statesToIgnore = 0):
        energies = self.getEnergies(lengthOfSample, statesToIgnore = statesToIgnore)
        plt.plot(energies)
        return np.std(energies)
     
    def getHeatCapacity(self, numberOfStates, statesToIgnore = 0):
        return (self.getStandardDeviationEnergyFluctuaton(self, numberOfStates, statesToIgnore) / self.getInitialTemperature() ) ** 2

    def getAverageMagnetisation(self, numberOfStates, distanceBetweenStates = 1, statesToIgnore = 0):
        magnetisations = self.getMagnetisations(numberOfStates, distanceBetweenStates, statesToIgnore)
        return np.average(magnetisations)
    
    def getAverageState(self, numberOfStates, distanceBetweenStates):
        states = self.getStates(numberOfStates, distanceBetweenStates)
        return np.average(states, axis = 0)
    
    def getAverageFourierTransform(self, numberOfStates, distanceBetweenStates):
        transforms = self.getFourierTransforms(numberOfStates, distanceBetweenStates)
        return np.average(np.absolute(transforms), axis = 0)
    
    @classmethod
    def generateSim(cls, stateType, bias = 0.5, **kwargs):
        initialState = stateType.generateRandomState(bias = bias, **kwargs)
        return cls(initialState)
    
    @classmethod
    def generateOscillatingSim(cls, stateType, **kwargs):
        initialState = stateType.generateRandomOscillatingState(**kwargs)
        return cls(initialState)
    
    @classmethod
    def generateOscillatingSimFromArray(cls, stateType, **kwargs):
        initialState = stateType.generateOscillatingStateFromArray(**kwargs)
        return cls(initialState)
    
class PlottableIsingSimulatuion(IsingSimulation):
    
    def animateStates(self, numberOfFrames, distanceBetweenFrames, fileName = 'im.mp4'):
        fig = plt.figure()
        images = []
        for x in self.getStates(numberOfFrames, distanceBetweenFrames):
            im = plt.imshow(x, aspect='equal', cmap=plt.cm.gray, interpolation='nearest', animated = True, norm=plt.Normalize(0, 1))
            images.append((im,))
      
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                
        im_ani = animation.ArtistAnimation(fig, images, interval=10, repeat_delay=0,
                                   blit=True)
        im_ani.save(fileName, writer=writer)
        plt.close('all')
        return fig, im_ani
    
    def animateFourierTransforms(self, numberOfFrames, distanceBetweenFrames, fileName = 'fourierTransformAnimation.mp4'):
        fig = plt.figure()
        images = []
        for x in self.getFourierTransforms(numberOfFrames, distanceBetweenFrames):
            im = plt.imshow(np.absolute(x), aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            images.append((im,))
            
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                
        im_ani = animation.ArtistAnimation(fig, images, interval=10, repeat_delay=0,
                                   blit=True)
        im_ani.save(fileName, writer=writer)
        plt.close('all')
        return fig, im_ani

   
if __name__ == '__main__':
    # Lowest temp 1

    start = time.time()
    sim = IsingSimulation.generateSim(stateType = VectorisedRandomNearestNeighbourMagnetisedIsingState2D, temperature = 1.75, dimensions = (200,200))
    #sim.getFields(200000)
    end = time.time()
    print(end-start)
    start = time.time()
    
    #TimeToEquilibriumAnalyser.plotMagnetisationsAboveTc()
    #TimeToEquilibriumAnalyser.collectMagnetisations()
    #TimeToEquilibriumAnalyser.plotMagnetisationsForTemperatures()
    #TimeToEquilibriumAnalyser.plotEquilibriumTimeAgainstTempuratureAboveTc()
    #TimeToEquilibriumAnalyser.plotEquilibriumTimeAgainstTempuratureAverage()
    #TimeToEquilibriumAnalyser.collectEquilibriumTimeAgainstTempurature()
    
    #AutocorrelationAnalyser.collectAutocorrelationLagTimeAgainstTempurature()
    #AutocorrelationAnalyser.plotAutocorrelationAgainstLatticeSize()
    #AutocorrelationAnalyser.collectAutocorrelationLagTimeAgainstTempurature3D()
    #AutocorrelationAnalyser.collectAutocorrelationLagTimeAgainstTempurature1D()
    #AutocorrelationAnalyser.estimateAutocorrelationLagTimeHighTempurature()
    #AutocorrelationAnalyser.plotAutocorrelationLagTime()
    
    #EquilibriumAnalyser.collectMagnetisationAgainstTemperature()
    #EquilibriumAnalyser.collectMagnetisationAgainstTemperature1D()
    #EquilibriumAnalyser.collectMagnetisationAgainstTemperature3D()
    #EquilibriumAnalyser.collectMagnetisationAgainstVaryingTemperature()
    #EquilibriumAnalyser.plotMagnetisationAgainstTemperatureAroundTc2()
    #EquilibriumAnalyser.plotMagnetisationAgainstTemperature3()
    #EquilibriumAnalyser.collectMagnetisationAgainstVaryingTemperatureWithField()
    #EquilibriumAnalyser.plotMagnetisationAgainstTemperature()
    
    #HeatCapacityAnalyser.plotDriftingEnergyAtTc()
    #HeatCapacityAnalyser.plotHeatCapacityAgainstTempurature4()
    #HeatCapacityAnalyser.collectHeatCapacityData()
    #HeatCapacityAnalyser.collectDiscreteHeatCapacityData()
    #HeatCapacityAnalyser.plotHeatCapacityData()
    
    #FiniteSizeScalingAnalyser.plotTcAgainstSize()
    
    #HysteresisAnalyser.collectLoopAreaAgainstTemperature()
    #HysteresisAnalyser.plotLoopAreaAgainstTemperature()
    #HysteresisAnalyser.collectStableMagnetisations()
    #HysteresisAnalyser.plotStableMagnetisations()
    #HysteresisAnalyser.plotMagnetisationAgainstFieldForDifferentTempuratures()
    
    #LocalOrderAnalyser.plotMagetisationRateAgainstMagnetisationAboveTc()
    #LocalOrderAnalyser.plotMagetisationRateAgainstMagnetisationBelowTc()
    #LocalOrderAnalyser.plotOrderedState()
    
    #GlobalOrderAnalyser.orderInSystematicPlot()
    #GlobalOrderAnalyser.animateOrderedFourierTransforms()
    #GlobalOrderAnalyser.collectOrderedTransformData()
    #GlobalOrderAnalyser.plotOrderedTransformData()
    #GlobalOrderAnalyser.orderInFastSystematicPlot()
    #GlobalOrderAnalyser.disorderInRandomPlot()
    #GlobalOrderAnalyser.animateOrderWithTemperature()
    #GlobalOrderAnalyser.plotSpinUpAgainstTemp2()
    
    #SusceptibilityAnalyser.collectSusceptibility()
    SusceptibilityAnalyser.plotSusceptibility()
    end = time.time()
    print(end-start)