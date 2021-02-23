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

    def _applyToStates(self, function, numberOfStates, distanceBetweenStates):
        # Equivalent to but faster than [function(self.getStateNumber(N)) for N in indexOfStates]
        vals = []
        currentState = self._getInitialState()
        for x in range(numberOfStates):
            vals.append(function(currentState))
            currentState = currentState.getStateNumber(distanceBetweenStates)
        return np.array(vals)

    def getMagnetisations(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getMagnetisation(), numberOfStates, distanceBetweenStates)

    def getStates(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getStateAsArray()  , numberOfStates, distanceBetweenStates)

    def getFourierTransforms(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getStateAsDFT()    , numberOfStates, distanceBetweenStates)
    
    def getEnergies(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getTotalEnergy()   , numberOfStates, distanceBetweenStates)
    
    def getAverageFields(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getAverageField()  , numberOfStates, distanceBetweenStates)
    
    def getTemperatures(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getTemperature()   , numberOfStates, distanceBetweenStates)

    def getImages(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getStateAsImage()  , numberOfStates, distanceBetweenStates)

    def getSpinUp(self, numberOfStates, distanceBetweenStates = 1):
        return self._applyToStates(lambda x: x.getNumberOfSpinUp(), numberOfStates, distanceBetweenStates)
  
    def getTimeAveragedMagnetisations(self, lengthOfSample, averageOver = 50):
        magnetisations = self.getMagnetisations(lengthOfSample + averageOver)
        accumilative = np.cumsum(magnetisations) 
        timeAverage = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
        return timeAverage
    
    def getTimeAveragedEnergies(self, lengthOfSample, averageOver = 50):
        magnetisations = self.getEnergies(lengthOfSample + averageOver)
        accumilative = np.cumsum(magnetisations) 
        timeAverage = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
        return timeAverage
    
    def getPeriodAveragedMagnetisations(self, period, numberOfPeriods = 10):
        lengthOfSample = numberOfPeriods * period
        magnetisations = self.getMagnetisations(lengthOfSample)
        splitMagnetisations = np.split(magnetisations, numberOfPeriods)
        periodAverage = np.average(np.stack(splitMagnetisations, axis = -1), axis = -1)
        return periodAverage
    
    def getAreaOfHysteresis(self, period, numberOfPeriods = 10):
        magnetisations = self.getPeriodAveragedMagnetisations(period, numberOfPeriods)
        fields = self.getAverageFields(period)
        def area(x,y):
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        areaOfHysteresis = area(fields, magnetisations)
        return areaOfHysteresis
    
    def getEquilibriumTime(self, lengthOfSample, distanceBetweenSamples = 1, tolerance = 0, equilibriumMagnetisation = None):
        magnetisations = self.getMagnetisations(lengthOfSample)
        equilibriumMagnetisation = self.getAverageMagnetisation(lengthOfSample, distanceBetweenStates = distanceBetweenSamples) if equilibriumMagnetisation == None else equilibriumMagnetisation
        
        hasConverged = np.greater(tolerance, np.abs(magnetisations - equilibriumMagnetisation))

        #plt.plot(np.abs(np.gradient(np.polyval(fit, range(lengthOfSample)))))
        yetToConverge = itertools.takewhile(lambda x: not x, hasConverged)        
        equilibriumTime = sum(1 for _ in yetToConverge)
        return (equilibriumTime * distanceBetweenSamples) / self.getNumberOfSpins()
                       
    def getAutocorrelation(self, lengthOfSample = 10000, width = None):
        width = int(lengthOfSample) if width == None else width
        magnetisations = self.getMagnetisations(lengthOfSample)
        autocorrelation = statsmodels.tsa.stattools.acf(magnetisations, nlags = width, fft=True)
        return autocorrelation

    def getAutocorrelationLagTime(self, lengthOfSample = 10000):
        #The lag time is the time taken for the autocorrelation to fall by e
        autocorrelation = self.getAutocorrelation(lengthOfSample)
        isAboveE     = np.greater(autocorrelation, 1/np.e)
        aboveE = itertools.takewhile(lambda x: x, isAboveE)        
        lagTime = sum(1 for _ in aboveE)
        return lagTime / self.getNumberOfSpins()
    
    def getStandardDeviationEnergyFluctuaton(self, lengthOfSample = 20000, statesToIgnore = 10000):
        energies = self.getEnergies(lengthOfSample)[statesToIgnore:]
        plt.plot(energies)
        return np.std(energies)
    
    def getPeriodSplitEnergyFluctuations(self, period, numberOfPeriods):
        energies = self.getEnergies(period * numberOfPeriods)
        energiesSplit = np.split(energies, numberOfPeriods)
        return np.std(energiesSplit, axis = -1)
     
    def getHeatCapacity(self, lengthOfSample = 20000, statesToIgnore = 10000):
        return (self.getStandardDeviationEnergyFluctuaton(lengthOfSample = 20000, statesToIgnore = 10000) / self.getInitialTemperature() ) ** 2

    def getPeriodSplitHeatCapacity(self, period, numberOfPeriods):
        temperatures = [self.getStateNumber(N).getTemperature() for N in range(0, period * numberOfPeriods, period)] #if temperatures == None else temperatures
        return np.divide(self.getPeriodSplitEnergyFluctuations(period, numberOfPeriods), temperatures) ** 2

    def getAverageMagnetisation(self, numberOfStates, distanceBetweenStates = 1):
        magnetisations = self.getMagnetisations(numberOfStates, distanceBetweenStates)
        return np.average(magnetisations, weights = range(magnetisations.size), axis = 0)
    
    def getAverageState(self, numberOfStates, distanceBetweenStates):
        
        states = self.getStates(numberOfStates, distanceBetweenStates)
        return np.average(states, axis = 0)
    
    def getAverageFourierTransform(self, numberOfStates, distanceBetweenStates):
        
        average = np.average(np.absolute(self.getFourierTransforms(numberOfStates, distanceBetweenStates)), axis = 0)
        return average
    
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
        for x in self.getImages(numberOfFrames, distanceBetweenFrames):
            images.append((x,))
      
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
    
class SystematicIsingSim(IsingSimulation):
    
    def getAllMagnetisations(self, numberOfFullStates):
        currentState = self._getInitialState()
        magnetisations = []
        for x in range(numberOfFullStates):
            for N in range(self.getNumberOfSpins()):
                state = currentState.getIntermediateStatesArray(N)
                mag = (2 * np.count_nonzero(state) - self.getNumberOfSpins()) / self.getNumberOfSpins()
                magnetisations.append(mag)
            currentState = currentState.getStateNumber(self.getNumberOfSpins())
        return magnetisations
            
    def plotAllMagnetisations(self, numberOfFullStates, ax, resolution = None, label = ""):
        xSpacing = np.arange(0,numberOfFullStates * self.getNumberOfSpins(),1)

        magnetisations = self.getAllMagnetisations(numberOfFullStates)
        times = xSpacing / self.getNumberOfSpins()
        ax.plot(times, magnetisations, label = label)
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnetisation")
        return ax
    
class TemperatureVaryingIsingSim(PlottableIsingSimulatuion):
    
    def getAverageMagnetisationsAgainstTemperature(self, lengthOfSample):
        temperatures = self.getTemperatures(lengthOfSample)
        magnetisations = self.getMagnetisations(lengthOfSample)
        l1 = pd.DataFrame([temperatures,magnetisations]).T.groupby(0,as_index=False)[1].mean().values.tolist()
        return np.swapaxes(np.array(l1),0,-1)
        
    def getAverageMagnetisationsSplit(self, lengthOfSample, numberOfSplits):
        lengthOfSplit = int(lengthOfSample/numberOfSplits)
        magnetisations = self.getMagnetisations(lengthOfSample)
        splitMagnetisations = np.split(magnetisations, numberOfSplits)
        return np.average(splitMagnetisations, axis = -1, weights = range(lengthOfSplit))
        
    def calculateTc(self, lengthOfSample):
        currentState = self.firstIsingState
        for x in range(lengthOfSample):
            if(currentState.getMagnetisation() < 0.5):
                #T=Tc
                return currentState.getTemperature()
            currentState = currentState.getStateNumber(1)
        return currentState.getTemperature()
    
    def plotAveragedMagnetisations(self, lengthOfSample, ax, withTc = False):
        temperatures, averageMagnetisations  = self.getAverageMagnetisationsAgainstTemperature(lengthOfSample)
        if(withTc):
            ax.plot(*self._calculateTc(temperatures, averageMagnetisations))
        ax.plot(temperatures, averageMagnetisations)
        
        
        return ax
    
if __name__ == '__main__':
    # Lowest temp 1

    start = time.time()
    sim = IsingSimulation.generateSim(stateType = VectorisedRandomNearestNeighbourIsingState2D, temperature = 1.75, dimensions = (2000,2000))
    #sim._getInitialState().getAllInitialPositiveNeighbours()
    #oldData = np.load("5-FiniteSizeScaling\TcAgainstSizeData.npy")
    #newData = np.load("5-FiniteSizeScaling\TcAgainstSizeData5.npy")
    #a = newData[1,]
    #newData = newData[newData[:,0]< 10000, :]
    #newData = np.append([a], newData[3:-1], axis = 0)
    
    #print(newData)
    #print(oldData)
    #oldData = np.array([[0,0]])
    #np.save("5-FiniteSizeScaling\TcAgainstSizeData5.npy", newData)
    #sim.getStateNumber(400000)
    # sim.getStateNumber(20000)
    end = time.time()
    print(end-start)
    start = time.time()
    #LocalOrderAnalyser.animateOrderedState()
    #FiniteSizeScalingAnalyser.plotTcAgainstSize()
    #FiniteSizeScalingAnalyser.plotDiscritisedMagAgainstT()
    #sim.plotFirstNMagnetisations(30000)
    #TimeToEquilibriumAnalyser.plotEquilibriumTimeAgainstTempuratureAboveTc2()
    
    #EquilibriumAnalyser.plotFirstNAveragedMagnetisations(30000, [sim])
    
    #TimeToEquilibriumAnalyser.plotMagnetisationsAboveTc()
    #TimeToEquilibriumAnalyser.plotFirstNMagnetisationsBelowTc()
    #TimeToEquilibriumAnalyser.plotFirstNMagnetisationsBelowTc2()
    #TimeToEquilibriumAnalyser.plotEquilibriumTimeAgainstTempuratureAboveTc()
    #TimeToEquilibriumAnalyser.plotEquilibriumTimeAgainstTempuratureAverage()
    
    #AutocorrelationAnalyser.plotAutocorrelationLagTimeAgainstLatticeSize()
    #AutocorrelationAnalyser.plotAutocorrelationLagTimeAgainstTempurature()
    #AutocorrelationAnalyser.plotAutocorrelationAgainstLatticeSize()
    #AutocorrelationAnalyser.collectAutocorrelationLagTimeAgainstTempurature3D()
    #AutocorrelationAnalyser.collectAutocorrelationLagTimeAgainstTempurature1D()
    
    #EquilibriumAnalyser.plotMagnetisationAgainstTemperature()
    #EquilibriumAnalyser.collectMagnetisationAgainstTemperatureAroundTc()
    #EquilibriumAnalyser.collectMagnetisationAgainstVaryingTemperature()
    #EquilibriumAnalyser.plotMagnetisationAgainstTemperatureAroundTc2()
    #EquilibriumAnalyser.plotMagnetisationAgainstTemperature3()
    #EquilibriumAnalyser.plotMagnetisations()
    
    #HeatCapacityAnalyser.plotDriftingEnergyAtTc()
    #HeatCapacityAnalyser.plotHeatCapacityAgainstTempurature4()
    HeatCapacityAnalyser.collectHeatCapacityData()
    #HeatCapacityAnalyser.collectDiscreteHeatCapacityData()
    #HeatCapacityAnalyser.plotHeatCapacityData()
    
    FiniteSizeScalingAnalyser.plotTcAgainstSize()
    
    #HysteresisAnalyser.plotLoopAreaAgainstTemperature()
    
    #GlobalOrderAnalyser.orderInSystematicPlot()
    #GlobalOrderAnalyser.animateOrderedFourierTransforms()
    #GlobalOrderAnalyser.plotOrderedTransform()
    #GlobalOrderAnalyser.plotOrderedTransform2()
    #GlobalOrderAnalyser.orderInFastSystematicPlot()
    #GlobalOrderAnalyser.disorderInRandomPlot()
    #GlobalOrderAnalyser.animateOrderWithTemperature()
    #GlobalOrderAnalyser.plotSpinUpAgainstTemp2()
    
    print("pass")
    # IsingSimulationPlotter2D.animateFirstNFourierTransforms(100, sim)
    print("pass")
    # IsingSimulationPlotter2D.plotAverageFirstNStates(100, [sim])
    # IsingSimulationPlotter2D.animateFirstNStates(50)
    # IsingSimulationPlotter2D.plotFirstNMagnetisations(2000, [sim])
    
    # sim.getStateNumber(60000)
    # IsingSimulationPlotter2D.plotMeanMagnetisationAgainsyTemperatureForDifferentMethods()
   
    end = time.time()
    print(end-start)