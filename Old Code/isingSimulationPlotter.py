# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:12:25 2020

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt
from isingSimulation import IsingSimulation, PlottableIsingSimulatuion
from isingState import VectorisedRandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D, VectorisedRandomNearestNeighbourIsingState2D, FastSystematicNearestNeighbourForgetfulIsingState2D, SystematicNearestNeighbourTemperatureVaryingIsingState2D, SystematicNearestNeighbourForgetfulIsingState2D, RandomNearestNeighbourTemperatureVaryingIsingState2D, RandomNearestNeighbourForgetfulIsingState3D, RandomNearestNeighbourForgetfulIsingState1D, RandomNearestNeighbourForgetfulIsingState2D, RandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D, SystematicIsingState, RandomNearestNeighbourMagnetisedIsingState2D, RandomNearestNeighbourIsingState2D, InterpolatedIsingState, SystematicNearestNeighbourIsingState2D, IsingState, IsingState2D, NearestNeighbourIsingState
import pandas as pd
import itertools
import scipy
   
class TimeToEquilibriumAnalyser():
    
    def plotEquilibriumTimeAgainstTempuratureAverage():
        
        fig, ax = plt.subplots()
        '''
        data = np.load("1-TimeToEquilibrium\EquilibriumTimeTemperatureData.npy")
        averages1 = pd.DataFrame(data[1:]).groupby(0,as_index=False).mean().values
        standardErrors1 = pd.DataFrame(data[1:]).groupby(0,as_index=False).sem().values[:,1]
        print(averages1)
        temperatures1 = averages1[:-6,0]
        times1 = averages1[:-6,1]
        ax.errorbar(temperatures1, times1, yerr = standardErrors1[:-6],marker = "+")
        '''
        data = np.load("1-TimeToEquilibrium\EquilibriumTimeTemperatureDataFinal.npy")
        data = data[data[:,0] > 1,:]
        
        averages2 = pd.DataFrame(data).groupby(0,as_index=False).mean().values
        standardErrors2 = pd.DataFrame(data).groupby(0,as_index=False).sem().values[:,1]
        temperatures2 = averages2[:,0] * 2 # Before error in energy was found
        times2 = averages2[:,1]
        
        data2 = np.load("1-TimeToEquilibrium\EquilibriumTimeTemperatureData5.npy")
        average = np.average(data2[1:, 1])
        standardError = np.std(data2[1:, 1]) / np.sqrt(data2[1:, 1].size - 1)
        print("{:.4} +- {:.4}".format(average, standardError))
        print(data[:,1])
        ax.errorbar(temperatures2, times2, yerr = standardErrors2,marker = "+")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel("Equilibrium Time")
        ax.set_title("Plot of Equilibrium Time Against Temperature")
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        #data2 = np.load("EquilibriumTimeTemperatureData2.npy")
        #print(data2[:,0])
        #ax.plot(data2[:,0], data2[:,1], "+")
        fig.savefig("1-TimeToEquilibrium\EquilibriumTimeAgainstTemperaturePlot5.png")
        
    def plotEquilibriumTimeAgainstTempuratureAboveTc2():
        dims = (150,150)
        temperatures = np.linspace(2,10, num = 10)
        N = 340000
        equilibriumTimes = []
        stateType = RandomNearestNeighbourIsingState2D
        
        fig, ax = plt.subplots()
        for T in temperatures:
            sim = IsingSimulation.generateSim(dimensions = dims, temperature = T, stateType = stateType)
            equilibriumTimes.append(sim.getEquilibriumTime(N, tolerance = .001, equilibriumMagnetisation = 0))
        ax.plot(temperatures, np.array(equilibriumTimes) / (dims[0] * dims[1]))
        
        ax.set_xlabel(r"Temperature  $T / T_0$")
        ax.set_ylabel("Equilibrium Time")
        ax.set_title("Plot of Equilibrium Time Against Temperature")
        fig.savefig("1-TimeToEquilibrium\EquilibriumTimeAgainstTemperaturePlot.png")
        
    def collectEquilibriumTimeAgainstTempuratureAboveTc():
        dims = (40,40)
        #temperatures = np.arange(1.5, 2.5, .1)
        #temperatures = np.arange(1, 50, .2)
        #temperatures = np.arange(10, 50, .4)
        temperatures = [10000000]
        #N = 30000
        N = 16000
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        
        fig, ax = plt.subplots()
        data = np.array([[0,0]])
        for x in range(200):
            for T in temperatures:
                #data = np.load("1-TimeToEquilibrium\EquilibriumTimeTemperatureData5.npy")
                sim = IsingSimulation.generateSim(dimensions = dims, temperature = T, stateType = stateType)
                a = [T, sim.getEquilibriumTime(N, tolerance = .001, equilibriumMagnetisation = 0)]
                data = np.append(data, [a], axis = 0)
                print("done {:}".format(x))
                np.save("1-TimeToEquilibrium\EquilibriumTimeTemperatureData5.npy", data)
            
        temperatures = data[:,0]
        times = data[:,1]
        ax.plot(temperatures, times, "+")
        ax.set_xlabel(r"Temperature  $T / T_0$")
        ax.set_ylabel("Equilibrium Time")
        ax.set_title("Plot of Equilibrium Time Against Temperature")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig("1-TimeToEquilibrium\EquilibriumTimeAgainstTemperaturePlot4.png")
        
    def plotMagnetisationsAboveTc():
        dims = (200,200)
        numberOfFullStates = 2
        temperatures = [1,2,5,10,20]
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        
        fig, ax = plt.subplots()
        for T in temperatures:
            sim = PlottableIsingSimulatuion.generateSim(dimensions = dims, temperature = T, stateType = stateType)
            numberOfStates = numberOfFullStates * dims[0] * dims[1]
            xSpacing = np.arange(0,numberOfStates,1)

            magnetisations = sim.getMagnetisations(numberOfStates = numberOfStates, distanceBetweenStates = 1)
            times = xSpacing / sim.getNumberOfSpins()
            ax.plot(times, magnetisations, label = sim.getInitialTemperature())

        ax.legend(loc = 'upper right')
        ax.set_xlabel(r"Time  $T / T_0$")
        ax.set_ylabel(r"Magnetisation M / M_0")
        ax.set_title("Plot of magnetisation against time for a range of temperatures")

        fig.savefig('1-TimeToEquilibrium\MagnetisationAgainstTimeAboveTcPlot.png')
        
    def plotFirstNMagnetisationsBelowTc():
        N = 640000
        dim = (200,200)
        temperatures = [0.01, 0.1, .5, 1, 1.5]
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        
        fig, ax = plt.subplots()
        for T in temperatures:
            sim = PlottableIsingSimulatuion.generateSim(stateType = stateType, dimensions = dim, temperature = T, bias = 0.25)
            xSpacing = np.arange(0,N,1)

            magnetisations = sim.getMagnetisations(numberOfStates = N, distanceBetweenStates = 1)
            times = xSpacing / sim.getNumberOfSpins()
            ax.plot(times, magnetisations, label = sim.getInitialTemperature())

        ax.legend(loc = 'upper right')
        ax.set_title("Plot of magnetisation against time for a range of temperatures")

        fig.savefig('1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcPlot.pdf')
        
    def plotFirstNMagnetisationsBelowTc2():
        N = 160000
        dim = (100,100)
        temperature = .01
        bias = np.linspace(.3,.1,4)
        stateType = RandomNearestNeighbourIsingState2D
        
        fig, ax = plt.subplots()
        for b in bias:
            sim = PlottableIsingSimulatuion.generateSim(stateType = stateType, dimensions = dim, temperature = temperature, bias = b)
            xSpacing = np.arange(0,N,1)

            magnetisations = sim.getMagnetisations(numberOfStates = N, distanceBetweenStates = 1)
            times = xSpacing / sim.getNumberOfSpins()
            ax.plot(times, magnetisations, label = sim.getInitialTemperature())

        ax.legend(loc = 'upper right')
        ax.set_title("Plot of magnetisation against time for a range of initial bias")

        fig.savefig('1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcPlot2.pdf')
        
class AutocorrelationAnalyser():
            
    def plotAutocorrelationLagTimeAgainstTempurature():
        a = np.load("2-Autocorrelation\lagTimeAgainstTempurature3.npy")
        
        temperatures = a[:,0] * 2 #Before energy scale correction
        lagTimes = a[:,1]

        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(temperatures, lagTimes, "+")
        ax2.set_title("Plot of Equilibrium Lag Time against Temperature")
        ax2.set_xlabel("Temperature of Lattice")
        ax2.set_ylabel("Lag Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot.png")
        
        data = np.load("2-Autocorrelation\lagTimeAgainstTempurature3D.npy")
        temperatures = data[2:,0] * 2
        lagTimes = data[2:,1]
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(temperatures, lagTimes, "+")
        ax2.set_title("Plot of Equilibrium Lag Time against Temperature for a 3D lattice")
        ax2.set_xlabel("Temperature of Lattice")
        ax2.set_ylabel("Lag Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot3D.png")
        
        data = np.load("2-Autocorrelation\lagTimeAgainstTempurature1D.npy")
        temperatures = data[1:,0] * 2
        lagTimes = data[1:,1]
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(temperatures, lagTimes, "+")
        ax2.set_title("Plot of Equilibrium Lag Time against Temperature for a 1D lattice")
        ax2.set_xlabel(r"Temperature $T / T_0$")
        ax2.set_ylabel("Lag Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot1D.png")
        
    def collectEquilibriumTimeAgainstTempurature():
        temperatures = [1,2]
        dim = (60,60)
        stateType = RandomNearestNeighbourIsingState2D
        numberOfStates = 100
        
        for T in temperatures:
            #data = np.load()
            #temps = list(data[0])
            #times = list(data[1])
            temps = []
            times = []
            sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, dimensions = d, bias = 0.5)
            times.append(sim.getEquilibriumTime(lenghtOfSample = numberOfStates, distanceBetweenSamples = sim.getNumberOfSpins(), tolerance = 0))
            temps.append(T)
            np.save("", [temps, times])
            
    def collectMagnetisationsForTemperatures():
        temperatures = [1,2]
        dim = (60,60)
        stateType = RandomNearestNeighbourIsingState2D
        numberOfStates = 100
        distanceBetweenStates = 1

        for T in temperatures:
            #data = np.load()
            #temps = list(data[0])
            #times = list(data[1])
            temps = []
            times = []
            sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, dimensions = d, bias = 0.5)
            
            time = np.arange(0, numberOfStates * distanceBetweenStates, distanceBetweenStates) / sim.getNumberOfSpins()
            times.append(time)
            magnetisations.append(sim.getMagnetisations(numberOfStates = numberOfStates, distanceBetweenStates = distanceBetweenStates))
            temps.append(T)
            np.save("", [temps, times, magnetisations])
    
    def plotMagnetisationsForTemperatures():
        data = np.load()
        temperatures = data[0]
        times = data[1]
        magnetisations = data[2]
        fig, ax = plt.subplots()
        for x in range(len(temperatures)):
            ax.plot(times[i], magnetisations[i], label = temperatures[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        ax.set_title("Plot of Magnetisatiion for a range of Temperatures")
        
class AutocorrelationAnalyser():
    
    def collectAutocorrelationLagTimeAgainstTempurature():
        dim = (300,300)
        temperatures = np.linspace(2,20,18)
        sampleTime = 20
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        
        for T in temperatures:
            data = np.load("2-Autocorrelation\lagTimeAgainstTempurature3.npy")
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dim, temperature = T)
            lengthOfSample = sampleTime * dim[0] * dim[1]
            data = np.append(data, [[sim.getInitialTemperature(), sim.getAutocorrelationLagTime(lengthOfSample)]], axis = 0)
            print("Done : {}".format(T))
            np.save("2-Autocorrelation\lagTimeAgainstTempurature3.npy", data)
        
  
    def collectAutocorrelationLagTimeAgainstTempurature1D():
        dim = (90000)
        temperatures = np.linspace(2.5,19.5,18)
        sampleTime = 20
        stateType = RandomNearestNeighbourForgetfulIsingState1D

        data = np.array([[0,0]])
        for T in temperatures:
            data = np.load("2-Autocorrelation\lagTimeAgainstTempurature1D.npy")
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dim, temperature = T)
            lengthOfSample = sampleTime * dim
            data = np.append(data, [[sim.getInitialTemperature(), sim.getAutocorrelationLagTime(lengthOfSample)]], axis = 0)
            print("Done : {}".format(T))
            np.save("2-Autocorrelation\lagTimeAgainstTempurature1D.npy", data)
        
    def collectAutocorrelationLagTimeAgainstTempurature3D():
        dim = (30,30,30)
        temperatures = np.linspace(2.5,19.5,18)
        sampleTime = 20
        stateType = RandomNearestNeighbourForgetfulIsingState3D

        data = np.array([[0,0]])
        for T in temperatures:
            data = np.load("2-Autocorrelation\lagTimeAgainstTempurature3D.npy")
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dim, temperature = T)
            lengthOfSample = sampleTime * dim[0] * dim[1] * dim[2]
            data = np.append(data, [[sim.getInitialTemperature(), sim.getAutocorrelationLagTime(lengthOfSample)]], axis = 0)
            print("Done : {}".format(T))
            np.save("2-Autocorrelation\lagTimeAgainstTempurature3D.npy", data)
  
    def plotAutocorrelationLagTimeAgainstTempuratureSystematic():
        dim = (300,300)
        temperatures = np.linspace(2,20,18)
        sampleTime = 20
        stateType = SystematicNearestNeighbourForgetfulIsingState2D

        data = np.array([[0,0]])
        for T in temperatures:
            data = np.load("2-Autocorrelation\lagTimeAgainstTempuratureSystematic.npy")
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dim, temperature = T)
            lengthOfSample = sampleTime * dim[0] * dim[1]
            data = np.append(data, [[sim.getInitialTemperature(), sim.getAutocorrelationLagTime(lengthOfSample)]], axis = 0)
            print("Done : {}".format(T))
            np.save("2-Autocorrelation\lagTimeAgainstTempuratureSystematic.npy", data)
        '''
        data = np.load("2-Autocorrelation\lagTimeAgainstTempuratureSystematic.npy")
        temperatures = data[:,0]
        lagTimes = data[:,1]
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(temperatures, lagTimes, "+")
        ax2.set_title("Plot of Equilibrium Lag Time against Temperature")
        ax2.set_xlabel("Temperature of Lattice")
        ax2.set_ylabel("Lag Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot.png")
        '''
    
    def plotAutocorrelationAgainstLatticeSize():
        tempurature = 1.8
        dimensions = np.linspace(50,150,6, dtype = int)
        sampleTime = 10
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        
        fig, ax = plt.subplots()
        for D in dimensions:
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = (D,D), temperature = tempurature)
            lengthOfSample = sampleTime * D * D
            autocorrelation = sim.getAutocorrelation(lengthOfSample)
            times = np.arange(lengthOfSample) / sim.getNumberOfSpins()
            ax.plot(times, autocorrelation, label = "({},{})".format(D,D))
            print("Done : {}".format(D))
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Plot of Autocorrelation for several lattuice sizes")
        fig.savefig("2-Autocorrelation\AutocorrelationForSizePlot.png")
        
    def plotAutocorrelationLagTimeAgainstLatticeSize():
        tempurature = 1000000
        dimensions = np.linspace(500,700,3, dtype = int)
        sampleTime = 10
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        data = np.array([[0,0]])
        for D in dimensions:
            #data = np.load("2-Autocorrelation\lagTimeAgainstSize{:}.npy".format(temperature))
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = (D,D), temperature = tempurature)
            lengthOfSample = sampleTime * D * D
            data = np.append(data, [[sim.getNumberOfSpins(), sim.getAutocorrelationLagTime(lengthOfSample)]], axis = 0)
            print("Done : {}".format(D))
            np.save("2-Autocorrelation\lagTimeAgainstSize{:}.npy".format(tempurature), data)

        #data = np.append(data, np.load("2-Autocorrelation\lagTimeAgainstSize.npy"))
        areas = data[data[:,0]> 150,0]
        lagTimes = data[data[:,0]> 150,1]
        print(data[data[:,1]> 1.5, :])
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(areas, lagTimes, "+")
        ax2.set_title("Plot of Equilibrium Lag Time against Size")
        ax2.set_xlabel("Area of Lattice")
        ax2.set_ylabel("Lag Time")
        fig2.savefig("2-Autocorrelation\AutocorrelationTimeAgainstSizePlot{:}.png".format(tempurature))
    
class EquilibriumAnalyser():
        
        
    def collectMagnetisationAgainstTemperatureAroundTc():
        timeAtEachTemperature = 5 #50
        startTemp = 2.36 #2.3
        endTemp = 2.46 #2.4
        resolution = 200
        timeOfSimulation = timeAtEachTemperature * resolution
        dims =  [(25,25),(25,25),(25,25), (25,25), (25,25), (25,25), (25,25), (25,25), (50,50),(50,50),(50,50),(50,50),(100,100)]
        #dims = [(25,25),(50,50), (100,100)]
        stateType = SystematicNearestNeighbourTemperatureVaryingIsingState2D
        
        for d in dims:
            temperatures, magnetisations = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureContiuousData2-{}.npy".format(d[0]))
            temperatures = list(temperatures)
            magnetisations = list(magnetisations)
            #temperatures = []
            #magnetisations = []
            sim = IsingSimulation.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, bias = 0.5, numberOfSteps = timeOfSimulation, resolution = resolution, dimensions = d)
            magnetisations += list(sim.getMagnetisations(timeOfSimulation, distanceBetweenStates = sim.getNumberOfSpins()))
            temperatures   += list(sim.getTemperatures(timeOfSimulation,  distanceBetweenStates = sim.getNumberOfSpins()))
            np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureContiuousData2-{}.npy".format(d[0]), [temperatures, magnetisations])
            print("done {}".format(d))
            
    def plotMagnetisationAgainstTemperatureAroundTc2():
        fig, ax = plt.subplots()
        dims = [(25,25),(50,50), (100,100)]
        
        for d in dims:
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureContiuousData2-{}.npy".format(d[0]))
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]
            ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = d)
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot.png")

        
    def plotMagnetisationAgainstTemperatureAroundTc():
        dims = [(100,100), (200,200)]
        fig, ax = plt.subplots()

        data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureContiuousData200.npy")
        averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
        standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
        temperatures = averages[:,0]
        magnetisations = averages[:,1]
        (m,c), V = np.polyfit(temperatures[temperatures < 2.37], magnetisations[temperatures < 2.37], deg = 1, cov = True)
        mError = np.sqrt(V[0][0])
        cError = np.sqrt(V[1][1])
        print(data.shape)
        ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = 200)
        ax.plot(temperatures, m * temperatures + c)
        
        data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureContiuousData150.npy")
        #data = data[:,20000:]
        averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
        standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
        temperatures = averages[:,0]
        magnetisations = averages[:,1]
        (m,c), V = np.polyfit(temperatures[temperatures < 2.37], magnetisations[temperatures < 2.37], deg = 1, cov = True)
        mError = np.sqrt(V[0][0])
        cError = np.sqrt(V[1][1])
        print(data.shape)
        ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = 150)
        ax.plot(temperatures, m * temperatures + c, label = 150)
        
        data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureContiuousData100.npy")
        print(data)
        #data = data[:,20000:30000][:,np.logical_and(np.logical_or(data[0,20000:30000] < 2.34, data[0,20000:30000] > 2.353), data[0,20000:30000] < 2.361)]
        #data = data[:,10000:20000][:,np.logical_and(np.logical_or(data[0,20000:30000] < 2.342, data[0,20000:30000] > 2.35), data[0,20000:30000] < 2.39)]
        #data = data[:,0:10000][:,np.logical_and(np.logical_or(data[0,20000:30000] < 2.318, data[0,20000:30000] > 2.33), data[0,20000:30000] < 2.365)]
        #data = data[:,30000:40000][:,np.logical_and(np.logical_or(data[0,20000:30000] < 2.34, data[0,20000:30000] > 2.358), data[0,20000:30000] < 2.365)]
        #data = data[:,40000:50000]
        #data = np.concatenate((data[:,:10000], data[:,10000:20000], data[:,20000:30000]), axis = 0)
        averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
        standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
        temperatures = averages[:,0]
        magnetisations = averages[:,1]
        (m,c), V = np.polyfit(temperatures, magnetisations, deg = 1, cov = True)
        mError = np.sqrt(V[0][0])
        cError = np.sqrt(V[1][1])
        
        ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = 100)
        ax.plot(temperatures, m * temperatures + c)
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot.png")
        
    def collectMagnetisationAgainstVaryingTemperature():
        timeAtEachTemperature = 100
        startTemp = 0.1
        endTemp = 5
        resolution = 100
        dims = [(50,50), (100,100), (150,150), (200,200)]
        stateType = RandomNearestNeighbourTemperatureVaryingIsingState2D
        
        fig, ax = plt.subplots()
        for d in dims:
            sim = IsingSimulation.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, bias = 0.5, numberOfSteps = timeAtEachTemperature * resolution, resolution = resolution, dimensions = d)
            magnetisations = []
            temps = []
            magnetisations += list(sim.getAverageMagnetisation(resolution * timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins()))
            temps += list(sim.getTemperatures(resolution * timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins()))
            ax.plot(temps, magnetisations, label = d)
            np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2-{}.npy".format(d[0]), [temps, magnetisations])
            
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot2.png")
        
    def collectMagnetisationAgainstTemperature():
        timeAtEachTemperature = 20
        temperatures = np.linspace(0.1,2.4,num = 30)
        d = (50,50)
        stateType = SystematicNearestNeighbourIsingState2D
        
        
        #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData.npy")
        #magnetisations = list(data[1])
        #temps = list(data[0])
        magnetisations = []
        temps = []
        for t in temperatures:
            sim = IsingSimulation.generateSim(stateType = stateType, bias = 0.5, dimensions = d, temperature = t)
            magnetisations.append(sim.getAverageMagnetisation(timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins()))
            temps.append(t)
        np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureData1.npy", [temps, magnetisations])
        

        
    def plotMagnetisationAgainstTemperature():
        fig, ax = plt.subplots()
        data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData.npy")
        ax.plot(data[0], data[1], "+", label = "100")
        
        #data1 = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData1.npy")
        #ax.plot(data1[0], data1[1], "+", label = "200")
        
        data2 = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2.npy")
        ax.plot(data2[0], data2[1], "+", label = "50")
        
        temperatures = np.linspace(1, 2.3, num = 1000)
        theoreticalMagnetisation = (1 - (np.sinh(2/temperatures)) ** -4) ** (1/8)
        ax.plot(temperatures, theoreticalMagnetisation)
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot2.png")

    def plotMagnetisations():
        dims = (100,100)
        numberOfFullStates = 800
        temperatures = np.linspace(2.4,2.41, num = 1)
        stateType = SystematicNearestNeighbourIsingState2D
        
        fig, ax = plt.subplots()
        for T in temperatures:
            sim = IsingSimulation.generateSim(stateType = stateType, bias = 0.5, dimensions = dims, temperature = T)
            magnetisations = sim.getMagnetisations(numberOfStates = numberOfFullStates, distanceBetweenStates = sim.getNumberOfSpins())
            ax.plot(magnetisations, label = T)
        ax.legend(loc = 'upper right')
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnetisation")
        ax.set_title("Plot of magnetisation against time for T = 2.4")

        fig.savefig('3-MeanMagnetisation\MagnetisationAgainstTimePlot.png')
'''   
    def plotFirstNAveragedMagnetisations(N, simulations, averageOver = 20, sameAxis = False, simulationLables = None):
        numberOfSimulations = len(simulations)
        sameAxis |= (numberOfSimulations == 1)
        
        fig, axs = plt.subplots(1 if sameAxis else numberOfSimulations, 1, sharex=True)
        for i in range(numberOfSimulations):
            simulations[i].plotFirstNAveragedMagnetisations(N, ax = (axs if sameAxis else axs[i]), averageOver = averageOver, label = ("" if simulationLables == None else simulationLables[i]))
        
        fig.savefig('AverageMagnetisationPlot.pdf')
        return fig, axs
  
    def plotMeanMagnetisationAgainstTemperatureForRandomAndSystematic():
        N = 50000
        temperatures = np.linspace(.5, 2.5, num = 30)
        xDim, yDim = (40,40)
        
        simulations = [NearestNeighbourIsingSimulation2D.generateOrderedRandomNearestNeighbourSim(T, xDim, yDim)     for T in temperatures]
        simulations2= [NearestNeighbourIsingSimulation2D.generateOrderedSystematicNearestNeighbourSim(T, xDim, yDim) for T in temperatures]
        
        
        averageMagnetisations = [sim.averageFirstNMagnetisations(N) / 1600 for sim in simulations]
        temperatures =          [sim.getInitialTemperature()                      for sim in simulations]
        
        averageMagnetisations2= [sim.averageFirstNMagnetisations(N) / 1600 for sim in simulations2]
        temperatures2 =         [sim.getInitialTemperature()                      for sim in simulations2]

        fig, ax = plt.subplots()
        ax.plot(temperatures, averageMagnetisations, label = "Random")
        ax.plot(temperatures2, averageMagnetisations2, label = "Systematic")
        ax.set_title("Plot of mean |M| against tempurature for Random \nand Systematic Methods in a 40x40 grid")
        ax.set_ylabel("Normalised Magnetisation")
        ax.set_xlabel("Tempurature")
        fig.savefig("MagnetisationAgainstTemperatureForRandomAndSystematicPlot.pdf")
        
    def plotMeanMagnetisationAgainstTemperature():
        #Old
        N = 100000
        temperatures = np.linspace(0.1, 3, num = 8)
        dims = np.linspace(40,100, num=8, dtype = int)
        
        simulationss = [[NearestNeighbourIsingSimulation2D.generateOrderedRandomNearestNeighbourSim(T, D, D)     for T in temperatures] for D in dims]
        
        fig, ax = plt.subplots()
        for i in range(dims.size):
            averageMagnetisations = [sim.averageFirstNMagnetisations(N) for sim in simulationss[i]]
            temperatures =          [sim.getInitialTemperature()                                for sim in simulationss[i]]
            ax.plot(temperatures, averageMagnetisations, label = str(dims[i]))

        
        ax.set_title("Plot of mean |M| against tempurature for Random in a 40x40 grid")
        ax.set_ylabel("Normalised Magnetisation")
        ax.set_xlabel("Tempurature")
        ax.legend()
        fig.savefig("EquilibriumAnalysis\MagnetisationAgainstTemperaturePlot.pdf")
'''      
class HeatCapacityAnalyser():
    
    def plotFirstNEnergies(N, simulations, sameAxis = False, simulationLables = None):
        numberOfSimulations = len(simulations)
        sameAxis |= (numberOfSimulations == 1)
        
        fig, axs = plt.subplots(1 if sameAxis else numberOfSimulations, 1, sharex=True)
        for i in range(numberOfSimulations):
            simulations[i].plotFirstNEnergies(N, ax = (axs if sameAxis else axs[i]), label = ("" if simulationLables == None else simulationLables[i]))
        
        fig.savefig('EnergyPlot.pdf')
        
    def plotDriftingEnergyAtTc():
        temperatures = np.linspace(1.5, 1.6, 3)
        dims = (20,20)
        lengthOfRun = dims[0] * dims[0] * 40
        stateType = RandomNearestNeighbourIsingState2D
        
        

        
        fig, ax = plt.subplots()
        
        for T in temperatures:
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dims, temperature = T, bias = 0.2)
            energies = sim.getEnergies(lengthOfRun)
            time = np.linspace(0,lengthOfRun / sim.getNumberOfSpins() , num = lengthOfRun)
            ax.plot(time, energies, label = "{:.3f}".format(T))
            
        ax.set_title(r"Plot of Energy against time for T ~ $T_c$")
        ax.set_ylabel("Energy")
        ax.set_xlabel("Time")
        ax.legend(loc = "lower left")
        fig.savefig("4-HeatCapacity\FluctuatingEnergyNearTcPlot.png")
        
        
    def collectHeatCapacityData():
        #dims = [(2,2), (3,3), (4,4), (5,5), (6,6), (8,8), (10,10), (15,15), (20,20), (25,25), (30,30)]
        dims = [(5,5), (8,8), (5,5), (8,8), (5,5), (8,8), (5,5), (8,8)]
        #stateType = SystematicNearestNeighbourTemperatureVaryingIsingState2D
        stateType = VectorisedRandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D
        startTemp = 3
        endTemp = 4
        resolution = 100
        statesAtEachTemp = 1500
        temperatures = np.linspace(startTemp, endTemp, num = resolution)
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]))
            temps = list(data[0])
            heatCapacity = list(data[1])
            sim = IsingSimulation.generateOscillatingSim(stateType = stateType, dimensions = D, startTemp = startTemp, endTemp = endTemp, numberOfSteps = statesAtEachTemp * resolution, resolution = resolution, bias = 0.5)
            #temps = []
            #heatCapacity = []

            energies = sim.getEnergies(numberOfStates = statesAtEachTemp * resolution, distanceBetweenStates = sim.getNumberOfSpins())
            energiesSplit = np.array(np.split(energies, resolution))[:, 1000:]  #Ignore first 500 full states
            standardDeviations = np.std(energiesSplit, axis = -1)
            heatC = np.divide(standardDeviations,  temperatures) ** 2
            
            heatCapacity += list(heatC)
            temps += list(temperatures)
            print("done {}".format(D))
            np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), [temps, heatCapacity])

    def collectDiscreteHeatCapacityData():
        dims = [(30,30), (50,50), (80,80), (100,100), (30,30), (50,50), (80,80), (100,100), (30,30), (50,50), (80,80), (100,100)]
        #dims = [(80,80), (100,100), (150,150), (200,200)]
        #dims = [(125,125), (175,175), (225,225)]
        stateType = FastSystematicNearestNeighbourForgetfulIsingState2D
        startTemp = 3 #1
        endTemp = 2 #4
        resolution = 100
        statesAtEachTemp = 500
        temperatures = np.linspace(startTemp, endTemp, num = resolution)
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]), allow_pickle=True)
            temps = list(data[0])
            heatCapacity = list(data[1])
            #temps = []
            #heatCapacity = []
            for T in temperatures:
                sim = IsingSimulation.generateSim(stateType = stateType, dimensions = D, temperature = T)
                energies = sim.getEnergies(numberOfStates = statesAtEachTemp, distanceBetweenStates = sim.getNumberOfSpins())[400:]
                standardDeviation = np.std(energies)
                hC = (standardDeviation / T) ** 2
                temps.append(T)
                heatCapacity.append(hC)
                print("done {}".format(T))
                
            print("done {}".format(D))
            np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]), [temps, heatCapacity])

        
    def plotHeatCapacityData():
        fig, ax = plt.subplots()
        data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureData2.npy", allow_pickle=True)
        for d in data[1:]:
            temperatures = d[2]
            heatCapacity = d[1]
            ax.plot(temperatures, heatCapacity, label = "{}".format(d[0]))
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Heat Capacity")
        ax.set_title("Plot of Heat Capacity Against Temperature")
        ax.legend(loc = "upper right")
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot4.png")
        
        fig2, ax2 = plt.subplots()
        averageOver = 10
        for d in data[-5:]:
            temperatures = d[2]
            accumilativeTemp = np.cumsum(temperatures) 
            timeAverageTemp = (accumilativeTemp[averageOver:] - accumilativeTemp[:-averageOver]) / averageOver
            
            heatCapacity = d[1]
            accumilativeCapacity = np.cumsum(heatCapacity) 
            timeAverageCapacity = (accumilativeCapacity[averageOver:] - accumilativeCapacity[:-averageOver]) / averageOver
            ax2.plot(timeAverageTemp, timeAverageCapacity, label = "{}".format(d[0]))
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Heat Capacity")
        ax2.set_title("Plot of Heat Capacity Against Temperature")
        ax2.legend(loc = "upper right")
        fig2.savefig("4-HeatCapacity\HeatCapacityAgainstTemperatureSmoothedPlot.png")
        
        fig3, ax3 = plt.subplots()
        Tcs = []
        for d in data[1:]:
            temperatures = d[2]
            heatCapacity = d[1]
            area = d[0][0] * d[0][1]
            Tc = temperatures[np.argmax(heatCapacity)]
            Tcs.append([area, Tc])
        Tcs = np.array(Tcs)
        ax3.plot(Tcs[:,0], Tcs[:,1], label = "{}".format(d[0]))
        
        fig, ax = plt.subplots()
        dims = [(30,30), (50,50), (100,100)]
        for D in dims:
            temperatures, heatCapacity = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureData2-{}.npy".format(D[0]))
            ax.plot(temperatures, heatCapacity)
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot2.png")
        
        fig, ax = plt.subplots()
        dims = [(30,30), (50,50), (100,100)]
        for D in dims:
            temperatures, heatCapacity = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData-{}.npy".format(D[0]))
            ax.plot(temperatures, heatCapacity)
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot3.png")
        
        fig, ax = plt.subplots()
        dims = [(30,30), (50,50), (80,80), (100,100)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData3-{}.npy".format(D[0]))
            data2= np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]))
            data = np.concatenate((data, data2), axis = -1)
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            
            def f(x, a, b, c, Z):
                ret = (a /((x - b) ** 2 + c ** 2))
                ret[x>b] += Z
                return ret
            
            popt, _ = scipy.optimize.curve_fit(f, temperatures, heatCapacity)
            #print("centre:", popt[1])
            ax.plot(temperatures, f(temperatures, *popt))
            ax.errorbar(temperatures, heatCapacity, yerr = standardErrors, fmt = "+", label = D)
        ax.legend()
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel("Heat Capacity")
        ax.set_title("Plot of heat capacity at several different Temperatures")
        fig.tight_layout()
        
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot4.png", dpi=300)
        
        fig, ax = plt.subplots()
        dims = [(30,30), (50,50), (80,80), (100,100)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]))
            data = data[:,400:]
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]/ max(averages[:,1])
            
            accumilativeCapacity = np.cumsum(heatCapacity) 
            timeAverageCapacity = (accumilativeCapacity[averageOver:] - accumilativeCapacity[:-averageOver]) / averageOver
            
            accumilativeTemp = np.cumsum(temperatures) 
            timeAverageTemp = (accumilativeTemp[averageOver:] - accumilativeTemp[:-averageOver]) / averageOver
            
            
            ax.errorbar(timeAverageTemp, timeAverageCapacity, yerr = 0, label = D)
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot5.png")
        
        fig, ax = plt.subplots()
        dims = [(2,2)] #, (10,10)] #, (30,30)] #, (50,50), (80,80), (100,100)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureData3-{}.npy".format(D[0]))
                        
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1] #/ max(averages[:,1])
            
            ax.errorbar(timeAverageTemp, timeAverageCapacity, yerr = 0, label = D)
        ax.legend()
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot6.png")
        
        fig, ax = plt.subplots()
        dims = [(4,4), (5,5), (6,6), (8,8), (10,10), (15,15), (20,20), (25,25), (30,30)] # (1,1)  [(1,2), (2,2), (3,3),
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), allow_pickle = True)
            #np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), data)
            df = pd.DataFrame(data.T)
            df = df[df[0] < 3]
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            
            def f(x, a, b, c, Z):
                ret = (a /((x - b) ** 2 + c ** 2))
                ret[x>b] += Z
                return ret
            
            popt, _ = scipy.optimize.curve_fit(f, temperatures, heatCapacity, sigma = standardErrors)
            print("centre:", popt[1], D)
            '''
            accumilativeCapacity = np.cumsum(heatCapacity) 
            timeAverageCapacity = (accumilativeCapacity[averageOver:] - accumilativeCapacity[:-averageOver]) / averageOver
            
            accumilativeTemp = np.cumsum(temperatures) 
            timeAverageTemp = (accumilativeTemp[averageOver:] - accumilativeTemp[:-averageOver]) / averageOver
            '''
            ax.errorbar(temperatures, heatCapacity, yerr = standardErrors,label = D)
            ax.plot(temperatures, f(temperatures, *popt))
        ax.legend()
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot7.png", dpi = 300)
        
        
class FiniteSizeScalingAnalyser():
    
    def plotTcAgainstSize():
        areas = []
        Tcs = []
        
        def f1(x, a, b, c, Z):
            ret = (a /((x - b) ** 2 + c ** 2))
            ret[x>b] += Z
            return ret
            
        def f2(x, a, b, c, Z):
            ret = (a /((x - b) ** 2 + c ** 2))
            return ret
        
        
        fig, ax = plt.subplots()
        dims = [(10,10), (15,15), (20,20), (25,25), (30,30)] # (1,1)  [(1,2), (2,2), (3,3),
        
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), allow_pickle = True)
            #np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), data)
            df = pd.DataFrame(data.T)
            df = df[df[0] < 3]
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            
            popt, _ = scipy.optimize.curve_fit(f1, temperatures, heatCapacity, sigma = standardErrors)
            print("centre:", popt[1], D)
            areas.append(D[0] * D[1])
            Tcs.append(popt[1])
            
        dims = [(4,4), (5,5), (6,6), (8,8)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), allow_pickle = True)
            #np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), data)
            df = pd.DataFrame(data.T)
            #df = df[df[0] < 3]
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            
            popt, _ = scipy.optimize.curve_fit(f2, temperatures, heatCapacity, sigma = standardErrors)
            print("centre:", popt[1], D)
            areas.append(D[0] * D[1])
            Tcs.append(popt[1])
            ax.errorbar(temperatures, heatCapacity, yerr = standardErrors,label = D)
            ax.plot(temperatures, f2(temperatures, *popt))
        fig, ax = plt.subplots()
        ax.plot(areas, Tcs)
        
    def plotTcAgainstSize2():
        #lengthOfRun = 2000000
        #lengthOfRun = 8000000
        #lengthOfRun = 32000000
        lengthOfRun = 64000000
        startTemp = .1
        endTemp = 2.3
        resolution = 6400
        possibleT = np.linspace(startTemp, endTemp, num = resolution)

        #dims = [(10,10), (20,20), (30,30), (40,40), (50,50), (60,60), (70,70), (80,80),(90,90)]
        #dims = [(5,5), (15,15), (25,25), (35,35), (45,45), (55,55), (65,65), (75,75), (85,85), (95,95)]
        #dims = [(105,105), (115,115), (125,125), (135,135), (145,145), (155,155), (165,165), (175,175), (185,185), (195,195)]
        dims = [(185,185)]
        stateType = RandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D
        
        fig, ax = plt.subplots()
        
        for d in dims:
            sim = TemperatureVaryingIsingSim.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, bias = 0.5, numberOfSteps = lengthOfRun, resolution = resolution, dimensions = d)
            magnetisations = sim.getAverageMagnetisationsSplit(lengthOfRun, numberOfSplits = resolution)
            ax.plot(possibleT, magnetisations, label = d)
            argAtTc = np.argmax(magnetisations<0.5)
            Tc = possibleT[argAtTc]
            area = d[0] * d[1]
            data = np.array([[area, Tc]])
            print("done", d, data)
            oldData = np.load("5-FiniteSizeScaling\TcAgainstSizeData5.npy")
            data = np.append(data, oldData, axis = 0)
            np.save("5-FiniteSizeScaling\TcAgainstSizeData5.npy", data)

        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against T showing a change in Tc")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        plt.show()

        
        data = np.load("5-FiniteSizeScaling\TcAgainstSizeData5.npy")
        fig, ax = plt.subplots()
        areas = data[:,0]
        TcAtInftyMinusTc = 2.269185314 - data[:,1]
        ax.plot(areas, TcAtInftyMinusTc, "+")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig("5-FiniteSizeScaling\TcAgainstSize2.png")
        
    def plotDiscritisedMagAgainstT():
        lengthOfRun = 2000000
        startTemp = .1
        endTemp = 2.3
        resolution = 100
        possibleT = np.linspace(startTemp, endTemp, num = resolution)

        dims = [(100,100)]
        stateType = RandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D
        
        fig, ax = plt.subplots()
        for d in dims:
            sim = TemperatureVaryingIsingSim.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, bias = 0.5, numberOfSteps = lengthOfRun, resolution = resolution, dimensions = d)
            magnetisations = sim.getAverageMagnetisationsSplit(lengthOfRun, numberOfSplits = resolution)
            ax.plot(possibleT, magnetisations, label = d)

        sim = TemperatureVaryingIsingSim.generateSim(stateType = RandomNearestNeighbourForgetfulIsingState2D, temperature = endTemp, bias = 0.5, dimensions = dims[-1])
        magnetisations = sim.getAverageMagnetisationsSplit(lengthOfRun, numberOfSplits = resolution)
        ax.plot(possibleT, magnetisations, label = d)

        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against T showing a change in Tc")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        fig.savefig("5-FiniteSizeScaling\DiscretisedMagnetisationAgainstTemperaturePlot2")
    
    def plotMagnetisationAgainstTemperature():
        lengthOfRun = 600000
        startTemp = 0.5
        endTemp = 3
        resolution = 100
        dims = [(50,50), (100,100), (150,150)]
        stateType = RandomNearestNeighbourTemperatureVaryingIsingState2D
        
        fig, ax = plt.subplots()
        for d in dims:
            sim = IsingSimulation.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, bias = 0.5, numberOfSteps = lengthOfRun, resolution = resolution, dimensions = d)
            magnetisations = sim.getMagnetisations(lengthOfRun)
            temperatures   = sim.getTemperatures(lengthOfRun)
            ax.plot(temperatures, magnetisations, label = d)

        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against T showing a change in Tc")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Magnetisation")
        fig.savefig("5-FiniteSizeScaling\MagnetisationAgainstTemperaturePlot")
    
    def plotMagnetisationForVaryingTemperatureState():
        lengthOfRun = 300000
        startTemp = 0.1
        endTemp = 2
        resolution = 100
        dims = (50,50)
        stateType = RandomNearestNeighbourTemperatureVaryingIsingState2D
        
        sim = IsingSimulation.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, numberOfSteps = lengthOfRun, resolution = resolution, dimensions = dims)
        fig, ax = plt.subplots()
        sim.plotFirstNMagnetisations(lengthOfRun, ax = ax)
        fig.savefig("5-FiniteSizeScaling\MagnetisationAgainstTimePlot")
    
class HysteresisAnalyser():
    
    def plotLoopAreaAgainstTemperature():
        data = np.load("6-Hysteresis/AreaData.npy")

        fig, ax = plt.subplots()
        ax.plot(data[:,0], data[:,1], "+")
        (a,b), V = np.polyfit(np.log(data[:,0]), np.log(data[:,1]), deg = 1, cov = True)
        A, N = np.exp(b), a
        AError = np.sqrt(V[1][1]) * A
        NError = np.sqrt(V[0][0])
        print("a: {} +/- {}".format(A, AError))
        print("N: {} +/- {}".format(N, NError))
        fittedAreas = A * data[:,0] ** N
        ax.plot(data[:,0], fittedAreas)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Plot of Hysteresis Area against Temperature\n showing Area proportional to T ^ {:.2f}".format(a))
        ax.set_ylabel("Area of Hysteresis")
        ax.set_xlabel("Temperature")
        fig.savefig("6-Hysteresis\HysteresisAreaAgainstTempLogPlot.png")
        
        fig, ax = plt.subplots()
        ax.plot(data[:,0], data[:,1], "+")
        ax.set_title("Plot of Hysteresis Area against Temperature")
        ax.set_ylabel("Area of hysteresis")
        ax.set_xlabel("Temperature")
        fig.savefig("6-Hysteresis\HysteresisAreaAgainstTempPlot.png")
        
        
    def collectLoopAreaAgainstTemperature():
        numberOfPeriods = 60
        temperatures = np.logspace(.6,3.1,num=10)
        dims = (150,150)
        amplitude = 6
        numberOfSteps = 5000
        stateType = RandomNearestNeighbourMagnetisedIsingState2D
        areas = []
        
        for T in temperatures:
            sim = IsingSimulation.generateOscillatingSim(stateType, temperature = T, dimensions = dims, amplitude = amplitude, numberOfSteps = numberOfSteps, resolution = 600)
            areas.append(sim.getAreaOfHysteresis(period = numberOfSteps, numberOfPeriods = numberOfPeriods))
            
        previousData = np.load("6-Hysteresis/AreaData.npy")
        currentData = np.stack((temperatures, areas), axis = -1)
        currentData = np.append(currentData, previousData, axis = 0)
        np.save("6-Hysteresis/AreaData.npy", currentData)
        
    def plotMagnetisationAgainstFieldForDifferentTempuratures(N = 20):
        temperatures = [1,5,10,15,20,25]
        dims = (40,40)
        amplitude = 6
        numberOfSteps = 4000
        stateType = RandomNearestNeighbourMagnetisedIsingState2D
        fig, ax = plt.subplots()
        
        for T in temperatures:
            sim = IsingSimulation.generateOscillatingSim(stateType = stateType, dimensions = dims, amplitude = amplitude, numberOfSteps = numberOfSteps, resolution = 100, temperature = T)
            magnetisations =  sim.getPeriodAveragedMagnetisations(numberOfSteps, N)
            fields = sim.getFields(numberOfSteps)
            ax.plot(fields, magnetisations, label = "{:.1f}".format(T))
        ax.set_title("Plot of Hysteresis Loop at Different Temperatures")
        ax.legend(loc = "lower left")
        ax.set_xlabel("H Field")
        ax.set_ylabel("Magnetisation")
        fig.savefig('6-Hysteresis\MagnetisationAgainstFieldAtDifferentTempsPlot.png')
        
class LocalOrderAnalyser():
    
    def plotMagetisationRateAgainstMagnetisationAboveTc():
        temp = 10
        dims = (60,60)
        runTime = 50000
        stateType = RandomNearestNeighbourIsingState2D
        startBias = np.linspace(.1,.4, num = 8)
        fig, ax = plt.subplots()
        
        for b in startBias:
            sim = IsingSimulation.generateSim(stateType, bias = b, temperature = temp, dimensions = dims)
            magnetisations = sim.getTimeAveragedMagnetisations(runTime, averageOver = 3600)
            divMagnetisations = np.gradient(magnetisations)
            initialM = 2 * b
            ax.plot(magnetisations, divMagnetisations, label = "{:.3f}".format(initialM))
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation against dM/dt for different initial M")
        ax.set_xlabel("Magnetisation")
        ax.set_ylabel("dM/dT")
        fig.tight_layout()
        fig.savefig('7-LocalOrder\RateOfMagnetisationPlotAboveTc.png')
        
    def plotMagetisationRateAgainstMagnetisationBelowTc():
        temp = .1
        dims = (60,60)
        runTime = 50000
        stateType = RandomNearestNeighbourIsingState2D
        startBias = np.linspace(.1,.4, num = 8)
        fig, ax = plt.subplots()
        
        for b in startBias:
            sim = IsingSimulation.generateSim(stateType, bias = b, temperature = temp, dimensions = dims)
            magnetisations = sim.getTimeAveragedMagnetisations(runTime, averageOver = 3600)
            divMagnetisations = np.gradient(magnetisations)
            initialM = 2 * b
            ax.plot(magnetisations, divMagnetisations, label = "{:.3f}".format(initialM))
        ax.legend(loc = "lower left")
        ax.set_title("Plot of Magnetisation against dM/dt for different initial M")
        ax.set_xlabel("Magnetisation")
        ax.set_ylabel("dM/dT")
        fig.tight_layout()
        fig.savefig('7-LocalOrder\RateOfMagnetisationPlotBelowTc.png')
        
    def animateOrderedState():
        temp = .1
        dims = (100,100)
        numberOfFrames = 400
        distanceBetweenFrames = 250
        stateType = RandomNearestNeighbourIsingState2D
        startBias = 0.2
        
        simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
        
        fig, ani = simulation.animateStates(numberOfFrames, distanceBetweenFrames)
        
    def plotOrderedState():
        temp = .1
        dims = (100,100)
        positions = np.linspace(0,50000, num = 4, dtype = int)
        stateType = RandomNearestNeighbourIsingState2D
        startBias = 0.2
        
        simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
        
        fig, axs = plt.subplots(2,2, figsize = (8,8))
        axs = np.array(axs).flatten()
        for i in range(4):
            axs[i].imshow(simulation.getStateNumber(positions[i]).getStateAsArray(), aspect='equal', cmap=plt.cm.gray, interpolation='nearest', norm=plt.Normalize(0, 1))
            axs[i].set_title("After {:.1f} steps".format(positions[i] / simulation.getNumberOfSpins()))
            
        fig.suptitle("Image of the ising state at a temperature of 0.1")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("7-LocalOrder\OrderedStatesPlot.png")
        
class GlobalOrderAnalyser():
    
    def orderInSystematicPlot():
        temp = 10
        dims = (200,200)
        numberOfStates = 64
        distanceBetweenStates = 40000
        stateType = SystematicNearestNeighbourForgetfulIsingState2D
        fig, axs = plt.subplots()

        sim = PlottableIsingSimulatuion.generateSim(stateType = stateType, temperature = temp, dimensions = dims)
        averageTransform = sim.getAverageFourierTransform(numberOfStates, distanceBetweenStates)
        #averageTransform = np.absolute(sim.getStateNumber(40000).getStateAsDFT())
        axs.imshow(averageTransform, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
        axs.set_title("Plot of Average Fourier Transform")
        fig.tight_layout()
        fig.savefig("8-GlobalOrder\OrderInSystematicPlot.png")
    
    def orderInFastSystematicPlot():
        temp = 10
        dims = (200,200)
        numberOfStates = 64
        distanceBetweenStates = 40000
        stateType = FastSystematicNearestNeighbourForgetfulIsingState2D
        numberOfSplits = [1,2,4,10,400,40000]
        fig, axs = plt.subplots(2,3, figsize = (8,6))
        axs = np.array(axs).flatten()
        for i in range(6):
            sim = PlottableIsingSimulatuion.generateSim(stateType = stateType, temperature = temp, dimensions = dims, numberOfSplits = numberOfSplits[i])
            averageTransform = sim.getAverageFourierTransform(numberOfStates, distanceBetweenStates)
            #averageTransform = np.absolute(sim.getStateNumber(40000).getStateAsDFT())
            axs[i].imshow(averageTransform, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            axs[i].set_title("{:} splits".format(numberOfSplits[i]))
            
        fig.suptitle("Image of the IsingState as a Fourier Transform for different number of splits")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\OrderInFastSystematicPlot.png")
        
    def disorderInRandomPlot():
        temp = 10
        dims = (200,200)
        numberOfStates = 3200
        distanceBetweenStates = 400
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        fig, axs = plt.subplots()
        sim = PlottableIsingSimulatuion.generateSim(stateType = stateType, temperature = temp, dimensions = dims) #axs = np.array(axs).flatten()
        averageTransform = sim.averageFourierTransforms(numberOfStates, distanceBetweenStates)
        axs.imshow(averageTransform, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
        axs.set_title("Fourier Transform for a random IsingState")
        fig.tight_layout()
        fig.savefig("8-GlobalOrder\DisorderInRandomPlot.png")
        
    def animateOrderedFourierTransforms():
        temp = .1
        dims = (100,100)
        numberOfFrames = 400
        distanceBetweenFrames = 250
        stateType = RandomNearestNeighbourIsingState2D
        startBias = 0.2
        
        simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
        
        fig, ani = simulation.animateFourierTransforms(numberOfFrames, distanceBetweenFrames)
        
    def plotOrderedTransform():
        temp = .1
        dims = (200,200)
        positions = np.linspace(0,100000, num = 4, dtype = int)
        stateType = RandomNearestNeighbourIsingState2D
        startBias = 0.2
        
        simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
        
        fig, axs = plt.subplots(2,2, figsize = (8,8))
        axs = np.array(axs).flatten()
        for i in range(4):
            axs[i].imshow(np.absolute(simulation.getStateNumber(positions[i]).getStateAsDFT()), aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            axs[i].set_title("After {:.1f} steps".format(positions[i] / simulation.getNumberOfSpins()))
            
        fig.suptitle("Image of the Ising State as a Fourier Transform\nat a temperature of 0.1")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\OrderedTransformPlot.png")
        
    def plotOrderedTransform2():
        temp = .1
        dims = (200,200)
        positions = np.linspace(0,400000, num = 16, dtype = int)
        stateType = RandomNearestNeighbourIsingState2D
        startBias = 0.2
        
        simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
        
        fig, axs = plt.subplots()
        data = np.array([[0,0]])
        for i in range(16):
            data = np.load("8-GlobalOrder\OrderedTransformData.npy")
            fourierTransform = np.absolute(simulation.getStateNumber(positions[i]).getStateAsDFT())
            fourierTransform = fourierTransform[:,100] + fourierTransform[100,:] + fourierTransform[::-1,100] + fourierTransform[100,::-1]
            
            averageOver = 4
            accumilative = np.cumsum(fourierTransform) 
            rollingAverage = (accumilative[averageOver:] - accumilative[:-averageOver])
            rollingAverage = np.split(rollingAverage, 2)[1]
            rollingAverage = rollingAverage - np.amin(rollingAverage)
            rollingAverage = rollingAverage / rollingAverage[0]
            hasFallenByE = np.greater(1/np.e, rollingAverage)
            yetToFall = itertools.takewhile(lambda x: not x, hasFallenByE)        
            eFallTime = sum(1 for _ in yetToFall)
            print("{}".format(positions[i]))
            data = np.append(data, [[positions[i] / simulation.getNumberOfSpins(), eFallTime]], axis = 0)
            np.save("8-GlobalOrder\OrderedTransformData.npy", data)
        
        axs.plot(data[:,0], data[:,1])
        fig.suptitle("Image of the Ising State as a Fourier Transform\nat a temperature of 0.1")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\OrderedTransformPlot3.png")
        
    def animateOrderWithTemperature():
        timeOfRun = 1200
        startTemp = .1
        endTemp = 1.8
        resolution = 200
        dims = (200,200)
        stateType = SystematicNearestNeighbourTemperatureVaryingIsingState2D
        
        initialState = np.zeros(shape = dims)
        initialState[:dims[0] // 2, :] = 1
        
        sim = PlottableIsingSimulatuion.generateOscillatingSimFromArray(stateType = stateType, array = initialState, startTemp = startTemp, endTemp = endTemp, numberOfSteps = timeOfRun, resolution = resolution) #, dimensions = dims)
        states = sim.getStates(numberOfStates = timeOfRun, distanceBetweenStates = sim.getNumberOfSpins())
        temperatures = sim.getTemperatures(numberOfStates = timeOfRun, distanceBetweenStates = sim.getNumberOfSpins())
        numberOfSpinUp = [np.count_nonzero(state[:dims[0] // 2, :]) for state in states]
        plt.plot(temperatures, numberOfSpinUp)
        plt.show()
        '''
        for i in range(timeOfRun):
            numberOfSpinUp = np.count_nonzero(states[i][:dims[0] // 2, :] == 1)
            print(numberOfSpinUp - dims[0]< dims[0] * dims[0] // 4, "{:0.2}".format(temperatures[i]), numberOfSpinUp)
        #sim.animateFourierTransforms(numberOfFrames = timeOfRun, distanceBetweenFrames = sim.getNumberOfSpins(), fileName = "OrderTransformWithTemperature.mp4")
        #sim.animateStates(numberOfFrames = timeOfRun, distanceBetweenFrames = sim.getNumberOfSpins(), fileName = "OrderWithTemperature.mp4")
        '''
        
    def plotSpinUpAgainstTemp():
        timeOfRun = 4800
        startTemp = .75
        endTemp = 2.3
        resolution = 800
        #dims = [(300,300)]
        dims = [(50,50),(100,100), (200,200), (300,300)]
        stateType = SystematicNearestNeighbourTemperatureVaryingIsingState2D
        data = list(np.load("8-GlobalOrder\SpinUpData2.npy", allow_pickle=True))
        #data = [[np.array([0,0]),np.array([0,0]),np.array([0,0])]]
        
        for D in dims:
            data = list(np.load("8-GlobalOrder\SpinUpData2.npy", allow_pickle=True))
            initialState = np.zeros(shape = D)
            initialState[:D[0] // 2, :] = 1
        
            sim = PlottableIsingSimulatuion.generateOscillatingSimFromArray(stateType = stateType, array = initialState, startTemp = startTemp, endTemp = endTemp, numberOfSteps = timeOfRun, resolution = resolution) #, dimensions = dims)
            states = sim.getStates(numberOfStates = timeOfRun, distanceBetweenStates = sim.getNumberOfSpins())
            temperatures = sim.getTemperatures(numberOfStates = timeOfRun, distanceBetweenStates = sim.getNumberOfSpins())
            numberOfSpinUp = np.array([np.count_nonzero(state[:D[0] // 2, :]) for state in states])
            data.append([np.array(D), temperatures, numberOfSpinUp])
            np.save("8-GlobalOrder\SpinUpData2.npy", data)
            print("done {}".format(D))
        
        for d in data[1:]:
            dim = d[0]
            area = dim[0] * dim[1]
            temperatures = d[1]
            numberOfSpinUp = np.array(d[2])
            plt.plot(temperatures, 2 * numberOfSpinUp / area, label = "{}".format(dim))
        plt.show()
        
    def plotSpinUpAgainstTemp2():
        timeOfRun = 1200
        startTemp = 2.3
        endTemp = .1
        resolution = 800
        #dims = [(300,300)]
        dims = [(50,50),(100,100), (150,150)]
        stateType = SystematicNearestNeighbourTemperatureVaryingIsingState2D
        #data = list(np.load("8-GlobalOrder\SpinUpData3.npy", allow_pickle=True))
        data = [[np.array([0,0]),np.array([0,0]),np.array([0,0])]]
        
        for D in dims:
            #data = list(np.load("8-GlobalOrder\SpinUpData3.npy", allow_pickle=True))
            initialState = np.zeros(shape = D)
            initialState[:D[0] // 2, :] = 1
        
            sim = PlottableIsingSimulatuion.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, numberOfSteps = timeOfRun, resolution = resolution, dimensions = D)
            states = sim.getStates(numberOfStates = timeOfRun, distanceBetweenStates = sim.getNumberOfSpins())
            temperatures = sim.getTemperatures(numberOfStates = timeOfRun, distanceBetweenStates = sim.getNumberOfSpins())
            numberOfSpinUp = np.array([np.count_nonzero(state[:D[0] // 2, :]) for state in states])
            data.append([np.array(D), temperatures, numberOfSpinUp])
            np.save("8-GlobalOrder\SpinUpData3.npy", data)
            print("done {}".format(D))
        
        for d in data[1:]:
            dim = d[0]
            area = dim[0] * dim[1]
            temperatures = d[1]
            numberOfSpinUp = np.array(d[2])
            plt.plot(temperatures, 2 * numberOfSpinUp / area, label = "{}".format(dim))
        plt.show()
        
        