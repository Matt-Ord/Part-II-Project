# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:12:25 2020

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt
from isingSimulation import IsingSimulation, PlottableIsingSimulatuion
from isingState import VectorisedRandomNearestNeighbourIsingState1D, VectorisedRandomNearestNeighbourIsingState3D, SystematicNearestNeighbourIsingState1D, VectorisedRandomNearestNeighbourTemperatureVaryingIsingState2D, RandomNearestNeighbourMagnetisedForgetfulIsingState2D, VectorisedRandomNearestNeighbourMagnetisedIsingState2D, VectorisedRandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D, VectorisedRandomNearestNeighbourIsingState2D, FastSystematicNearestNeighbourForgetfulIsingState2D, SystematicNearestNeighbourTemperatureVaryingIsingState2D, SystematicNearestNeighbourForgetfulIsingState2D, RandomNearestNeighbourTemperatureVaryingIsingState2D, RandomNearestNeighbourForgetfulIsingState3D, RandomNearestNeighbourForgetfulIsingState1D, RandomNearestNeighbourForgetfulIsingState2D, RandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D, SystematicIsingState, RandomNearestNeighbourMagnetisedIsingState2D, RandomNearestNeighbourIsingState2D, InterpolatedIsingState, SystematicNearestNeighbourIsingState2D, IsingState, IsingState2D, NearestNeighbourIsingState
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
        ax.errorbar(temperatures2, times2, yerr = standardErrors2,fmt = "+")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel("Equilibrium Time")
        ax.set_title("Plot of Equilibrium Time Against Temperature")
        ax.set_xscale("log")
        ax.set_yscale("log")
        #data2 = np.load("EquilibriumTimeTemperatureData2.npy")
        #print(data2[:,0])
        #ax.plot(data2[:,0], data2[:,1], "+")
        fig.savefig("1-TimeToEquilibrium\EquilibriumTimeAgainstTemperaturePlot6.png")
        
        fig, ax = plt.subplots()
        data = np.load("1-TimeToEquilibrium\TimeAgainstTempData{:}.npy".format(60))
        temperatures = list(data[0])
        times = list(data[1])
        
        ax.errorbar(temperatures, times, fmt = "+")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel("Equilibrium Time")
        ax.set_title("Plot of Equilibrium Time Against Temperature")
        ax.set_xscale("log")
        ax.set_yscale("log")
        #data2 = np.load("EquilibriumTimeTemperatureData2.npy")
        #print(data2[:,0])
        #ax.plot(data2[:,0], data2[:,1], "+")
        fig.savefig("1-TimeToEquilibrium\EquilibriumTimeAgainstTemperaturePlot6.png")
        
    def collectEquilibriumTimeAgainstTempurature():
        temperatures = 2.269185314 + np.logspace(-1, 3, 80)[40:]
        dim = (60,60)
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        numberOfStates = 14400
        
        #data = np.load()
        #temps = list(data[0])
        #times = list(data[1])
        temps = []
        times = []
        for T in temperatures:
            sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, dimensions = dim, bias = 0.2)
            times.append(sim.getEquilibriumTime(lengthOfSample = numberOfStates, distanceBetweenSamples = 1, tolerance = 0.3678794412, equilibriumMagnetisation = 0))
            temps.append(T)
            print("Done {}".format(T))
        np.save("1-TimeToEquilibrium\TimeAgainstTempData{:}.npy".format(dim[0]), [temps, times])
                
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
            
    def collectMagnetisations():
        temperatures = [0.01,0.1,.5,1,1.4]
        dim = (200,200)
        stateType = SystematicNearestNeighbourIsingState2D
        numberOfStates = 30
        distanceBetweenStates = dim[0] * dim[0]
        
        #data = np.load()
        #temps = list(data[0])
        #times = list(data[1])
        temps = []
        times = []
        magnetisations = []
        for T in temperatures:
            sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, dimensions = dim, bias = 0.3)
            
            time = np.arange(0, numberOfStates * distanceBetweenStates, distanceBetweenStates) / sim.getNumberOfSpins()
            times.append(time)
            magnetisations.append(sim.getMagnetisations(numberOfStates = numberOfStates, distanceBetweenStates = distanceBetweenStates))
            temps.append(T)
            print("Done: {}".format(T))
        np.save("1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcDataLargeGrid.npy", [temps, times, magnetisations])
    
    def plotMagnetisationsForTemperatures():
        data = np.load("1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcData.npy", allow_pickle = True)
        temperatures = data[0]
        times = data[1]
        magnetisations = data[2]
        fig, ax = plt.subplots()
        for i in range(len(temperatures)):
            ax.plot(times[i], magnetisations[i], label = temperatures[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        ax.set_title(r"Plot of Magnetisation Below $T_c$")
        ax.legend()
        fig.savefig("1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcPlot.png", dpi = 300)
        
        data = np.load("1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcDataLargeGrid.npy", allow_pickle = True)
        temperatures = data[0]
        times = data[1]
        magnetisations = data[2]
        fig, ax = plt.subplots()
        for i in range(len(temperatures)):
            ax.plot(times[i], magnetisations[i], label = temperatures[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        ax.set_title(r"Plot of Magnetisation Below $T_c$")
        ax.legend()
        fig.savefig("1-TimeToEquilibrium\MagnetisationAgainstTimeBelowTcPlot2.png", dpi = 300)
        
class AutocorrelationAnalyser():
            
    def plotAutocorrelationLagTime():
        
        calculatedCs = np.load("2-Autocorrelation\lagTimeHighTemp2D.npy")
        print(calculatedCs)
        print("{} +- {} 2D".format(np.average(calculatedCs), np.std(calculatedCs)))
        
        
        
        data = np.load("2-Autocorrelation\lagTimeAgainstTempurature.npy")
        df = pd.DataFrame(data)
        averages = df.groupby(0,as_index=False).mean().values
        standardErrors = df.groupby(0,as_index=False).sem().values[:,1]

        temperatures = averages[:,0]
        lagTimes = averages[:,1]
        
        fig2, ax2 = plt.subplots()
        def f4(x, A, B, c):
            return  (A * (x) ** 2) /((x - B) ** 2 + c ** 2 )

        popt, pcov = scipy.optimize.curve_fit(f4, temperatures,lagTimes, maxfev=10000, absolute_sigma = False)

        spacing = np.linspace(0, 20, 1000)
        ax2.errorbar(temperatures, lagTimes, yerr = standardErrors, fmt = "+")
        ax2.plot(spacing, f4(spacing, *popt))
        ax2.set_title("Plot of Autocorrelation fall Time against Temperature")
        ax2.set_xlabel(r"Temperature  $T / T_0$")
        ax2.set_ylabel("Fall Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot.png", dpi = 300)
            
        a,b,c = popt
        tc = (c ** 2 + b ** 2) / b
        print("Tc 2D", tc)
        
        fig, ax = plt.subplots()
        dims = [(50,50),]
        for d in dims:
            data = np.load("2-Autocorrelation\lagTimeAgainstTempuratureDim{}.npy".format(d[0]))
            df = pd.DataFrame(data)
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
    
            temperatures = averages[:,0]
            lagTimes = averages[:,1]

            ax.errorbar(temperatures, lagTimes, yerr = standardErrors, fmt = "+")
        
        ax.set_title("Plot of Autocorrelation fall Time against Temperature")
        ax.set_xlabel(r"Temperature  $T / T_0$")
        ax.set_ylabel("Fall Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig.savefig("2-Autocorrelation\AutocorrelationAgainstTempuratureLongSample.png", dpi = 300)
        
        fig, ax = plt.subplots()
        dims = [(10,10), (5,5), (2,2)]
        for d in dims:
            data = np.load("2-Autocorrelation\lagTimeAgainstTempuratureDim{}2.npy".format(d[0]))
            df = pd.DataFrame(data)
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
    
            temperatures = averages[:,0]
            lagTimes = averages[:,1]

            ax.errorbar(temperatures, lagTimes, label = d, yerr = standardErrors, fmt = "+")
        
        ax.set_title("Plot of Autocorrelation Fall Time against Temperature")
        ax.set_xlabel(r"Temperature  $T / T_0$")
        ax.set_ylabel("Fag Time")
        ax.legend()
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlotLowN.png", dpi = 300)
        print("Values", popt)
        print("errors", np.sqrt(np.diag(pcov)))
        
    
        data = np.load("2-Autocorrelation\lagTimeAgainstTempurature3D.npy")
        
        df = pd.DataFrame(data)
        averages = df.groupby(0,as_index=False).mean().values
        temperatures = averages[2:,0]
        lagTimes = averages[2:,1]
        #temperatures = data[1:,0][data[1:,0] > 5]
        #lagTimes = data[1:,1][data[1:,0] > 5]
        calculatedCs = np.load("2-Autocorrelation\lagTimeHighTemp3D.npy")
        print("{} +- {}".format(np.average(calculatedCs), np.std(calculatedCs)))
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(temperatures, lagTimes, "+")
        ax2.set_title("Plot of Autocorrelation Fall Time against Temperature for 3D")
        ax2.set_xlabel(r"Temperature  $T / T_0$")
        ax2.set_ylabel("Fall Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot3D.png", dpi = 300)
        
        fig, ax = plt.subplots()
        def f4(x, A, B, c):
            return  (A * (x) ** 2) /((x - B) ** 2 + c ** 2 )

        popt, pcov = scipy.optimize.curve_fit(f4, temperatures,lagTimes, maxfev=10000, absolute_sigma = False)

        ax.plot(temperatures, lagTimes, "+")
        spacing = np.linspace(0, 30, 1000)
        ax.plot(spacing, f4(spacing, *popt))
        ax.set_xlabel("Temperature T/$T_0$")
        ax.set_ylabel("Fall Time")
        ax.set_title("Plot of Autocorrelation Fall Time against Temperature for 3D")
        fig.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot3D2.png", dpi = 300)
        print("Values", popt)
        print("errors", np.sqrt(np.diag(pcov)))
        a,b,c = popt
        tc = (c ** 2 + b ** 2) / b
        print("Tc 3D", tc)
        
        data = np.load("2-Autocorrelation\lagTimeAgainstTempurature1D.npy")[:-38]
        
        temperatures = data[2:-37,0]
        lagTimes = data[2:-37,1]
        #9000
        
        temperatures2 = data[-18:,0]
        lagTimes2 = data[-18:,1]
        
        calculatedCs = np.load("2-Autocorrelation\lagTimeHighTemp1D.npy")
        print(calculatedCs)
        print("{} +- {}".format(np.average(calculatedCs), np.std(calculatedCs)))
        
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(temperatures, lagTimes, "+")
        #ax2.plot(temperatures2, lagTimes2, "+")
        ax2.set_title("Plot of Autocorrelation Fall Time against Temperature for 1D")
        ax2.set_xlabel(r"Temperature $T / T_0$")
        ax2.set_ylabel("Fall Time")
        #ax2.set_xscale("log")
        #ax2.set_yscale("log")
        fig2.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot1D.png", dpi = 300)
        
        #height = 0.5 * (b ** 2 + c ** 2) / (c ** 2))
        fig, ax = plt.subplots()
        def f4(x, B):
            H = 4.5
            return  (0.5 * (x) ** 2) /((x - B) ** 2 + ((B ** 2) / (2 * (H - 0.5))) )
        popt, pcov = scipy.optimize.curve_fit(f4, temperatures,lagTimes, maxfev=10000, absolute_sigma = False)
        
        ax.plot(temperatures, lagTimes, "+")
        spacing = np.linspace(0, 25, 1000)
        ax.plot(spacing, f4(spacing, *popt))
        ax.set_xlabel("Temperature T/$T_0$")
        ax.set_ylabel("Fall Time")
        ax.set_title("Plot of Autocorrelation Fall Time against Temperature for 1D")

        fig.savefig("2-Autocorrelation\AutocorrelationAgainstTempuraturePlot1D2.png", dpi = 300)
        print("Values", popt)
        print("errors", np.sqrt(np.diag(pcov)))
        
        b = popt[0]
        c = 0.35355 * b
        tc = (c ** 2 + b ** 2) / b
        print("Tc 1D", tc)
        
        data = np.load("2-Autocorrelation\lagTimeAgainstSize1.8.npy")

        areas = data[data[:,0]> 150,0]
        lagTimes = data[data[:,0]> 150,1]
        #print(data[data[:,1]> 1.5, :])
        
        #Plot lag times
        fig2, ax2 = plt.subplots()
        ax2.plot(areas, lagTimes, "+")
        ax2.set_title("Plot of Autocorrelation Fall Time against Size")
        ax2.set_xlabel("Area of Lattice")
        ax2.set_ylabel("Fall Time")
        fig2.savefig("2-Autocorrelation\AutocorrelationTimeAgainstSizePlot1.8.png", dpi = 300)
        
    
    def collectAutocorrelationLagTimeAgainstTempurature():
        dims = [(50,50)] #[(2,2), (2,2), (2,2), (2,2), (2,2), (2,2)] #13.1
        temperatures = np.arange(2.025, 3.025, step = 0.05)
        #sampleTime = 1500
        #ignore = 500
        ignore = 200
        sampleTime = 500
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        data = [[0,0], [0,0]]
        for dim in dims:
            for T in temperatures:
                #data = np.load("2-Autocorrelation\lagTimeAgainstTempuratureDim{}2.npy".format(dim[0]))
                sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dim, temperature = T, bias = 0.5)
                lengthOfSample = sampleTime * dim[0] * dim[1]
                data = np.append(data, [[sim.getInitialTemperature(), sim.getAutocorrelationLagTime(lengthOfSample, distanceBetweenSamples = 1, statesToIgnore = 200)]], axis = 0)
                print("Done : {}".format(T))
                np.save("2-Autocorrelation\lagTimeAgainstTempuratureDim{}2.npy".format(dim[0]), data)
        
    def estimateAutocorrelationLagTimeHighTempurature():
        dim = (150,150)
        T = 100000000
        sampleTime = 200
        stateType = RandomNearestNeighbourForgetfulIsingState2D
        lengthOfSample = sampleTime * dim[0] * dim[1]
        times = []
        for _ in range(10):
            sim = IsingSimulation.generateSim(stateType = stateType, dimensions = dim, temperature = T)
            times.append(sim.getAutocorrelationLagTime(lengthOfSample, distanceBetweenSamples = 1, statesToIgnore = 100))
            print("Done : {}".format(T))
            np.save("2-Autocorrelation\lagTimeHighTemp2D.npy", times)
        print(np.average(times))
        print(np.std(times))
        np.save("2-Autocorrelation\lagTimeHighTemp2D.npy", times)
        
  
    def collectAutocorrelationLagTimeAgainstTempurature1D():
        dim = (900000)
        temperatures = np.arange(0.1, 2, 0.1)
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
        temperatures = np.arange(0.1, 5, 0.1)
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
        ax.set_title("Plot of Autocorrelation for several lattice sizes")
        fig.savefig("2-Autocorrelation\AutocorrelationForSizePlot.png")
        
    def collectAutocorrelationLagTimeAgainstLatticeSize():
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
    
class EquilibriumAnalyser():
                
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
        endTemp = 10
        resolution = 400
        dims = [(5,5)] #[(10,10), (10,10), (10,10), (10,10)] #[(50,50)] #[(5,5),(8,8),(10,10)] #[(50,50), (100,100), (150,150), (200,200)]
        stateType = RandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D
        
        for d in dims:
            #temps, magnetisations = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData3-{}.npy".format(d[0]))
            #temps = list(temps)
            #magnetisations = list(magnetisations)
            sim = IsingSimulation.generateOscillatingSim(stateType = stateType, startTemp = startTemp, endTemp = endTemp, bias = 0.5, numberOfSteps = timeAtEachTemperature * resolution, resolution = resolution, dimensions = d)
            magnetisations = []
            temps = []
            mags = sim.getMagnetisations(resolution * timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins())
            magsSplit = np.split(mags, resolution)
            magnetisations += list(np.average(magsSplit, axis = 1))
            temps += list(np.linspace(startTemp, endTemp, num = resolution))
            print("Done {}".format(d))
            np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureData3-{}.npy".format(d[0]), [temps, magnetisations])
        
    def collectMagnetisationAgainstTemperature():
        timeAtEachTemperature = 40
        temperatures = np.linspace(0.1, 2, 10)
        dims = [(50,50),] # (20,20), (50,50)] #, (5,5), (5,5), (5,5), (5,5)]
        stateType = VectorisedRandomNearestNeighbourIsingState2D
        
        for d in dims:
            magnetisations = []
            temps = []
            #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3-{}.npy".format(d[0]))
            #magnetisations = list(data[1])
            #temps = list(data[0])
            for t in temperatures:
                sim = IsingSimulation.generateSim(stateType = stateType, bias = 0.5, dimensions = d, temperature = t)
                magnetisations.append(abs(sim.getAverageMagnetisation(timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins())))
                temps.append(t)
                np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3-{}.npy".format(d[0]), [temps, magnetisations])
            print("done {}".format(d))
            np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3-{}.npy".format(d[0]), [temps, magnetisations])

    def collectMagnetisationAgainstTemperature1D():
        timeAtEachTemperature = 200
        statesToIgnore = 50
        temperatures = np.linspace(0.0001, 10, 100)
        dims = [(1000,)]
        stateType = VectorisedRandomNearestNeighbourIsingState1D
        #stateType = SystematicNearestNeighbourIsingState1D
        globalH = 1
        for d in dims:
            #magnetisations = []
            #temps = []
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData1D-{}.npy".format(d[0]))
            magnetisations = list(data[1])
            temps = list(data[0])
            for t in temperatures:
                sim = IsingSimulation.generateSim(stateType = stateType, bias = 0, dimensions = d, temperature = t, solverKwargs = {"globalH": globalH})
                magnetisations.append(sim.getAverageMagnetisation(timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins(), statesToIgnore = statesToIgnore))
                temps.append(t)
                np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData1D-{}.npy".format(d[0]), [temps, magnetisations])
            print("done {}".format(d))
            np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData1D-{}.npy".format(d[0]), [temps, magnetisations])

    def collectMagnetisationAgainstTemperature3D():
        timeAtEachTemperature = 100
        statesToIgnore = 50
        temperatures = np.linspace(4,5, 5)
        d = (60,60,60)
        stateType = VectorisedRandomNearestNeighbourIsingState3D
        
        magnetisations = []
        temps = []
        #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3D-{}.npy".format(d[0]))
        #magnetisations = list(data[1])
        #temps = list(data[0])
        for t in temperatures:
            sim = IsingSimulation.generateSim(stateType = stateType, bias = 0.5, dimensions = d, temperature = t)
            magnetisations.append(sim.getAverageMagnetisation(timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins(), statesToIgnore = statesToIgnore))
            temps.append(t)
            np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3D-{}.npy".format(d[0]), [temps, magnetisations])
        print("done {}".format(d))
        np.save("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3D-{}.npy".format(d[0]), [temps, magnetisations])


    def plotMagnetisationAgainstTemperature():
        fig, ax = plt.subplots()
        data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData.npy")
        ax.plot(data[0], data[1], "+", label = "100")
        
        #data1 = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData1.npy")
        #ax.plot(data1[0], data1[1], "+", label = "200")
        
        data2 = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2.npy")
        ax.plot(data2[0], data2[1], "+", label = "50")
        
        temperatures = np.linspace(1, 5, num = 2000)
        theoreticalMagnetisation = np.nan_to_num((1 - (np.sinh(2/temperatures)) ** -4) ** (1/8))
        ax.plot(temperatures, theoreticalMagnetisation)
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot2.png")
        
        
        fig, ax = plt.subplots()
        dims = [(50,50), (10,10), (5,5)] #[(5,5),(8,8),(10,10)]
        
        for D in dims:
            #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2-{}.npy".format(D[0]))
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData3-{}.npy".format(D[0]))
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]

            ax.plot(temperatures, magnetisations, label = D)
                
        temperatures = np.linspace(1, 5, num = 2000)
        theoreticalMagnetisation = np.nan_to_num((1 - (np.sinh(2/temperatures)) ** -4) ** (1/8))
        ax.plot(temperatures, theoreticalMagnetisation, label = "Theory")
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot3.png")
        
        fig, ax = plt.subplots()
        dims = [(2,2), (5,5), (50,50)] #[(5,5),(8,8),(10,10)]
        
        for D in dims:
            #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2-{}.npy".format(D[0]))
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData1-{}.npy".format(D[0]))
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]

            ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = D)
                
        temperatures = np.linspace(1, 5, num = 2000)
        theoreticalMagnetisation = np.nan_to_num((1 - (np.sinh(2/temperatures)) ** -4) ** (1/8))
        ax.plot(temperatures, theoreticalMagnetisation, label = "Theory")
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot4.png")
        
        fig, ax = plt.subplots()
        dims = [(10,10), (20,20), (50,50)]
        
        for D in dims:
            #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2-{}.npy".format(D[0]))
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData2-{}.npy".format(D[0]))
            print(data)
            data = data[:,-6:]
            print(data)
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]

            ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = D)
                
        temperatures = np.linspace(1, 5, num = 2000)
        theoreticalMagnetisation = np.nan_to_num((1 - (np.sinh(2/temperatures)) ** -4) ** (1/8))
        ax.plot(temperatures, theoreticalMagnetisation, label = "Theory")
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot5.png")
        
        fig, ax = plt.subplots()
        dims = [(10,10), (20,20), (50,50), (30,30)]
        
        for D in dims:
            #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2-{}.npy".format(D[0]))
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3-{}.npy".format(D[0]))
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]

            ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = D)
            
                
        temperatures = np.linspace(1, 5, num = 2000)
        theoreticalMagnetisation = np.nan_to_num((1 - (np.sinh(2/temperatures)) ** -4) ** (1/8))
        ax.plot(temperatures, theoreticalMagnetisation, label = "Theory")
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot5.png")
        
        
        fig, ax = plt.subplots()
        dims = [(10,10), (30,30), (50,50)]
        
        for D in dims:
            #data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData2-{}.npy".format(D[0]))
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureData-{}withBias.npy".format(D[0]))
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]

            ax.errorbar(temperatures, magnetisations, yerr = standardErrors, fmt = "+", label = D)
                
        temperatures = np.linspace(1, 5, num = 2000)
        theoreticalMagnetisation = np.nan_to_num((1 - (np.sinh(2/temperatures)) ** -4) ** (1/8))
        #ax.plot(temperatures, theoreticalMagnetisation, label = "Theory")
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot5.png")
        
        fig, ax = plt.subplots()
        dims = [(1000,)]
        
        for D in dims:            
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData1D-{}.npy".format(D[0]))
            df = pd.DataFrame(data.T)
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]
            
            ax.errorbar(temperatures, magnetisations,yerr = standardErrors ,fmt = "+", label = "N = {}".format(D[0]))
                
        temperatures = np.linspace(0.01, 10, num = 2000)
        h = 1
        beta = 1 / temperatures
        #theoreticalMagnetisation = np.sin(a) / np.sqrt(np.sin(a) ** 2 + np.exp(-2 / temperatures))
        theoreticalMagnetisation = (np.sinh(beta * h) * (1 + (np.cosh(h * beta) / np.sqrt(np.sinh(h * beta) ** 2 + np.exp(-4 * beta))))) / (np.cosh(beta * h) + np.sqrt(np.sinh(h * beta) ** 2 + np.exp(-4 * beta)))
        ax.plot(temperatures, theoreticalMagnetisation, label = "Theory")
        
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot1D.png")
        
        fig, ax = plt.subplots()
        dims = [(30,30,30),]
        
        for D in dims:            
            data = np.load("3-MeanMagnetisation\MagnetisationAgainstTemperatureDiscreteData3D-{}.npy".format(D[0]))
            df = pd.DataFrame(data.T)
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            magnetisations = averages[:,1]
            
            ax.errorbar(temperatures, magnetisations,yerr = standardErrors ,fmt = "+", label = "N = {}".format(D[0] * D[1] * D[2]))

        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation Against Temperature")
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Magnetisation $M / M_0$")
        fig.savefig("3-MeanMagnetisation\MagnetisationAgainstTemperaturePlot3D.png")

class HeatCapacityAnalyser():
            
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
        #dims = [(2,2), (3,3),(3,4), (4,4), (5,5), (6,6),(7,7), (8,8), (9,9), (10,10),(12,12), (15,15), (17,17), (20,20), (25,25), (30,30)]
        dims = [(12,12), (17,17), (7,7), (3,4), (9,9), (12,12), (17,17), (7,7), (3,4), (9,9)]
        dims = [[2,3], [2,3], [2,3]]
        #stateType = SystematicNearestNeighbourTemperatureVaryingIsingState2D
        stateType = VectorisedRandomNearestNeighbourTemperatureVaryingForgetfulIsingState2D
        startTemp = 9
        endTemp = 11
        resolution = 200
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
        #dims = [(30,30), (50,50), (80,80), (100,100), (30,30), (50,50), (80,80), (100,100), (30,30), (50,50), (80,80), (100,100)]
        #dims = [(80,80), (100,100), (150,150), (200,200)]
        dims = [(200,200), (300,300)]
        stateType = FastSystematicNearestNeighbourForgetfulIsingState2D
        startTemp = 2.27 #1
        endTemp = 2.25 #4
        resolution = 3
        statesAtEachTemp = 1500
        temperatures = np.linspace(startTemp, endTemp, num = resolution)
        for D in dims:
            #data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]), allow_pickle=True)
            #temps = list(data[0])
            #heatCapacity = list(data[1])
            temps = []
            heatCapacity = []
            for T in temperatures:
                sim = IsingSimulation.generateSim(stateType = stateType, dimensions = D, temperature = T)
                energies = sim.getEnergies(numberOfStates = statesAtEachTemp, distanceBetweenStates = sim.getNumberOfSpins())[1000:]
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
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Heat Capacity $C$ / $C_0$")
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
        ax2.set_xlabel(r"Temperature $T / T_0$")
        ax2.set_ylabel(r"Heat Capacity $C$ / $C_0$")
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
            #ax.plot(temperatures, f(temperatures, *popt))
            ax.errorbar(temperatures, heatCapacity, yerr = standardErrors, fmt = "+", label = D)
        ax.legend()
        ax.set_xlabel(r"Temperature $T / T_0$")
        ax.set_ylabel(r"Heat Capacity $C$ / $C_0$")
        ax.set_title("Plot of heat capacity at several different Temperatures")
        fig.tight_layout()
        
        fig.savefig("4-HeatCapacity\HeatCapacityAgainstTemperaturePlot.png", dpi=300)
        
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
        widths = []
        heights = []
        areas2 = []
        heights2 = []
        
        def f1(x, a, b, c, Z):
            ret = (a /((x - b) ** 2 + c ** 2))
            ret[x>b] += Z
            return ret
            
        def f2(x, a, b, c):
            ret = (a /((x - b) ** 2 + c ** 2))
            return ret
        
        def f3(x, a, b, c):
            ret = (a  * (x ** 2))/((x - b) ** 2 + c ** 2)
            return ret
        
        
        fig, ax = plt.subplots()
        dims = [(2,3)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), allow_pickle = True)
            #np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), data)
            df = pd.DataFrame(data.T)
            #df = df[df[0] <5.5]
            #df = df[df[0] > 1.5]
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            ax.plot(temperatures, heatCapacity)
            
            a = 0.42
            b = 2.08
            c = 0.83
            popt = a,b,c
            tc = (c ** 2 + b ** 2) / b
            print("Tc:", tc, D)
            temperatures = np.linspace(0,7, num = 1000)
            tc = temperatures[np.argmax(f3(temperatures, *popt))]
            areas.append(D[0] * D[1])
            areas2.append(D[0] * D[1])
            Tcs.append(tc)
            widths.append(c)
            heights.append(a * (b ** 2 + c ** 2) / (c ** 2))
            heights2.append(max(heatCapacity))
            ax.plot(temperatures, f3(temperatures, *popt))
            
        averageOver = 10
        #dims = [(10,10), (15,15), (20,20),]# (25,25), (30,30)] # (1,1)  [(1,2), (2,2), (3,3),
        dims = [(2,2),(3,3),(3,4), (4,4), (5,5), (6,6),(7,7), (8,8), (9,9), (10,10),(12,12), (15,15), (17,17), (20,20), (25,25), (30,30)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), allow_pickle = True)
            #np.save("4-HeatCapacity\HeatCapacityAgainstTemperatureSmallGridData-{}.npy".format(D[0] * D[1]), data)
            df = pd.DataFrame(data.T)
            #df = df[df[0] <5.5]
            #df = df[df[0] > 1.5]
            averages = df.groupby(0,as_index=False).mean().values
            standardErrors = df.groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            ax.plot(temperatures, heatCapacity)
            
            accumilativeCapacity = np.cumsum(heatCapacity) 
            timeAverageCapacity = (accumilativeCapacity[averageOver:] - accumilativeCapacity[:-averageOver]) / averageOver
            
            accumilativeTemp = np.cumsum(temperatures) 
            timeAverageTemp = (accumilativeTemp[averageOver:] - accumilativeTemp[:-averageOver]) / averageOver
            

            p0 = (max(heatCapacity) / 4,2.6, 1)
            popt, _ = scipy.optimize.curve_fit(f3, timeAverageTemp, timeAverageCapacity, p0 = p0, maxfev=10000)
            a, b, c = popt
            tc = (c ** 2 + b ** 2) / b
            print("Tc:", tc, D, popt)
            temperatures = np.linspace(0,7, num = 1000)
            tc = temperatures[np.argmax(f3(temperatures, *popt))]
            areas.append(D[0] * D[1])
            areas2.append(D[0] * D[1])
            Tcs.append(tc)
            widths.append(c)
            heights.append(a * (b ** 2 + c ** 2) / (c ** 2))
            heights2.append(max(heatCapacity))
            ax.plot(temperatures, f3(temperatures, *popt))
            
        dims = [(30,30), (50,50), (80,80), (100,100)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]))
            data = data[:,400:]
            averages = pd.DataFrame(data.T).groupby(0,as_index=False).mean().values
            standardErrors = pd.DataFrame(data.T).groupby(0,as_index=False).sem().values[:,1]
            temperatures = averages[:,0]
            heatCapacity = averages[:,1]
            
            accumilativeCapacity = np.cumsum(heatCapacity) 
            timeAverageCapacity = (accumilativeCapacity[averageOver:] - accumilativeCapacity[:-averageOver]) / averageOver
            
            accumilativeTemp = np.cumsum(temperatures) 
            timeAverageTemp = (accumilativeTemp[averageOver:] - accumilativeTemp[:-averageOver]) / averageOver
            heights.append(np.amax(timeAverageCapacity))
            heights2.append(np.amax(timeAverageCapacity))
            areas2.append(D[0] * D[1])
        dims = [(200,200), (300,300)]
        for D in dims:
            data = np.load("4-HeatCapacity\HeatCapacityAgainstTemperatureDiscreteData4-{}.npy".format(D[0]))
            heights.append(np.amax(data[1]))
            heights2.append(np.amax(data[1]))
            areas2.append(D[0] * D[1])
            
        ax.set_xlim(left = None, right = 4)
        ax.set_title("Plot of Heat Capacity against Temeprature")
        ax.set_xlabel("Temeprature T / $T_0$")
        ax.set_ylabel("Heat Capacity C / $C_0$")
        fig.savefig("4-HeatCapacity\HeatCapacityFit.png", dpi = 300)
        
        fig, ax = plt.subplots()
        Tcs = np.array(Tcs) - 2.269185314
        
        (b,a), V = np.polyfit(np.log(areas), np.log(Tcs), deg = 1, cov = True)
        A, N = np.exp(a), b
        AError = np.sqrt(V[1][1]) * A
        NError = np.sqrt(V[0][0])
        print("a: {} +/- {}".format(A, AError))
        print("N: {} +/- {}".format(N, NError))
        spacing = np.logspace(.5,3)
        fittedTcs = A * spacing ** N
        ax.plot(spacing, fittedTcs)

        #areas.append(2)
        #Tcs.append(3.5)
        ax.plot(areas, Tcs, "+")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number Of Spins in Lattice")
        ax.set_ylabel(r"Critical Temp. (T - $T_c(\infty)$) / $T_0$")
        ax.set_title(r"Plot of $T_c$ against Lattice Area")
        fig.savefig("5-FiniteSizeScaling\TcAgainstAreaPlot.png", dpi = 300)
        
        fig, ax = plt.subplots()
        (b,a), V = np.polyfit(np.log(areas[:-2]), np.log(widths[:-2]), deg = 1, cov = True)
        A, N = np.exp(a), b
        AError = np.sqrt(V[1][1]) * A
        NError = np.sqrt(V[0][0])
        print("A: {} +/- {}".format(A, AError))
        print("N: {} +/- {}".format(N, NError))
        spacing = np.logspace(.5,3)
        fittedWidth = A * spacing ** N
        ax.plot(spacing, fittedWidth)
        
        ax.plot(areas, widths , "+")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number Of Spins in Lattice")
        ax.set_ylabel("Heat capacity width")
        ax.set_title("Plot of Heat Capacity width against Lattice Area")
        fig.savefig("5-FiniteSizeScaling\WidthAgainstAreaPlot.png", dpi = 300)
        
        fig, ax = plt.subplots()
        (b,a), V = np.polyfit(np.log(areas2[5:]), np.log(heights[5:]), deg = 1, cov = True)
        A, N = np.exp(a), b
        AError = np.sqrt(V[1][1]) * A
        NError = np.sqrt(V[0][0])
        print("A: {} +/- {}".format(A, AError))
        print("N: {} +/- {}".format(N, NError))
        spacing = np.logspace(.5,5)
        fittedWidth = A * spacing ** N
        ax.plot(spacing, fittedWidth)
        
        ax.plot(areas2, heights , "+")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number Of Spins in Lattice")
        ax.set_ylabel(r"Max Heat capacity C / $C_0$")
        ax.set_title("Plot of max Heat Capacity against Lattice Area")
        fig.savefig("5-FiniteSizeScaling\HeightAgainstAreaPlot.png", dpi = 300)
        
        heightsDivArea = np.divide(heights, areas2)
        heights2DivArea = np.divide(heights2, areas2)
        
        fig, ax = plt.subplots()
        (m,c), V = np.polyfit(np.log(areas2)[:12], heightsDivArea[:12], deg = 1, cov = True)
        (m2,c2), V2 = np.polyfit(np.log(areas2)[:16], heights2DivArea[:16], deg = 1, cov = True)
        #A, N = np.exp(a), b
        CError = np.sqrt(V[1][1])
        MError = np.sqrt(V[0][0])
        print("m1: {} +/- {}".format(m, MError))
        print("c1: {} +/- {}".format(c, CError))
        C2Error = np.sqrt(V[1][1])
        M2Error = np.sqrt(V[0][0])
        print("m2: {} +/- {}".format(m2, M2Error))
        print("c2: {} +/- {}".format(c2, C2Error))
        
        spacing = np.logspace(.5,5)
        fittedWidth = m * np.log(spacing)  + c
        fittedWidth2 = m2 * np.log(spacing)  + c2
        ax.plot(np.log(spacing), fittedWidth, label = "Max of LOBF")
        ax.plot(np.log(spacing), fittedWidth2, label = "Max of Data")
        
        ax.plot(np.log(areas2), heightsDivArea , "+", label = "Max from LOBF")
        ax.plot(np.log(areas2), heights2DivArea , "+", label = "Max from Data")
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("Number Of Spins log(N)")
        ax.set_ylabel(r"Max Heat capacity C / $C_0 N$")
        ax.set_title("Plot of Max Heat Capacity per Spin against log(N)")
        fig.savefig("5-FiniteSizeScaling\HeightAgainstAreaPlot2.png", dpi = 300)
        
        ##Estimating TcInfty
        fig, ax = plt.subplots()
        def f4(x,TcInfty, A, N):
            return  TcInfty + A * (x) ** N
        Tcs = Tcs + 2.269185314
        p0 = (2.269185314, .2, -.28)
        popt, pcov = scipy.optimize.curve_fit(f4, areas,Tcs, p0 = p0, maxfev=10000, absolute_sigma = False)
        print(p0[0], popt[0])
        ax.plot(areas, Tcs - popt[0], "+")
        spacing = np.logspace(.5, 3)
        ax.plot(spacing, f4(spacing, *popt) - popt[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Area of Lattice")
        ax.set_ylabel("Temperature $T_c - T_c(optimal)$ / $T_0$")
        ax.set_title("Plot of $T_c$ against Area with an 'optimal' $T_c(\infty)$ of {:0.2f}".format(popt[0]))
        fig.savefig("5-FiniteSizeScaling\TcAgainstAreaUnknownTcInftyPlot.png", dpi = 300)
        print("Values", popt)
        print("errors", np.sqrt(np.diag(pcov)))
    
class HysteresisAnalyser():
    
    def plotLoopAreaAgainstTemperature():
        fig, ax = plt.subplots()
        temperatures = np.logspace(.5,4.5,num=80)
        dims = (300,300)
        averages = []
        errors = []
        for T in temperatures:
            areas = np.load("6-Hysteresis/AreaDataSize{}Temp{:.3f}.npy".format(dims[0], T))
            print(areas)
            averages.append(np.average(areas))
            errors.append(np.std(areas) / np.sqrt(len(areas) - 1))
        #print(temperatures)
        #print(averages)
        logErrors = np.divide(errors, averages)
        (b,a), V = np.polyfit(np.log(temperatures[30:-27]), np.log(averages[30:-27]),deg = 1, cov = True, )
        A, N = np.exp(a), b
        AError = np.sqrt(V[1][1]) * A
        NError = np.sqrt(V[0][0])
        print("a: {} +/- {}".format(A, AError))
        print("N: {} +/- {}".format(N, NError))
        spacing = np.logspace(.5,4.5)
        fittedAverage = A * spacing ** N
        ax.plot(spacing, fittedAverage)

        ax.errorbar(temperatures, averages, yerr = errors, fmt = "+")
        ax.set_title("Plot of Hysteresis Area against Temperature")
        ax.set_ylabel("Area of hysteresis")
        ax.set_xlabel("Temperature")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig("6-Hysteresis\HysteresisAreaAgainstTempPlot2.png", dpi = 300)
        
        fig, ax = plt.subplots()
        ax.errorbar(temperatures, averages, yerr = errors, fmt = "+")
        ax.set_title("Plot of Hysteresis Area against Temperature")
        ax.set_ylabel("Area of hysteresis")
        ax.set_xlabel("Temperature")
        fig.savefig("6-Hysteresis\HysteresisAreaAgainstTempPlot.png", dpi = 300)
        
    def collectLoopAreaAgainstTemperature():
        numberOfPeriods = 60
        temperatures = np.logspace(.5,4.5,num=80)[40:]
        dims = (300,300)
        amplitude = 10
        numberOfSteps = 20000
        stateType = RandomNearestNeighbourMagnetisedForgetfulIsingState2D
        areas = []
        
        for T in temperatures:
            areas = list(np.load("6-Hysteresis/AreaDataSize{}Temp{:.3f}.npy".format(dims[0], T)))
            #areas = []
            sim = IsingSimulation.generateOscillatingSim(stateType, temperature = T, dimensions = dims, amplitude = amplitude, numberOfSteps = numberOfSteps, resolution = 10000)
            fields    = np.array([amplitude  * np.cos(x * 2 * np.pi / numberOfSteps) for x in range(numberOfSteps)])
            possibleH = np.linspace(-amplitude, amplitude, num = 10000)
        
            discretisedfields = possibleH[abs(fields[None, :] - possibleH[:, None]).argmin(axis=0)]
            areas.append(sim.getAreaOfHysteresis(period = numberOfSteps, numberOfPeriods = numberOfPeriods, fields = discretisedfields))
            print("Done")
            np.save("6-Hysteresis/AreaDataSize{}Temp{:.3f}.npy".format(dims[0], T), areas)
        
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
        
    def collectStableMagnetisations():
        temps = np.linspace(2.61, 2.81, num = 20)
        dims = [(100,100)]
        possibleFields = np.linspace(-0.2,0.2, 500)
        timeAtEachField = 10
        fields = np.repeat(possibleFields, timeAtEachField)
        stateType = VectorisedRandomNearestNeighbourMagnetisedIsingState2D
        for D in dims:
            for T in temps:
                field = []
                magnetisations = []
                sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, dimensions = D, fields = fields, bias = -0.5)
                magnetisations += list(sim.getMagnetisations(numberOfStates = len(fields), distanceBetweenStates = sim.getNumberOfSpins()))
                field += list(fields)
                print("done", T, D)
                np.save('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}2F.npy'.format(T, D[0]), [field, magnetisations])
                
                field = []
                magnetisations = []
                sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, dimensions = D, fields = fields[::-1], bias = 0.5)
                magnetisations += list(sim.getMagnetisations(numberOfStates = len(fields), distanceBetweenStates = sim.getNumberOfSpins()))
                field += list(fields[::-1])
                print("done", T, D)
                np.save('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}2B.npy'.format(T, D[0]), [field, magnetisations])


    def plotStableMagnetisations():
        fig, ax = plt.subplots()
        dims = [50,50] #, [25,25],[10,10]]
        temps = np.linspace(0.5, 2, num = 4)
        fmtDict = {0.5: '#1f77b4',
                   1: '#ff7f0e',
                   1.5: '#2ca02c',
                   2: '#d62728',}
        for T in temps:
            print(T)
            data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}F.npy'.format(T, dims[0]))
            field = data[0]
            magnetisations = data[1]
            magnetisations = np.average(np.split(magnetisations, 1000), axis = 1)
            field = field[::20]

            ax.plot(field, magnetisations, fmtDict[T], label = "{} $T_0$".format(T))
            
            data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}B.npy'.format(T, dims[0]))
            field = data[0]
            magnetisations = data[1]

            ax.plot(field, magnetisations, fmtDict[T])
        ax.legend()
        ax.set_title("Plot of Magnetisation with a varying H Field")
        ax.set_ylabel("Magnetisation M / $M_0$")
        ax.set_xlabel("Field H / $H_0$")
        fig.savefig("6-Hysteresis\MetaStableMagnetisationPlot.png", dpi = 300)
        
        fig, ax = plt.subplots()
        dims = [10,10] #, [25,25],[10,10]]
        temps = np.linspace(0.25, 2.25, num = 9)
        criticalHF = []
        criticalHB = []
        for T in temps:
            print(T)
            data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}F.npy'.format(T, dims[0]))
            field = data[0]
            magnetisations = data[1]
            hC = field[np.argmax(magnetisations>0)]
            #ax.plot(field, magnetisations)
            criticalHF.append(hC)
            
            data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}B.npy'.format(T, dims[0]))
            field = data[0]
            magnetisations = data[1]
            hC = field[np.argmax(magnetisations<0)]
            #ax.plot(field, magnetisations)
            criticalHB.append(hC)
        ax.plot(temps, criticalHF, "+")
        ax.plot(temps, criticalHB, "+")
        ax.set_xscale("log")
        ax.set_yscale("symlog")
        
        fig, ax = plt.subplots()
        dims = [[100,100], [50,50],]#[10,10]]
        temps = np.linspace(0.25, 2.25, num = 9)
        fmtDict = {100: '#1f77b4',
                   50: '#ff7f0e',}
        for D in dims:
            criticalHF = []
            criticalHB = []
            for T in temps:
                print(T)
                data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}F.npy'.format(T, D[0]))
                field = data[0]
                magnetisations = data[1]
                hC = field[np.argmax(magnetisations>0)]
                #ax.plot(field, magnetisations)
                criticalHF.append(hC)
                
                data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}B.npy'.format(T, D[0]))
                field = data[0]
                magnetisations = data[1]
                
                hC = field[np.argmax(magnetisations<0)]
                #ax.plot(field, magnetisations)
                criticalHB.append(hC)
            tps =  temps #2.269185314 -
            ax.plot(tps, criticalHF, color = fmtDict[D[0]], linestyle = "None", marker = "+", label = D)
            ax.plot(tps, criticalHB, color = fmtDict[D[0]], linestyle = "None", marker = "+")
            """
            (bF,aF), VF = np.polyfit(np.log(temps), np.log(criticalHF), deg = 1, cov = True)
            AF, NF = np.exp(aF), bF
            AFError = np.sqrt(VF[1][1]) * AF
            NFError = np.sqrt(VF[0][0])
            print("a: {} +/- {}".format(AF, AFError))
            print("N: {} +/- {}".format(NF, NFError))
            spacing = np.logspace(.5,3)
            
            (bB,aB), VB = np.polyfit(np.log(temps), np.log(criticalHF), deg = 1, cov = True)
            AB, NB = np.exp(aB), bB
            ABError = np.sqrt(VB[1][1]) * AF
            NBError = np.sqrt(VB[0][0])
            print("a: {} +/- {}".format(AB, ABError))
            print("N: {} +/- {}".format(NB, NBError))
            spacing = np.logspace(0,1)
            
            fittedTcsF = AF * spacing ** NF
            #ax.plot(spacing, fittedTcsF)
            fittedTcsB = AB * spacing ** NB
            #ax.plot(spacing, fittedTcsB)
            
            (mF,cF), VF = np.polyfit(temps[6:], criticalHF[6:], deg = 1, cov = True)
            MFError = np.sqrt(VF[1][1])
            CFError = np.sqrt(VF[0][0])
            print("a: {} +/- {}".format(mF, MFError))
            print("N: {} +/- {}".format(cF, CFError))
            spacing = np.logspace(.5,3)
            print(-cF / mF)
            
            (bB,aB), VB = np.polyfit(np.log(temps), np.log(criticalHF), deg = 1, cov = True)
            AB, NB = np.exp(aB), bB
            ABError = np.sqrt(VB[1][1]) * AF
            NBError = np.sqrt(VB[0][0])
            print("a: {} +/- {}".format(AB, ABError))
            print("N: {} +/- {}".format(NB, NBError))
            spacing = np.logspace(0,1)
            
            fittedTcsF = mF * spacing + cF
            ax.plot(spacing, fittedTcsF)
            fittedTcsB = AB * spacing ** NB
            #ax.plot(spacing, fittedTcsB)
            """
        #ax.set_xscale("log")
        #ax.set_yscale("symlog")
        ax.legend()
        ax.set_title("Plot of maximum opposing H against Temperature")
        ax.set_xlabel("Temperature T / $T_0$")
        ax.set_ylabel("Opposing Field H / $H_0$")
        fig.savefig("6-Hysteresis\LargestStableMagnetisationPlot.png", dpi = 300)
        
        fig, ax = plt.subplots()
        dims = [[100,100], [50,50],]#[10,10]]
        temps = np.linspace(2.20, 2.26, num = 12)
        for D in dims:
            criticalHF = []
            criticalHB = []
            for T in temps:
                print(T)
                data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}2F.npy'.format(T, D[0]))
                field = data[0]
                magnetisations = data[1]
                hC = field[np.argmax(magnetisations>0)]
                #ax.plot(field, magnetisations)
                criticalHF.append(hC)
                
                data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}2B.npy'.format(T, D[0]))
                field = data[0]
                magnetisations = data[1]
                
                hC = field[np.argmax(magnetisations<0)]
                #ax.plot(field, magnetisations)
                criticalHB.append(hC)
                
            (mF,cF), VF = np.polyfit(temps, criticalHF, deg = 1, cov = True)
            MFError = np.sqrt(VF[1][1])
            CFError = np.sqrt(VF[0][0])
            print("a: {} +/- {}".format(mF, MFError))
            print("N: {} +/- {}".format(cF, CFError))
            spacing = np.linspace(2.2,2.3)
            print(-cF / mF)
            fittedTcsF = mF * spacing + cF
            ax.plot(spacing, fittedTcsF)
                
            ax.plot(temps, criticalHF, "+")
            ax.plot(temps, criticalHB, "+")
        #ax.set_xscale("log")
        #ax.set_yscale("symlog")
        
        fig, ax = plt.subplots()
        dims = [[100,100],]#[10,10]]
        temps = np.concatenate((np.linspace(2.20, 2.26, num = 12), np.linspace(2.26, 2.38, num = 12), np.linspace(2.4, 2.6, num = 10), np.linspace(2.51, 2.61, num = 10)[:8], np.linspace(2.61, 2.81, num = 20)[:16]))
        for D in dims:
            criticalHF = []
            criticalHB = []
            for T in temps:
                print(T)
                data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}2F.npy'.format(T, D[0]))
                field = data[0]
                magnetisations = data[1]
                hC = field[np.argmax(magnetisations>0)]
                #ax.plot(field, magnetisations)
                criticalHF.append(hC)
                
                data = np.load('6-Hysteresis\MagnetisationAgainstFieldTemp{:.3f}Data{}2B.npy'.format(T, D[0]))
                field = data[0]
                magnetisations = data[1]
                
                hC = field[np.argmax(magnetisations<0)]
                #ax.plot(field, magnetisations)
                criticalHB.append(hC)
                
            ax.set_title("Plot of maximum opposing H against Temperature")
            ax.set_ylabel("Maximum meta-stable H H / $H_0$")
            ax.set_xlabel("Temperature $T$ / $T_0$")
            ax.plot(temps, criticalHF, "+", label = "From Negative")
            ax.plot(temps, criticalHB, "+", label = "From Positive")
            ax.legend() 
            fig.savefig("6-Hysteresis\StabilityAroundTc.png", dpi = 300)
            
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
            averageOver = 3600
            magnetisations = sim.getMagnetisations(runTime + averageOver)
            accumilative = np.cumsum(magnetisations) 
            timeAverage = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
            divMagnetisations = np.gradient(timeAverage)
            initialM = 2 * b
            ax.plot(magnetisations, divMagnetisations, label = "{:.3f}".format(initialM))
        ax.legend(loc = "upper right")
        ax.set_title("Plot of Magnetisation against dM/dt for different initial M")
        ax.set_xlabel("Magnetisation $M$ / $M_0$")
        ax.set_ylabel("dM/dT / $M_0$")
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
            averageOver = 3600
            magnetisations = sim.getMagnetisations(runTime + averageOver)
            accumilative = np.cumsum(magnetisations) 
            timeAverage = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
            divMagnetisations = np.gradient(timeAverage)
            initialM = 2 * b
            ax.plot(magnetisations, divMagnetisations, label = "{:.3f}".format(initialM))
        ax.legend(loc = "lower left")
        ax.set_title("Plot of Magnetisation against dM/dt for different initial M")
        ax.set_xlabel("Magnetisation $M$ / $M_0$")
        ax.set_ylabel("dM/dT / $M_0$")
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
        positions = [0,2,4]
        stateType = VectorisedRandomNearestNeighbourIsingState2D
        startBias = 0.2
        
        simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
        
        fig, axs = plt.subplots(1,3, figsize = (12,4.3))
        axs = np.array(axs).flatten()
        for i in range(3):
            axs[i].imshow(simulation.getStateNumber(positions[i] * simulation.getNumberOfSpins()).getStateAsArray(), aspect='equal', cmap=plt.cm.gray, interpolation='nearest', norm=plt.Normalize(0, 1))
            axs[i].set_title("After {:.1f} steps".format(positions[i]))
            
        fig.suptitle("Image of the ising state at a temperature of 0.1")
        fig.tight_layout(rect=[0, 0.0, 1, 0.94])
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
        numberOfSplits = [1,2,4,400,40000]
        fig, axs = plt.subplots(1,5, figsize = (12,3.5))
        axs = np.array(axs).flatten()
        for i in range(5):
            sim = PlottableIsingSimulatuion.generateSim(stateType = stateType, temperature = temp, dimensions = dims, numberOfSplits = numberOfSplits[i])
            averageTransform = sim.getAverageFourierTransform(numberOfStates, distanceBetweenStates)
            #averageTransform = np.absolute(sim.getStateNumber(40000).getStateAsDFT())
            axs[i].imshow(averageTransform, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            axs[i].set_title("{:} splits".format(numberOfSplits[i]))
            
        fig.suptitle("Image of the IsingState as a Fourier Transform for different number of splits")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\OrderInFastSystematicPlot2.png")
        
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
        
    def collectOrderedTransformData():
        temp = .01
        dims = (400,400)
        #positions = [0,1,2,4,8,16,32]
        positions = range(1,401) #range(0,41)
        stateType = VectorisedRandomNearestNeighbourIsingState2D
        startBias = 0.05
        
        for x in range(20):
            simulation = PlottableIsingSimulatuion.generateSim(stateType, bias = startBias, temperature = temp, dimensions = dims)
            for pos in positions:
                fts = list(np.load("8-GlobalOrder\OrderedTransformPos{}Bias{}Data{}.npy".format(pos, startBias, dims[0])))
                fourierTransform = simulation.getStateNumber(pos * simulation.getNumberOfSpins()).getStateAsDFT(removeCentre = False)
                fts.append(fourierTransform)
                np.save("8-GlobalOrder\OrderedTransformPos{}Bias{}Data{}.npy".format(pos, startBias, dims[0]), fts)
            print("done")
        
    def plotOrderedTransformData():
        dims = (200,200)
        positions = [0,1,4]
        fig, axs = plt.subplots(1,3, figsize = (12,5))
        axs = np.array(axs).flatten()
        for i in range(len(positions)):
            data = np.load("8-GlobalOrder\OrderedTransformPos{}Data{}.npy".format(positions[i], dims[0]))
            axs[i].imshow(np.absolute(data[0]), aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            axs[i].set_title("After {:.1f} steps".format(positions[i]))
            
        fig.suptitle("Image of the Ising State as a Fourier Transform\nat a temperature of 0.1")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\OrderedTransformPlot2.png")
        
        def f(x, A, B, tendsTo = 0):
            return A * np.exp(-B*x) + tendsTo
        
        positions = [0,1,2,3,4,5]
        fig, axs = plt.subplots(2,3, figsize = (8,12))
        axs = np.array(axs).flatten()
        for i in range(len(positions)):
            data = np.load("8-GlobalOrder\OrderedTransformPos{}Data{}2.npy".format(positions[i], dims[0]))
            a = [np.absolute(x) for x in data]
            aveargeTransform = np.average(a, axis = 0)
            
            sliceTransform = aveargeTransform[:,100] + aveargeTransform[100,:] + aveargeTransform[::-1,100] + aveargeTransform[100,::-1]
            sliceTransform = sliceTransform[101:]
            distances = range(1, 100)
            
            p0 = (sliceTransform[0],1)
            popt, _ = scipy.optimize.curve_fit(f, distances, sliceTransform, p0 = p0, maxfev=10000)
            
            
            axs[i].plot(sliceTransform)
            axs[i].plot(f(distances, *popt))
            print(np.absolute(data[0]))
            #axs[i].imshow(aveargeTransform, aspect='equal', cmap=plt.cm.gray, interpolation='nearest')
            axs[i].set_title("After {:.1f} steps".format(positions[i]))
            
        fig.suptitle("Slice of the Fourier Transform")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\WidthOfFourierTransform1.png")
        
        dims = [400,400]
        positions = [1,2,32,50, 100]
        fig, ax = plt.subplots()
        fig2, axs = plt.subplots(1,5, figsize = (12, 5))
        axs = np.array(axs).flatten()
        for i in range(len(positions)):
            data = np.load("8-GlobalOrder\OrderedTransformPos{}Data{}.npy".format(positions[i], dims[0]))
            a = [np.absolute(x) for x in data]
            aveargeTransform = np.average(a, axis = 0)
            
            sliceTransform = aveargeTransform[:,200] + aveargeTransform[200,:] + aveargeTransform[::-1,200] + aveargeTransform[200,::-1]
            sliceTransform = sliceTransform[201:]
            distances = range(1, 200)
            tendsTo = sliceTransform[-1]
            
            p0 = (sliceTransform[0],1)
            popt, _ = scipy.optimize.curve_fit(lambda x, A, B: f(x, A, B, tendsTo = tendsTo), distances, sliceTransform, p0 = p0, maxfev=10000)
            
            axs[i].plot(sliceTransform)
            axs[i].plot(f(distances, *popt, tendsTo = tendsTo))
            axs[i].set_title("After {:.1f} steps".format(positions[i]))
            
        fig2.suptitle("Slice of the Fourier Transform")
        fig2.tight_layout(rect=[0, 0.0, 1, 0.96])
        fig2.savefig("8-GlobalOrder\WidthOfFourierTransform2.png")
            
        dims = [400,400]
        positions = np.arange(1,41)
        axs = np.array(axs).flatten()
        fallTime = []
        for i in range(len(positions)):
            data = np.load("8-GlobalOrder\OrderedTransformPos{}Data{}.npy".format(positions[i], dims[0]))
            a = [np.absolute(x) for x in data]
            aveargeTransform = np.average(a, axis = 0)
            
            sliceTransform = aveargeTransform[:,200] + aveargeTransform[200,:] + aveargeTransform[::-1,200] + aveargeTransform[200,::-1]
            sliceTransform = sliceTransform[201:]
            distances = range(1, 200)
            tendsTo = sliceTransform[-1]
            
            p0 = (sliceTransform[0],1, tendsTo)
            popt, _ = scipy.optimize.curve_fit(lambda x, A, B, C: f(x, A, B, tendsTo = C), distances, sliceTransform, p0 = p0, maxfev=10000)
            
            #axs[i].plot(sliceTransform)
            #axs[i].plot(f(distances, *popt, tendsTo = tendsTo))
            fallTime.append(1/popt[1])

        ax.plot(positions, fallTime, "+", label = r"0.01 $T_0$")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Plot of Characteristic Frequency against Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic Frequency of Fourier Transform")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\WidthOfOrderedTransform.png", dpi = 300)
        
        fig, ax = plt.subplots()
        def f4(x,FallTimeInfty, A, N):
            return  FallTimeInfty + A * (x) ** N
        p0 = (min(fallTime), 1, -1)
        popt, pcov = scipy.optimize.curve_fit(f4, positions[:25],fallTime[:25], p0 = p0, maxfev=10000, absolute_sigma = False)
        print(p0[0], popt[0])
        ax.plot(positions, fallTime - popt[0], "+")
        spacing = np.logspace(0, 2)
        ax.plot(spacing, f4(spacing, *popt) - popt[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic frequency $f - f(\infty)$")
        ax.set_title("Plot of Characteristic Frequency against Time")
        fig.savefig("8-GlobalOrder\WidthOfOrderedTransform2.png", dpi = 300)
        print("Values", popt)
        print("errors", np.sqrt(np.diag(pcov)))
        
        fig, ax = plt.subplots()
        dims = [400,400]
        bias = 0.1
        positions = np.arange(1,201)
        axs = np.array(axs).flatten()
        fallTime = []
        for i in range(len(positions)):
            data = np.load("8-GlobalOrder\OrderedTransformPos{}Bias{}Data{}.npy".format(positions[i], bias, dims[0]))
            a = [np.absolute(x) for x in data]
            aveargeTransform = np.average(a, axis = 0)
            
            sliceTransform = aveargeTransform[:,200] + aveargeTransform[200,:] + aveargeTransform[::-1,200] + aveargeTransform[200,::-1]
            sliceTransform = sliceTransform[201:]
            distances = range(1, 200)
            tendsTo = sliceTransform[-1]
            
            p0 = (sliceTransform[0],1, tendsTo)
            popt, _ = scipy.optimize.curve_fit(lambda x, A, B, C: f(x, A, B, tendsTo = C), distances, sliceTransform, p0 = p0, maxfev=10000)
            
            #axs[i].plot(sliceTransform)
            #axs[i].plot(f(distances, *popt, tendsTo = tendsTo))
            fallTime.append(1/popt[1])

        ax.plot(positions, fallTime, "+", label = r"0.01 $T_0$")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Plot of Characteristic Frequency against Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic Frequency of Fourier Transform")
        fig.tight_layout(rect=[0, 0.0, 1, 0.95])
        fig.savefig("8-GlobalOrder\WidthOfOrderedTransform3.png", dpi = 300)
        
        fig, ax = plt.subplots()
        def f4(x,FallTimeInfty, A, N):
            return  FallTimeInfty + A * (x) ** N
        p0 = (min(fallTime), 1, -1)
        popt, pcov = scipy.optimize.curve_fit(f4, positions[:80],fallTime[:80], p0 = p0, maxfev=10000, absolute_sigma = False)
        print(p0[0], popt[0])
        ax.plot(positions, fallTime - popt[0], "+")
        spacing = range(0,81)
        ax.plot(spacing, f4(spacing, *popt) - popt[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic frequency $f - f(\infty)$")
        ax.set_title("Plot of Characteristic Frequency against Time")
        fig.savefig("8-GlobalOrder\WidthOfOrderedTransform4.png", dpi = 300)
        print("Values", popt)
        print("errors", np.sqrt(np.diag(pcov)))
        
        
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

class SusceptibilityAnalyser():
    
    def plotSusceptibility():
        
        def theoreticalS(T):
            Z = 2 * np.exp(8 / T) + 12 + 2 * np.exp(-8 / T)
            M = (2 * 4 * np.exp(8 / T) + 8 * 2) / Z
            Msquared = (2 * 16 * np.exp(8 / T) + 8 * 4) / Z
            return (Msquared - M ** 2) / (T)
        
        fig, (ax1, ax2) = plt.subplots(2)
        dims = [(50,50)]
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}withBias.npy".format(d[0]))
            
            temps = list(data[0])
            positiveMags = data[1]
            negativeMags = data[2]
            ax1.plot(temps, positiveMags)
            ax1.plot(temps, negativeMags)
            
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            print(data)
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            
            ax2.plot(temps, averageSusceptibility)
            
        fig, (ax1, ax2) = plt.subplots(2, sharex = True)
        dims = [(50,50)]
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}withBias2.npy".format(d[0]))
            
            temps = list(data[0])
            positiveMags = data[1]
            negativeMags = data[2]
            ax1.plot(temps, positiveMags)
            ax1.plot(temps, negativeMags)
            
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            
            ax2.plot(temps, averageSusceptibility)
            
        ax1.set_title("Magnetisation against Temperature")
        ax1.set_ylabel("Magnetisation M / $M_0$")
        ax2.set_title("Susceptibility against Temperature")
        ax2.set_xlabel("Temperature T / $T_0$")
        ax2.set_ylabel("Susceptibility $\chi$ / $\chi_0$")
        fig.tight_layout()
        fig.savefig("9-Susceptibility\DemonstrationOfMethodPlot.png", dpi = 300)
        
        fig, (ax) = plt.subplots()
        dims = [(100,100)]
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]))
            temps = list(data[0])
            positiveMags = data[1]
            negativeMags = data[2]
            
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            #data = data[8:]
            data = np.concatenate([data[:8], data[9:]])
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            averageSusceptibility[averageSusceptibility < 0] = 0
            
            averageOver = 8
            accumilative = np.cumsum(averageSusceptibility) 
            timeAverage = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
            
            accumilative = np.cumsum(temps) 
            timeAverageT = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
            
            def f1(x, A, B, c, D):
                return A - B * (x - c) **2 + D * x
            p0 = None
            popt, pcov = scipy.optimize.curve_fit(f1, timeAverageT,timeAverage, p0 = p0, maxfev=10000, absolute_sigma = False)

            spacing = np.linspace(2.2, 2.4, 1000)
            ax.plot(spacing, f1(spacing, *popt))
            
            ax.plot(temps, averageSusceptibility)
            ax.plot(timeAverageT, timeAverage)
            print(temps[np.argmax(averageSusceptibility)])
        
        def f1(x, A, B, C):
            return A * x ** 2 / ((x-B) ** 2 + C ** 2)
            
        fig, (ax) = plt.subplots()
        dims = [(50,50),]
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]))
            temps = list(data[0])
            positiveMags = data[1]
            negativeMags = data[2]
            
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            errors = df.groupby(0, as_index = False).sem().values[:,1]
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            averageSusceptibility[averageSusceptibility < 0] = 0
            
            popt, pcov = scipy.optimize.curve_fit(f1, temps,averageSusceptibility, maxfev=10000, absolute_sigma = False)

            spacing = np.linspace(2, 3, 1000)
            #ax.plot(spacing, f1(spacing, *popt))
            #ax.plot(temperatures, theoreticalS(temperatures))
            ax.plot(temps, averageSusceptibility)
            print(temps[np.argmax(averageSusceptibility)], "T_c N ={}".format(d[0] ** 2))
        ax.set_title("Susceptibility against Temperature")
        ax.set_xlabel("Temperature T / $T_0$")
        ax.set_ylabel("Susceptibility $\chi$ / $\chi_0$")
        fig.savefig("9-Susceptibility\By50Plot.png", dpi = 300)
        
        fig, ax = plt.subplots()
        dims = [(100,100), (50,50),(30,30),(20,20), (10,10)]
        areas = []
        fittedTcs = []
        areas2 = []
        argmaxTcs = []
        argmaxTcs2 = []
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]))
            temps = list(data[0])
            positiveMags = data[1]
            negativeMags = data[2]
            
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            errors = df.groupby(0, as_index = False).sem().values[:,1]
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            averageSusceptibility[averageSusceptibility < 0] = 0
            
            averageOver = 8
            accumilative = np.cumsum(averageSusceptibility) 
            timeAverage = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
            
            accumilative = np.cumsum(temps) 
            timeAverageT = (accumilative[averageOver:] - accumilative[:-averageOver]) / averageOver
            
            
            popt, pcov = scipy.optimize.curve_fit(f1, timeAverageT - 2,timeAverage, maxfev=10000, absolute_sigma = False)

            spacing = np.linspace(2, 3, 1000)
            ax.plot(spacing, f1(spacing - 2, *popt))
            #ax.plot(temperatures, theoreticalS(temperatures))
            ax.plot(temps, averageSusceptibility)
            print(spacing[np.argmax(f1(spacing - 2, *popt))], "T_c N ={} Fitted".format(d[0] ** 2))
            print(temps[np.argmax(averageSusceptibility)], "T_c N ={} Argmax".format(d[0] ** 2))
            print(timeAverageT[np.argmax(timeAverage)], "T_c N ={} Argmax".format(d[0] ** 2))
            areas.append(d[0] ** 2)
            areas2.append(d[0] ** 2)
            fittedTcs.append(spacing[np.argmax(f1(spacing - 2, *popt))])
            argmaxTcs.append(temps[np.argmax(averageSusceptibility)])
            argmaxTcs2.append(timeAverageT[np.argmax(timeAverage)])
        
        ax.set_xlim([2,3])
        
        fig, ax = plt.subplots()
        ax.plot(areas, fittedTcs, "+", label = "fit")
        ax.plot(areas, argmaxTcs, "+", label = "argMax")
        ax.plot(areas, argmaxTcs2, "+", label = "smoothed Max")
        ax.legend()
        
        fig, (ax) = plt.subplots()
        dims = [(10,10), (5,5), (2,2),]
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]))
            positiveMags = data[1]
            negativeMags = data[2]
            temps = list(np.round(data[0], decimals = 5))
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            errors = df.groupby(0, as_index = False).sem().values[:,1]
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            averageSusceptibility[averageSusceptibility < 0] = 0
            
            ax.errorbar(temps, averageSusceptibility, yerr = errors, label = d)
        ax.legend()
        ax.set_title("Susceptibility against Temperature")
        ax.set_xlabel("Temperature T / $T_0$")
        ax.set_ylabel("Susceptibility $\chi$ / $\chi_0$")
        fig.savefig("9-Susceptibility\LowNPlot.png", dpi = 300)
            
        fig, (ax) = plt.subplots()
        dims = [(50,50), (30,30), (20,20),] #(5,5), (2,2)]
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]))
            temps = list(np.round(data[0], decimals = 5))
            positiveMags = data[1]
            negativeMags = data[2]
            
            susceptibility = list(positiveMags - negativeMags)
            df = pd.DataFrame(np.array([temps, susceptibility]).T)
            data = df.groupby(0,as_index=False).mean().values
            errors = df.groupby(0, as_index = False).sem().values[:,1]
            temps = data[:,0]
            averageSusceptibility = data[:,1]
            averageSusceptibility[averageSusceptibility < 0] = 0
            
            ax.errorbar(temps, averageSusceptibility, yerr = errors, label = d)
        ax.legend()
        ax.set_title("Susceptibility against Temperature")
        ax.set_xlabel("Temperature T / $T_0$")
        ax.set_ylabel("Susceptibility $\chi$ / $\chi_0$")
        fig.savefig("9-Susceptibility\HighNPlot.png", dpi = 300)

    def collectSusceptibility():
        timeAtEachTemperature = 1200
        ignore = 1000 #300
        temperatures = np.arange(2, 2.4, step = 0.01)#0.005
        dims = [(5,5), (5,5), (5,5), (5,5), (5,5), (5,5)] #[(50,50)] #[(5,5),(8,8),(10,10)] #[(50,50), (100,100), (150,150), (200,200)]
        #temperatures = np.arange(2.1, 2.4, step = 0.0125)
        #dims = [(30,30), (30,30)]
        stateType = VectorisedRandomNearestNeighbourIsingState2D
        for d in dims:
            data = np.load("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]))
            negativeMags = list(data[2])
            positiveMags = list(data[1])
            temps = list(data[0])
            
            #negativeMags = []
            #positiveMags = []
            #temps = []
            for T in temperatures:
                temps.append(T)
                
                sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, bias = -0.5, dimensions = d, field = 0.02)
                magnetisation = sim.getAverageMagnetisation(timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins(), statesToIgnore = ignore)
                positiveMags.append(magnetisation)
                
                sim = IsingSimulation.generateSim(stateType = stateType, temperature = T, bias = 0.5, dimensions = d, field = -0.02)
                magnetisation = sim.getAverageMagnetisation(timeAtEachTemperature, distanceBetweenStates = sim.getNumberOfSpins(), statesToIgnore = ignore)
                negativeMags.append(magnetisation)
                print("Done {}".format(T))
                np.save("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]), [temps, positiveMags, negativeMags])
            np.save("9-Susceptibility\MagnetisationAgainstTemperatureData-{}AroundTc.npy".format(d[0]), [temps, positiveMags, negativeMags])
                