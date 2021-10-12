#Exercise 3 template for CHE210D

#c:\mydir> f2py –c –m project2fortran project2fortran.f90 --fcompiler=gnu95 --compiler=mingw32


#import pdb writing
import atomicwrites as atomwrite

#import python modules
import numpy as np
import time

#import compiled fortran library
import projectfortran as exlib

import sys
import os
import matplotlib.pyplot as plt

#========== GLOBAL VARIABLES ==========
#distance cutoff for pairwise interactions
Cut = 2.5
N = 240
rho = 0.80

L = (N / rho)**(1./3.)

#whether or not to save the trajectory as a PdbFile
SaveTrajectory = True

#set the random number seed; useful for debugging
#np.random.seed(342324)


#NOTE:
#everything below assumes unit atomic masses,
#such that forces = accelerations.



def LineSearch(Pos, Charge, Dir, dx, EFracTol,M,Lam0, epsilon,k_sigma,
               Accel = 1.5, MaxInc = 10., MaxIter = 10000):
    """Performs a line search along direction Dir.
Input:
    Pos: starting positions, (N,3) array
    Dir: (N,3) array of gradient direction
    dx: initial step amount
    EFracTol: fractional energy tolerance
    Accel: acceleration factor
    MaxInc: the maximum increase in energy for bracketing
    MaxIter: maximum number of iteration steps
Output:
    PEnergy: value of potential energy at minimum along Dir
    Pos: minimum energy (N,3) position array along Dir
"""
    #identify global variables
    global L, Cut
    
    #start the iteration counter
    Iter = 0

    #find the normalized direction    
    NormDir = np.clip(Dir, -1.e100, 1.e100)
    NormDir = NormDir / np.sqrt(np.sum(NormDir * NormDir))

    #take the first two steps and compute energies    
    Dists = [0., dx]
    PEs = [exlib.calcenergy(Pos + NormDir * x,Charge, M, L, Cut,Lam0, epsilon,k_sigma) for x in Dists]
    
    #if the second point is not downhill in energy, back
    #off and take a shorter step until we find one
    while PEs[1] > PEs[0]:
        Iter += 1
        dx = dx * 0.5
        Dists[1] = dx
        PEs[1] = exlib.calcenergy(Pos + NormDir * dx,Charge, M, L, Cut,Lam0, epsilon,k_sigma)
        
    #find a third point
    Dists = Dists + [2. * dx]
    PEs = PEs + [exlib.calcenergy(Pos + NormDir * 2. * dx,Charge, M, L, Cut,Lam0, epsilon,k_sigma)]
    
    #keep stepping forward until the third point is higher
    #in energy; then we have bracketed a minimum
    while PEs[2] < PEs[1]:
        Iter += 1
            
        #find a fourth point and evaluate energy
        Dists = Dists + [Dists[-1] + dx]
        PEs = PEs + [exlib.calcenergy(Pos + NormDir * Dists[-1],Charge, M, L, Cut,Lam0, epsilon,k_sigma)]

        #check if we increased too much in energy; if so, back off
        if (PEs[3] - PEs[0]) > MaxInc * (PEs[0] - PEs[2]):
            PEs = PEs[:3]
            Dists = Dists[:3]
            dx = dx * 0.5
        else:
            #shift all of the points over
            PEs = PEs[-3:]
            Dists = Dists[-3:]
            dx = dx * Accel
            
    #we've bracketed a minimum; now we want to find it to high
    #accuracy
    OldPE3 = 1.e300
    while True:
        Iter += 1
        if Iter > MaxIter:
            print("Warning: maximum number of iterations reached in line search.")
            break
            
        #store distances for ease of code-reading
        d0, d1, d2 = Dists
        PE0, PE1, PE2 = PEs

        #use a parobolic approximation to estimate the location
        #of the minimum
        d10 = d0 - d1
        d12 = d2 - d1
        Num = d12*d12*(PE0-PE1) - d10*d10*(PE2-PE1)
        Dem = d12*(PE0-PE1) - d10*(PE2-PE1)
        if Dem == 0:
            #parabolic extrapolation won't work; set new dist = 0 
            d3 = 0
        else:
            #location of parabolic minimum
            d3 = d1 + 0.5 * Num / Dem
            
        #compute the new potential energy
        PE3 = exlib.calcenergy(Pos + NormDir * d3,Charge, M, L, Cut,Lam0, epsilon,k_sigma)
        
        #sometimes the parabolic approximation can fail;
        #check if d3 is out of range < d0 or > d2 or the new energy is higher
        if d3 < d0 or d3 > d2 or PE3 > PE0 or PE3 > PE1 or PE3 > PE2:
            #instead, just compute the new distance by bisecting two
            #of the existing points along the line search
            if abs(d2 - d1) > abs(d0 - d1):
                d3 = 0.5 * (d2 + d1)
            else:
                d3 = 0.5 * (d0 + d1)
            PE3 = exlib.calcenergy(Pos + NormDir * d3,Charge, M, L, Cut,Lam0, epsilon,k_sigma)
            
        #decide which three points to keep; we want to keep
        #the three that are closest to the minimum
        if d3 < d1:
            if PE3 < PE1:
                #get rid of point 2
                Dists, PEs = [d0, d3, d1], [PE0, PE3, PE1]
            else:
                #get rid of point 0
                Dists, PEs = [d3, d1, d2], [PE3, PE1, PE2]
        else:
            if PE3 < PE1:
                #get rid of point 0
                Dists, PEs = [d1, d3, d2], [PE1, PE3, PE2]
            else:
                #get rid of point 2
                Dists, PEs = [d0, d1, d3], [PE0, PE1, PE3]
                
        #check how much we've changed
        if abs(OldPE3 - PE3) < EFracTol * abs(PE3):
            #the fractional change is less than the tolerance,
            #so we are done and can exit the loop
            break
        OldPE3 = PE3

    #return the position array at the minimum (point 1)        
    PosMin = Pos + NormDir * Dists[1]
    PEMin = PEs[1]
        
    return PEMin, PosMin

        
def ConjugateGradient(Pos,Charge, dx, M,Lam0, epsilon,k_sigma, EFracTolLS, EFracTolCG):
    """Performs a conjugate gradient search.
Input:
    Pos: starting positions, (N,3) array
    dx: initial step amount
    EFracTolLS: fractional energy tolerance for line search
    EFracTolCG: fractional energy tolerance for conjugate gradient
Output:
    PEnergy: value of potential energy at minimum
    Pos: minimum energy (N,3) position array 
"""
    #identify global variables
    global  L, Cut
    #initial search direction
    Forces = np.zeros_like(Pos)
    PE, Forces = exlib.calcenergyforces(Pos,Charge, M, L, Cut,Lam0, epsilon,k_sigma, Forces)
    Dir = Forces
    OldPE = 1.e300
    #iterative line searches
    while abs(PE - OldPE) > EFracTolCG * abs(PE):
        OldPE = PE
        PE, Pos = LineSearch(Pos,Charge, Dir, dx, EFracTolLS, M,Lam0, epsilon,k_sigma)
        OldForces = Forces.copy()
        PE, Forces = exlib.calcenergyforces(Pos,Charge, M, L, Cut,Lam0, epsilon,k_sigma, Forces)
        Gamma = np.sum((Forces - OldForces) * Forces) / np.sum(OldForces * OldForces)
        Dir = Forces + Gamma *  Dir
    return PE, Pos


def InitPositions(N, L):
    """Returns an array of initial positions of each atom,
placed on a cubic lattice for convenience.
Input:
    N: number of atoms
    L: box length
Output:
    Pos: (N,3) array of positions
"""
    #make the position array
    Pos = np.zeros((N,3), float)
    #compute integer grid # of locations for cubic lattice
    NLat = int(N**(1./3.) + 1.)
    LatSpac = L / NLat
    #make an array of lattice sites
    r = LatSpac * np.arange(NLat, dtype=float) - 0.5*L
    #loop through x, y, z positions in lattice until done
    #for every atom in the system
    i = 0
    for x in r:
        for y in r:
            for z in r:
                Pos[i] = np.array([x,y,z], float)
                #add a random offset to help initial minimization
                Offset = 0.1 * LatSpac * (np.random.rand(3) - 0.5)
                Pos[i] = Pos[i] + Offset
                i += 1
                #if done placing atoms, return
                if i >= N:
                    return Pos
    return Pos


def RescaleVelocities(Vel, T):
    """Rescales velocities in the system to the target temperature.
Input:
    Vel: (N,3) array of atomic velocities
    T: target temperature
Output:
    Vel: same as above
"""
    #recenter to zero net momentum (assuming all masses same)
    Vel = Vel - Vel.mean(axis=0)
    #find the total kinetic energy
    KE = 0.5 * np.sum(Vel * Vel)
    #find velocity scale factor from ratios of kinetic energy
    VScale = np.sqrt(1.5 * len(Vel) * T / KE)
    Vel = Vel * VScale
    return Vel  


def InitVelocities(N, T):
    """Returns an initial random velocity set.
Input:
    N: number of atoms
    T: target temperature
Output:
    Vel: (N,3) array of atomic velocities
"""
    Vel = np.random.rand(N, 3)
    Vel = RescaleVelocities(Vel, T)
    return Vel


def InitAccel(Pos,Charge,M,Lam0, epsilon,k_sigma):
    """Returns the initial acceleration array.
Input:
    Pos: (N,3) array of atomic positions
Output:
    Accel: (N,3) array of acceleration vectors
"""
    #global variables
    global L, Cut
    #get the acceleration from the forces
    Forces = np.zeros_like(Pos)
    PEnergy, Accel = exlib.calcenergyforces(Pos,Charge, M, L, Cut,Lam0, epsilon,k_sigma, Forces)
    return Accel

def InitCharge(N,Config):
    
    if Config==0:
        Charge=N*[0]
    elif Config==1:
        Charge=(N/2)*[1]+(N/2)*[-1]
    elif Config==2:
        repeat=[1,0]
        ChargePos=np.array(repeat*(N/len(repeat)),float)
        OppositeCharge=np.array(((N/2)*[1]+(N/2)*[-1]),float)
        Charge=ChargePos*OppositeCharge
    elif Config==3:
        repeat=[1,0,0]
        ChargePos=np.array(repeat*(N/len(repeat)),float)
        OppositeCharge=np.array(((N/2)*[1]+(N/2)*[-1]),float)
        Charge=ChargePos*OppositeCharge    
    elif Config==4:
        repeat=[1,0,0,0]
        ChargePos=np.array(repeat*(N/len(repeat)),float)
        OppositeCharge=np.array(((N/2)*[1]+(N/2)*[-1]),float)
        Charge=ChargePos*OppositeCharge
    return Charge


def InstTemp(Vel):
    """Returns the instantaneous temperature.
Input:
    Vel: (N,3) array of atomic velocities
Output:
    Tinst: float
"""
    return np.sum(Vel * Vel) / (3. * len(Vel))




    
def SingleRun(k_sigma=0):
    #set the init box width, number of particles, temperature, and timestep
    SaveTrajectory=True 
    N = 240
    M=16
    rho = 0.80
    L = (N / rho)**(1./3.)
    

    print("L="+str(L))
    Temp = 1.0
    dt = 0.001
    
    k_sigma=0
    Lam0=L
    epsilon=1
    
    #set the frequency in seconds to update the display    
    DisplayFreq = 0.5
    #set the frequency in md steps to rescale velocities
    RescaleSteps = 10
    #set the frequency in md steps to write coordinates
    WriteSteps = 2

    #set the max number of md steps; 0 for infinite loop
    MaxSteps = WriteSteps*10000



    #get the initial positions, velocities, and acceleration (forces)    
    Charge=InitCharge(N,1)
    Pos = InitPositions(N, L)
    Vel = InitVelocities(N, Temp)
    Accel = InitAccel(Pos,Charge,M,Lam0, epsilon,k_sigma)
    
    PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)

    #MD steps
    StartTime = time.time()
    LastTime = time.time()
    i = 0
    if SaveTrajectory:

        #Names=['O']*M+(['C']*(N-M))
        Names=(N/2)*['O']+(N/2)*['C']
        Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
    
    PEnergyMat=range(MaxSteps)    
    while i < MaxSteps or MaxSteps <= 0:
        #do one step of the integration by calling the Fortran libraries
        Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos,Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
        PEnergyMat[i]=PEnergy
        i += 1
        
        #check if we need to rescale the velocities 
        if i % RescaleSteps == 0:
            Vel = RescaleVelocities(Vel, Temp)

        #check if we need to output the positions 
        if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
            Pdb.write(Pos)            

        #check if we need to update the display            
        if time.time() - LastTime > DisplayFreq:
            print ("%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy))
            LastTime = time.time()

    if SaveTrajectory:
        Pdb.close()     
        
    plt.figure(1,figsize=(3,3))
    plt.plot(range(MaxSteps),PEnergyMat,label=str(k_sigma)) 
    plt.xlabel('timeSteps')
    plt.ylabel('U')    
    
    #do one last update of the display         
    print ("%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy))

    StopTime = time.time()
    print ("Total time: %.1f s" % (StopTime - StartTime))



from scipy.stats import linregress
def DiffusionRunChargeVsLam():
    #set the init box width, number of particles, temperature, and timestep
    
    k_sigma=0
    epsilon=1
    
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.)  
    
    count3=0;
    Lam0Mat=[ (L/48),(L/24), (L/12), (L/6), (L/3), (L/2), (L/1)]
    M=4
    DMat=range(len(Lam0Mat))
    
    Charge=InitCharge(N,1)
  
    
    for Lam0 in Lam0Mat:
        
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
        
       
        N = 240
        
        rho = 0.80
        L = (N / rho)**(1./3.)
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        
        MeanSQRDispMat=np.zeros(int(NStepsProd/WriteSteps))
        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                Pos0=Pos.copy()
                
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:
                Pos0=np.array(Pos0,float)
                Pos=np.array(Pos,float)
                rMat=Pos-Pos0
                SQRdispMatInst=range(len(rMat[:,1]))
                for ii in range(len(rMat[:,1])):
                    SQRdispMatInst[ii]=sum(rMat[ii,:]*rMat[ii,:])
                SQRdispMatInst=np.array(SQRdispMatInst, float)
                MeanSQRDispMat[count2]= SQRdispMatInst.mean()
                TimeMat[count2]=(i-NStepsTot1)*dt
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
        plt.figure(1,figsize=(4,4))        
        plt.plot(TimeMat,MeanSQRDispMat,'o',label=str(Lam0))  
        plt.xlabel('time')
        plt.ylabel('MeanSQRDispMat')
        plt.legend()
        
        if SaveTrajectory:
            Pdb.close() 
    
        slope, intercept, r_value, p_value, std_err=linregress(TimeMat, MeanSQRDispMat)
        DMat[count3]=slope/6
        count3+=1
        print("Lam0="+str(Lam0))
    plt.figure(2,figsize=(4,4))
    plt.plot(Lam0Mat,DMat,'o')  
    plt.xlabel('Lam0')
    plt.ylabel('D')
    
    x=np.array(Lam0Mat,float)
    y=np.array(DMat,float)
    np.savetxt(str(np.random.normal())+"Lam0MatvsD.csv", [x,y], delimiter=",")

#DiffusionRunChargeVsLam()


def DiffusionRunChargeVsM(ChargeDist):
    #set the init box width, number of particles, temperature, and timestep
    
    
    epsilon=1
   
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.) 
    
    Lam0=(L/6)
    count3=0;
    MMat=[ 2, 4, 6, 8, 16]
    k_sigma=0
    DMat=range(len(MMat))
    
    Charge=InitCharge(N,ChargeDist)
  
    
    for M in MMat:
        
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
        
       
        N = 240
        
        rho = 0.80
        L = (N / rho)**(1./3.)
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        
        MeanSQRDispMat=np.zeros(int(NStepsProd/WriteSteps))
        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                Pos0=Pos.copy()
                
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:
                Pos0=np.array(Pos0,float)
                Pos=np.array(Pos,float)
                rMat=Pos-Pos0
                SQRdispMatInst=range(len(rMat[:,1]))
                for ii in range(len(rMat[:,1])):
                    SQRdispMatInst[ii]=sum(rMat[ii,:]*rMat[ii,:])
                SQRdispMatInst=np.array(SQRdispMatInst, float)
                MeanSQRDispMat[count2]= SQRdispMatInst.mean()
                TimeMat[count2]=(i-NStepsTot1)*dt
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
#        plt.figure(1,figsize=(4,4))        
#        plt.plot(TimeMat,MeanSQRDispMat,'o',label=str(k_sigma))  
#        plt.xlabel('time')
#        plt.ylabel('MeanSQRDispMat')
#        plt.legend()
        
        if SaveTrajectory:
            Pdb.close() 
    
        slope, intercept, r_value, p_value, std_err=linregress(TimeMat, MeanSQRDispMat)
        DMat[count3]=slope/6
        count3+=1
        print("M="+str(M))
    plt.figure(1,figsize=(4,4))
    plt.plot(MMat,DMat,'o',label=str(ChargeDist)) 
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('D')
    x=np.array(MMat,float)
    y=np.array(DMat,float)
    np.savetxt(str(np.random.normal())+"ChargeDist="+str(ChargeDist)+"MMatvsD.csv", [x,y], delimiter=",")



def DiffusionRunChargeVsK(ChargeDist):
    #set the init box width, number of particles, temperature, and timestep
    
    
    epsilon=1
   
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.) 
    
    Lam0=(L/1)
    count3=0;
    k_sigmaMat=[ 0, 1, 10, 100]
    M=16
    DMat=range(len(k_sigmaMat))
    
    Charge=InitCharge(N,ChargeDist)
  
    
    for k_sigma in k_sigmaMat:
        
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
        
       
        N = 240
        
        rho = 0.80
        L = (N / rho)**(1./3.)
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        
        MeanSQRDispMat=np.zeros(int(NStepsProd/WriteSteps))
        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                Pos0=Pos.copy()
                
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:
                Pos0=np.array(Pos0,float)
                Pos=np.array(Pos,float)
                rMat=Pos-Pos0
                SQRdispMatInst=range(len(rMat[:,1]))
                for ii in range(len(rMat[:,1])):
                    SQRdispMatInst[ii]=sum(rMat[ii,:]*rMat[ii,:])
                SQRdispMatInst=np.array(SQRdispMatInst, float)
                MeanSQRDispMat[count2]= SQRdispMatInst.mean()
                TimeMat[count2]=(i-NStepsTot1)*dt
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
#        plt.figure(1,figsize=(4,4))        
#        plt.plot(TimeMat,MeanSQRDispMat,'o',label=str(k_sigma))  
#        plt.xlabel('time')
#        plt.ylabel('MeanSQRDispMat')
#        plt.legend()
        
        if SaveTrajectory:
            Pdb.close() 
    
        slope, intercept, r_value, p_value, std_err=linregress(TimeMat, MeanSQRDispMat)
        DMat[count3]=slope/6
        count3+=1
        print("k_sigma="+str(k_sigma))
    plt.figure(2,figsize=(4,4))
    plt.plot(k_sigmaMat,DMat,'o',label=str(ChargeDist))  
    plt.legend()
    plt.xlabel('k_sigma')
    plt.ylabel('D')
    x=np.array(k_sigmaMat,float)
    y=np.array(DMat,float)
    np.savetxt(str(np.random.normal())+"ChargeDist="+str(ChargeDist)+"k_sigmaMatvsD.csv", [x,y], delimiter=",")

#DiffusionRunChargeVsK(1)
#DiffusionRunChargeVsK(1)
#DiffusionRunChargeVsK(1)

def DiffusionRunChargeVsDist():
    #set the init box width, number of particles, temperature, and timestep
    
    
    epsilon=1
   
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.) 
    
    Lam0=(L)
    count3=0;
    k_sigma=0
    ChargeConfigMat=[0,1,2,3,4]
    M=16
    DMat=range(len(ChargeConfigMat))
    
    
  
    
    for ChargeConfig in ChargeConfigMat:
        
        Charge=InitCharge(N,ChargeConfig)
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
        
       
        N = 240
        
        rho = 0.80
        L = (N / rho)**(1./3.)
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        
        MeanSQRDispMat=np.zeros(int(NStepsProd/WriteSteps))
        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                Pos0=Pos.copy()
                
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:
                Pos0=np.array(Pos0,float)
                Pos=np.array(Pos,float)
                rMat=Pos-Pos0
                SQRdispMatInst=range(len(rMat[:,1]))
                for ii in range(len(rMat[:,1])):
                    SQRdispMatInst[ii]=sum(rMat[ii,:]*rMat[ii,:])
                SQRdispMatInst=np.array(SQRdispMatInst, float)
                MeanSQRDispMat[count2]= SQRdispMatInst.mean()
                TimeMat[count2]=(i-NStepsTot1)*dt
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
        plt.figure(1,figsize=(4,4))        
        plt.plot(TimeMat,MeanSQRDispMat,'o',label=str(ChargeConfig))  
        plt.xlabel('time')
        plt.ylabel('MeanSQRDispMat')
        plt.legend()
        
        if SaveTrajectory:
            Pdb.close() 
    
        slope, intercept, r_value, p_value, std_err=linregress(TimeMat, MeanSQRDispMat)
        DMat[count3]=slope/6
        count3+=1
        print("ChargeConfig="+str(ChargeConfig))
    plt.figure(2,figsize=(4,4))
    plt.plot(ChargeConfigMat,DMat,'o')  
    plt.xlabel('Charge Distribution')
    plt.ylabel('D')
    x=np.array(ChargeConfigMat,float)
    y=np.array(DMat,float)
    np.savetxt(str(np.random.normal())+"ChargeConfigvsD.csv", [x,y], delimiter=",")

#DiffusionRunChargeVsDist()
#DiffusionRunChargeVsDist()
#DiffusionRunChargeVsDist()

 
    
def RgVsM(ChargeDist):
    
    #set the init box width, number of particles, temperature, and timestep
    
    k_sigma=0
    epsilon=1
    
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.)  
    Lam0=L/1
    count3=0;
    MMat=[2,4,6,8,16]

    Charge=InitCharge(N,ChargeDist)
  
    AvgEndEndDistMat=np.zeros_like(MMat,float)
    AvgRgMat=np.zeros_like(MMat,float)
    for M in MMat:
        
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        

        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
            
        AvgInstEndEndDistMat=np.zeros(NStepsTot2-NStepsTot1-1,float)
        AvgInstRgMat=np.zeros(NStepsTot2-NStepsTot1-1,float)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                

            
            #Production run     
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:

                InstRgMat=np.zeros((N/M),float)
                InstEndEndDistMat=np.zeros((N/M),float)
                for ix in range(N/M):
                    Pos=np.array(Pos,float)
                    Polymer=Pos[(ix*M):((ix+1)*M),:]
                    Polymer=np.array(Polymer,float)
                    #print(Polymer)
                    COM=[Polymer[:,0].mean()/len(Polymer[:,0]), Polymer[:,1].mean()/len(Polymer[:,0]), Polymer[:,2].mean()/len(Polymer[:,0])]                    
                    COM=np.array(COM,float)
                    COMPol=Polymer-COM

                    COMPol=COMPol-L*((COMPol/L).round()) 
                    #print(COMPol)
                    sqrDistMat=[sum((np.array([a,b,c],float))**2) for [a,b,c] in COMPol ]  

                    InstRgMat[ix]=((1./M)*sum(sqrDistMat))**0.5

                    InstEndEndVect=Pos[(ix*M),:]-Pos[((ix+1)*M)-1,:]
                    InstEndEndVect=np.array(InstEndEndVect,float)
                    InstEndEndVect=InstEndEndVect-L*((InstEndEndVect/L).round())
                    InstEndEndDistMat[ix]=(sum(InstEndEndVect**2))**0.5
                

                AvgInstEndEndDistMat[count2]=InstEndEndDistMat.mean()
                AvgInstRgMat[count2]=InstRgMat.mean()   
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
                

        AvgEndEndDistMat[count3]=AvgInstEndEndDistMat.mean()
        AvgRgMat[count3]=AvgInstRgMat.mean()
#        print(AvgInstEndEndDistMat.mean())
#        print(AvgInstEndEndDistMat)        
#        print(AvgEndEndDistMat)
        if SaveTrajectory:
            Pdb.close() 

        count3+=1
        print("M="+str(M))
        
    plt.figure(3,figsize=(4,4))
    plt.plot(MMat,AvgRgMat,'o',label=str(ChargeDist))  
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('Rg')

    x=np.array(MMat,float)
    y=np.array(AvgRgMat,float)
    np.savetxt(str(np.random.normal())+"ChargeDist="+str(ChargeDist)+"MvsRg.csv", [x,y], delimiter=",")
      
#    plt.figure(2,figsize=(4,4))
#    plt.plot(MMat,AvgEndEndDistMat,'o')  
#    plt.xlabel('M')
#    plt.ylabel('End to End Distance')    
#
#    x=np.array(MMat,float)
#    y=np.array(AvgEndEndDistMat,float)
#    np.savetxt(str(np.random.normal())+"MvsEEDist.csv", [x,y], delimiter=",")
    
    

    
def RgVsK(ChargeDist):
    
    #set the init box width, number of particles, temperature, and timestep
    
    k_sigmaMat=[ 0, 1, 10, 100]
    epsilon=1
    
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.)  
    Lam0=L/1
    count3=0;
    M=16

    Charge=InitCharge(N,ChargeDist)
  
    AvgEndEndDistMat=np.zeros_like(k_sigmaMat,float)
    AvgRgMat=np.zeros_like(k_sigmaMat,float)
    for k_sigma in k_sigmaMat:
        
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        

        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
            
        AvgInstEndEndDistMat=np.zeros(NStepsTot2-NStepsTot1-1,float)
        AvgInstRgMat=np.zeros(NStepsTot2-NStepsTot1-1,float)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                

            
            #Production run     
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:

                InstRgMat=np.zeros((N/M),float)
                InstEndEndDistMat=np.zeros((N/M),float)
                for ix in range(N/M):
                    Pos=np.array(Pos,float)
                    Polymer=Pos[(ix*M):((ix+1)*M),:]
                    Polymer=np.array(Polymer,float)
                    #print(Polymer)
                    COM=[Polymer[:,0].mean()/len(Polymer[:,0]), Polymer[:,1].mean()/len(Polymer[:,0]), Polymer[:,2].mean()/len(Polymer[:,0])]                    
                    COM=np.array(COM,float)
                    COMPol=Polymer-COM

                    COMPol=COMPol-L*((COMPol/L).round()) 
                    #print(COMPol)
                    sqrDistMat=[sum((np.array([a,b,c],float))**2) for [a,b,c] in COMPol ]  

                    InstRgMat[ix]=((1./M)*sum(sqrDistMat))**0.5

                    InstEndEndVect=Pos[(ix*M),:]-Pos[((ix+1)*M)-1,:]
                    InstEndEndVect=np.array(InstEndEndVect,float)
                    InstEndEndVect=InstEndEndVect-L*((InstEndEndVect/L).round())
                    InstEndEndDistMat[ix]=(sum(InstEndEndVect**2))**0.5
                

                AvgInstEndEndDistMat[count2]=InstEndEndDistMat.mean()
                AvgInstRgMat[count2]=InstRgMat.mean()   
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
                

        AvgEndEndDistMat[count3]=AvgInstEndEndDistMat.mean()
        AvgRgMat[count3]=AvgInstRgMat.mean()
#        print(AvgInstEndEndDistMat.mean())
#        print(AvgInstEndEndDistMat)        
#        print(AvgEndEndDistMat)
        if SaveTrajectory:
            Pdb.close() 

        count3+=1
        print("K="+str(k_sigma))
        
    plt.figure(4,figsize=(4,4))
    plt.plot(k_sigmaMat,AvgRgMat,'o',label=str(ChargeDist))  
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Rg')

    x=np.array(k_sigmaMat,float)
    y=np.array(AvgRgMat,float)
    np.savetxt(str(np.random.normal())+"ChargeDist="+str(ChargeDist)+"KvsRg.csv", [x,y], delimiter=",")
      
#    plt.figure(2,figsize=(4,4))
#    plt.plot(MMat,AvgEndEndDistMat,'o')  
#    plt.xlabel('M')
#    plt.ylabel('End to End Distance')    
#
#    x=np.array(MMat,float)
#    y=np.array(AvgEndEndDistMat,float)
#    np.savetxt(str(np.random.normal())+"MvsEEDist.csv", [x,y], delimiter=",")
    
    
   
def RgVsDist():
    
    #set the init box width, number of particles, temperature, and timestep
    
    ChargeDistMat=[0,1,2,3,4]
    epsilon=1
    k_sigma=0
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.)  
    Lam0=L/1
    count3=0;
    M=16
  
    AvgEndEndDistMat=np.zeros_like(ChargeDistMat,float)
    AvgRgMat=np.zeros_like(ChargeDistMat,float)
    
    for ChargeDist in ChargeDistMat:
        
        Charge=InitCharge(N,ChargeDist)
        SaveTrajectory=False;  
        dt=0.001
        #TotalTime=100
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 1000
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        

        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=['O']*M+(['C']*(N-M))
            Pdb = atomwrite.pdbfile('anim.pdb',L,AtomNames=Names)
            
        AvgInstEndEndDistMat=np.zeros(NStepsTot2-NStepsTot1-1,float)
        AvgInstRgMat=np.zeros(NStepsTot2-NStepsTot1-1,float)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                

            
            #Production run     
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:

                InstRgMat=np.zeros((N/M),float)
                InstEndEndDistMat=np.zeros((N/M),float)
                for ix in range(N/M):
                    Pos=np.array(Pos,float)
                    Polymer=Pos[(ix*M):((ix+1)*M),:]
                    Polymer=np.array(Polymer,float)
                    #print(Polymer)
                    COM=[Polymer[:,0].mean()/len(Polymer[:,0]), Polymer[:,1].mean()/len(Polymer[:,0]), Polymer[:,2].mean()/len(Polymer[:,0])]                    
                    COM=np.array(COM,float)
                    COMPol=Polymer-COM

                    COMPol=COMPol-L*((COMPol/L).round()) 
                    #print(COMPol)
                    sqrDistMat=[sum((np.array([a,b,c],float))**2) for [a,b,c] in COMPol ]  

                    InstRgMat[ix]=((1./M)*sum(sqrDistMat))**0.5

                    InstEndEndVect=Pos[(ix*M),:]-Pos[((ix+1)*M)-1,:]
                    InstEndEndVect=np.array(InstEndEndVect,float)
                    InstEndEndVect=InstEndEndVect-L*((InstEndEndVect/L).round())
                    InstEndEndDistMat[ix]=(sum(InstEndEndVect**2))**0.5
                

                AvgInstEndEndDistMat[count2]=InstEndEndDistMat.mean()
                AvgInstRgMat[count2]=InstRgMat.mean()   
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
                

        AvgEndEndDistMat[count3]=AvgInstEndEndDistMat.mean()
        AvgRgMat[count3]=AvgInstRgMat.mean()
#        print(AvgInstEndEndDistMat.mean())
#        print(AvgInstEndEndDistMat)        
#        print(AvgEndEndDistMat)
        if SaveTrajectory:
            Pdb.close() 

        count3+=1
        print("ChargeDist="+str(ChargeDist))
        
    plt.figure(4,figsize=(4,4))
    plt.plot(ChargeDistMat,AvgRgMat,'o')  
    plt.legend()
    plt.xlabel('Charge Distribution')
    plt.ylabel('Rg')

    x=np.array(ChargeDistMat,float)
    y=np.array(AvgRgMat,float)
    np.savetxt(str(np.random.normal())+"ChargeDistvsRg.csv", [x,y], delimiter=",")
      
    


from scipy.stats import linregress
def Movie():
    #set the init box width, number of particles, temperature, and timestep
    
    k_sigma=0
    epsilon=1
    
    N = 240
    rho = 0.80
    L = (N / rho)**(1./3.)  
    
    count3=0;
    Lam0Mat=[(L/1)]
    M=4
    DMat=range(len(Lam0Mat))
    
    Charge=InitCharge(N,1)
  
    
    for Lam0 in Lam0Mat:
        
        SaveTrajectory=True;  
        dt=0.001
        #TotalTime=100
        
       
        N = 240
        
        rho = 0.80
        L = (N / rho)**(1./3.)
    
        Temp = 1.0
        
        #set the frequency in seconds to update the display    
        DisplayFreq = 0.1
        #set the frequency in md steps to rescale velocities
        RescaleSteps = 10
        #set the frequency in md steps to write coordinates
        WriteSteps = 100
    
        #set the max number of md steps; 0 for infinite loop
        NStepsEquil1 = 10000
        NStepsEquil2 = 10000
        NStepsTot1=NStepsEquil1+NStepsEquil2
        NStepsProd = 100000
        NStepsTot2=NStepsTot1+NStepsProd
        #NStepsTotFinal=TotalTime/dt
        #get the initial positions, velocities, and acceleration (forces)    
        Pos = InitPositions(N, L)
        Vel = InitVelocities(N, Temp)
        Accel = InitAccel(Pos,Charge, M,Lam0, epsilon,k_sigma)
        
        PE, Pos=ConjugateGradient(Pos,Charge, 0.001, M,Lam0, epsilon,k_sigma, 1.e-8, 1.e-10)
    
        #MD steps
        StartTime = time.time()
        LastTime = time.time()
        i = 0
        EnergyMat=np.zeros(int(NStepsEquil2/RescaleSteps))
        count1=0
        count2=0
        
        MeanSQRDispMat=np.zeros(int(NStepsProd/WriteSteps))
        TimeMat=np.zeros(int(NStepsProd/WriteSteps))
        
        if SaveTrajectory:
    
            Names=(N/2)*['O']+(N/2)*['C']
            Pdb = atomwrite.pdbfile('anim2.pdb',L,AtomNames=Names)
        
        while i < NStepsTot2 or NStepsTot2 <= 0:
            #do one step of the integration by calling the Fortran libraries
            Pos, Vel, Accel, KEnergy, PEnergy = exlib.vvintegrate(Pos, Charge, Vel, Accel,M, L, Cut,Lam0, epsilon,k_sigma, dt)
            
            i += 1
            
            #first equilibration
            if i % RescaleSteps == 0 and i<NStepsEquil1:
                Vel = RescaleVelocities(Vel, Temp)
     
            #Second Equilibration
            if i % RescaleSteps == 0 and i>NStepsEquil1 and i<NStepsTot1:
                Vel = RescaleVelocities(Vel, Temp)           
                EnergyMat[count1]=KEnergy+PEnergy
                count1+=1
            
            #Rescale Velocities at End of second Equil
            if i==NStepsTot1:
                EnergyMat=np.array(EnergyMat,float)
                EnergyAverage=EnergyMat.mean()
                KETarget=EnergyAverage-PEnergy
                VScale = np.sqrt(1.5 * len(Vel) * Temp / KETarget)
                Vel = Vel * VScale
                Pos0=Pos.copy()
                
            if  i>NStepsTot1 and i<NStepsTot2 and i%WriteSteps==0:
                Pos0=np.array(Pos0,float)
                Pos=np.array(Pos,float)
                rMat=Pos-Pos0
                SQRdispMatInst=range(len(rMat[:,1]))
                for ii in range(len(rMat[:,1])):
                    SQRdispMatInst[ii]=sum(rMat[ii,:]*rMat[ii,:])
                SQRdispMatInst=np.array(SQRdispMatInst, float)
                MeanSQRDispMat[count2]= SQRdispMatInst.mean()
                TimeMat[count2]=(i-NStepsTot1)*dt
                count2+=1
            
            
            #check if we need to output the positions 
            if SaveTrajectory and WriteSteps > 0 and i % WriteSteps == 0:
                Pdb.write(Pos)            
    
            #check if we need to update the display            
            if time.time() - LastTime > DisplayFreq:
                #print "%d  %11.4f  %11.4f  %11.4f" % (i, PEnergy + KEnergy, PEnergy, KEnergy)
                #print(str(i))
                LastTime = time.time()
        plt.figure(1,figsize=(4,4))        
        plt.plot(TimeMat,MeanSQRDispMat,'o',label=str(Lam0))  
        plt.xlabel('time')
        plt.ylabel('MeanSQRDispMat')
        plt.legend()
        
        if SaveTrajectory:
            Pdb.close() 
    
        slope, intercept, r_value, p_value, std_err=linregress(TimeMat, MeanSQRDispMat)
        DMat[count3]=slope/6
        count3+=1
        print("Lam0="+str(Lam0))
    plt.figure(2,figsize=(4,4))
    plt.plot(Lam0Mat,DMat,'o')  
    plt.xlabel('Lam0')
    plt.ylabel('D')
    
    x=np.array(Lam0Mat,float)
    y=np.array(DMat,float)
    np.savetxt(str(np.random.normal())+"Lam0MatvsD.csv", [x,y], delimiter=",")


Movie()
