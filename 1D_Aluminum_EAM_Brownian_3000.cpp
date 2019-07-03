//File:               1D_Aluminum_EAM_Brownian
//Description:        This program essentially acts as a one-dimensional moving
//                    window framework for a shock wave. For this program, we use
//                    aluminum atoms and implement a Brownian (Langevin) equation
//                    of motion for these aluminum atoms in the discretization. The 
//                    Embedded Atom Model (EAM) Potential is used to calculate the 
//                    forces on the atoms in the chain. 
//Author:             Alexander Stephen Davis
//Revised:            09/20/18
//Global Variables:   aluminumMass           - The mass of an aluminum atom in amu.
//                    cShock                 - The sound velocity in aluminum in angstroms/picosecond; 
//                                             used in the shock wave equation.
//                    sShock                 - Unitless parameter used in shock wave equation.
//                    dSpaceDimen            - Spacial dimensionality for a linear chain.
//                    alattice               - Lattice constant for aluminum in angstroms.
//                    boltzmannConstant      - Boltzmann constant in relevant units.
//                    convertToGigaPascals   - Converts the units of amu / (angstroms * 
//                                             picoseconds^2) to GigaPascals.
//                    convertEV              - Converts the units of eV to (amu * 
//                                             square angstroms) / square picoseconds
//                    cutoffRadius           - Pairwise cutoff for the atoms when calculating 
//                                             the EAM potential values.
//                    debyeFrequency          - The Debye frequency of copper (1 / picoseconds)
//                    txtFileLength          - Number of values in the EAM txt files 
//                                             These values include energies, positions,
//                                             and electron densities.
//                    equilibrationTime      - Sets the equilibration time (number of time 
//                                             steps before calculations begin) for the
//                                             simulation.
//                    totalNumberOfAtoms     - Total number of atoms in simulation
//                    alAtom                 - Structure that contains parameters for
//                                             each atom.
//                    totalTimeSteps         - The total number of time steps for the 
//                                             simulation.
//                    timeAfterEquilibration - Total number of time steps in simulation 
//                                             after equilibration.
//                    firstCovarianceTerms   - Array that holds the values of the first 
//                                             component of the covariance of the potential 
//                                             virial stresses.
//                    secondCovarianceTerms  - Array that holds the values of the second 
//                                             component of the covariance of the potential 
//                                             virial stresses.
//                    zeroTemperatureTerms   - Array that holds the terms of the zero-
//                                             temperature (Born) expression used in the 
//                                             elasticity tensor.
//                    aheadOfShock           - Structure that contains the values ahead
//                                             of the shock wave front.
//                    behindShock            - Structure that contains the values behind
//                                             the shock wave front.
//                    simValue               - Structure that contains important values for
//                                             the MD simulation.
//                    EAMValue               - Structure that contains the values important 
//                                             for the EAM calculations
//                    totalEAMValue          - Structure that contains the total (summed) values 
//                                             for the EAM calculations
//                    timeStep               - Time step for simulation in picoseconds
//                    totalTime              - Total number of time steps for simulation.
//Initial Conditions: shockWaveVelocity       - Velocity of the moving shock wave in
//                                              angstroms/picosecond
//                    initParticleVelocity    - Initial particle velocity in angstroms / picosecond
//                    initialStrain           - Strain ahead of the shock wave (unitless).
//                    initialStress           - Stress ahead of the shock wave (amu / (
//                                              angstroms * square picoseconds)).
//                    initialDensity          - Initial density of material in amu / cubic angstrom
//                    initialTemperature      - Initial temperature in Kelvin
//                    initial Friction Coef   - Damping factor for the atoms ahead of the shock wave
//                                              (1 / picoseconds)
//                    final Friction Coef     - Damping factor for the atoms behind the shock wave
//                                              (1 / picoseconds)

#include <random>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "AtomInfo.h"
#include "EAMPotentialValues.h"
#include "ShockWaveValues.h"
#include "SimulationValues.h"

//*****************************
//Define Global Variables First
//*****************************


//*******************
//Important Constants
//*******************

//Mass of each aluminum atom in Atomic Mass Units (amu)
double aluminumMass = 26.98;

//Sound velocity in aluminum at zero pressure (angstroms/picosecond)
double cShock = 53.28;

//Shock equation parameter (unitless)
double sShock = 1.34;

//Space Dimensionality for a linear chain (unitless)
double dSpaceDimen = 1.00;

//Lattice constant for aluminum (angstroms)
double alattice = 4.05 / sqrt(2);

//Boltzmann Constant ((amu * square angstroms) / (square picoseconds * Kelvin))
double boltzmannConstant = 0.83144599;

//Converts the units of amu / (angstroms * picoseconds^2) to GigaPascals
double convertToGigaPascals = 0.0166053904;

//Converts the units of eV to (amu * square angstroms) / square picoseconds
double convertEV = 9648.32796;

//Pairwise cutoff for the atoms when calculating the EAM potential (angstroms). 
double cutoffRadius = 6.28721;

//The Debye frequency of copper (1 / picoseconds)
double debyeFrequency = (cShock * 3.1415) / alattice;

//Number of values in the EAM txt files These values include energies, positions,
//and electron densities. 
int txtFileLength = 10000;

//Sets the equilibration time (number of time steps before calculations begin) for 
//the simulation.
int equilibrationTime = 50000;


//***************************
//Important Simulation Values
//***************************

//Total number of atoms in the simulation
int totalNumberOfAtoms = 3000;

//Structure that contains the current as well as new position, acceleration,
//and velocity of each atom. Must have same number as totalNumberOfAtoms!
struct atom alAtom[3000];

//Total number of time steps in simulation
int totalTimeSteps = 300000;

//Total number of time steps in simulation after equilibration
int timeAfterEquilibration = 250000;

//Array that holds the values of the first component of the covariance of the 
//potential stresses. These values have the following units: (amu^2 * angstroms^4) / 
//(square picoseconds). This array must have the same number as timeAfterEquilibration 
//since a time average is performed on the array values. 
double firstCovarianceTerms[250000];

//Array that holds the values of the second component of the covariance of the 
//potential stresses. These values have the following units: (amu * angstroms^2) / 
//(square picoseconds). This array must have the same number as timeAfterEquilibration 
//since a time average is performed on the array values.
double secondCovarianceTerms[250000];

//Array that holds the terms of the zero-temperature (Born) expression
//used in the elasticity tensor. These values have the following units: (amu * angstroms^2) / 
//(square picoseconds). This array must have the same number as timeAfterEquilibration 
//since a time average is performed on the array values.
double zeroTemperatureTerms[250000];


//********************
//Important Structures
//********************

//Structure that contains the values ahead of the shock wave (initial values)
struct shockValues aheadOfShock;

//Structure that contains the values behind the shock wave (final values)
struct shockValues behindShock;

//Structure that contains the values important for the actual Velocity Verlet simulation
struct simValues simValue;

//Structure that contains the values important for the EAM calculations
struct EAMValues EAMValue[10000];

//Structure that contains the total (summed) values for the EAM calculations
struct EAMValues totalEAMValue;


//********************************
//More Important Simulation Values
//********************************

//Defining some of these important simulation values
void simulation_values() {

    //Time step in picoseconds
    simValue.timeStep = 0.001;
    
    //Total time for simulation
    simValue.totalTime = simValue.timeStep * totalTimeSteps;
    
}//end of simulation_values function


//*************************
//Define Initial Conditions
//*************************

//Function to define initial conditions
void initial_conditions() {

    //Shock velocity (angstroms/picosecond)
    aheadOfShock.shockWaveVelocity = 50.00;

    //Particle velocity ahead of the shock wave (angstroms / picosecond)
    aheadOfShock.particleVelocity = 0.0;
    
    //Stress value ahead of shock wave (amu / (angstroms * square picoseconds))
    aheadOfShock.stress = 0.0;
    
    //Strain value ahead of shock wave (unitless)
    aheadOfShock.strain = 0.0;

    //Initial density (amu/cubic angstrom)
    aheadOfShock.density = 1.680177;

    //Initial temperature (Kelvin)
    aheadOfShock.temperature = 298;
    
    //Damping factor for atoms ahead of the shock wave (1 / picosecond)
    aheadOfShock.frictionCoefficient = (debyeFrequency / 2);
    
    //Damping factor for atoms behind the shock wave (1 / picosecond)
    behindShock.frictionCoefficient = (debyeFrequency / 2);

}//end of initial_conditions function

//******************************************************************************
//******************************************************************************

//*******************
//Program Begins Here
//*******************

//Function:     shock_equation
//Description:  Uses the linear shock wave equation to calculate the particle 
//              velocity behind the shock wave in angstroms/picosecond. 
//Parameters:   None
//Returns:      Nothing
//Calls:        shockWaveVelocity
//              initial particle velocity
//Globals:      cShock
//              sShock
void shock_equation() {

    double finalParticleVelocity = 0;

    finalParticleVelocity = 0;//((aheadOfShock.shockWaveVelocity - cShock) / sShock) + 
            //aheadOfShock.particleVelocity;

    behindShock.particleVelocity = finalParticleVelocity;

}//end of shock_equation function
//******************************************************************************
//Function:     mass_equation
//Description:  Uses the conservation of mass equation to calculate the strain 
//              behind the shock wave. The value of strain is unitless.
//Parameters:   None
//Returns:      Nothing
//Calls:        initial particle Velocity
//              final particle Velocity
//              shockWaveVelocity
//              initial strain
//Globals:      None
void mass_equation() {

    double finalStrain = 0;

    finalStrain = - ((behindShock.particleVelocity - aheadOfShock.particleVelocity) / 
            aheadOfShock.shockWaveVelocity) + aheadOfShock.strain;

    behindShock.strain = finalStrain;

}//end of mass_equation function
//******************************************************************************
//Function:     momentum_equation
//Description:  Uses the conservation of momentum equation to calculate the stress
//              behind the shock wave in (amu / (angstroms * square picoseconds))
//Parameters:   None
//Returns:      Nothing
//Calls:        initial density
//              shockWaveVelocity
//              particleVelocity
//              initial stress
//Globals:      None
void momentum_equation() {

    double finalStress = 0;

    finalStress = - ((aheadOfShock.density * aheadOfShock.shockWaveVelocity * (behindShock.particleVelocity
            - aheadOfShock.particleVelocity)) + aheadOfShock.stress);

    behindShock.stress = finalStress;

}//end of momentum_equation function
//******************************************************************************
//Function:     temperature_equation
//Description:  Calculates the temperature change produced by the shock wave.
//Parameters:   None
//Returns:      Nothing
//Calls:        initial temperature
//              gruneisen_constant()
//              initial specific volume
//              final specific volume
//Globals:      None
void temperature_equation() {

    double finalTemperature = 0;

    finalTemperature = 298;//506.908;

    behindShock.temperature = finalTemperature;
    
}//end of temperature_equation function
//******************************************************************************
//Function:     length_and_spacing
//Description:  Calculates the new length of the cell as well as the equilibrium
//              distance between atoms based on the strain value behind the shock
//              wave front. 
//Parameters:   None
//Returns:      Nothing 
//Calls:        behindShock.strain
//Globals:      alattice 
//              totalNumberOfAtoms,
void length_and_spacing() {
    
    //New length of the cell due to the imparted strain
    simValue.strainLength = ((1 + behindShock.strain) * alattice) * totalNumberOfAtoms;
    
    //New equilibrium distance between the atoms due to the imparted strain
    simValue.strainSpacing = ((1 + behindShock.strain) * alattice);
    
}//end of length_and_spacing function
//******************************************************************************
//Function:     eam_structure_array
//Description:  Creates a structure array that includes the relevant values to 
//              calculate the EAM potential. The position quantities have units
//              of Angstroms; the energy quantities have units of eV; and the 
//              electron density quantities are unitless. However, we convert the 
//              units of energy from eV to (amu * angstroms^2) / picoseconds^2. 
//Parameters:   None
//Returns:      Nothing
//Calls:        Nothing
//Globals:      convertEV
void eam_structure_array() {
    
    FILE * FRho_Array;
    FILE * PhiR_Array;
    FILE * RhoR_Array;
    FILE * Rho_R_Array;
        
    FRho_Array = fopen("FRho.txt", "r");
        
    for (int count = 0; count < txtFileLength; count++) {

        fscanf(FRho_Array, "%lf", &EAMValue[count].energyF);
 
        EAMValue[count].energyF = EAMValue[count].energyF * convertEV;
   
    }
        
    fclose(FRho_Array);
    
    PhiR_Array = fopen("PhiR.txt", "r");
        
    for (int count = 0; count < txtFileLength; count++) {

        fscanf(PhiR_Array, "%lf", &EAMValue[count].energyPhi);
 
        EAMValue[count].energyPhi = EAMValue[count].energyPhi * convertEV;
   
    }
        
    fclose(PhiR_Array);
    
    RhoR_Array = fopen("RhoR.txt", "r");
        
    for (int count = 0; count < txtFileLength; count++) {

        fscanf(RhoR_Array, "%lf", &EAMValue[count].electronDensity);
   
    }
        
    fclose(RhoR_Array);
    
    Rho_R_Array = fopen("Rho_R.txt", "r");
    
    for (int count = 0; count < txtFileLength; count++) {

        fscanf(Rho_R_Array, "%lf %lf", &EAMValue[count].totalDensity,
                &EAMValue[count].distance);   
        
    }
    
    fclose(Rho_R_Array);
   
}//end of eam_structure_array function
//******************************************************************************
//Function:     kronecker_delta
//Description:  Uses the kronecker delta function to return either a 1 or a 0
//              given a pair of integers.
//Parameters:   firstInteger  - first of the pair of integers
//              secondInteger - second of the pair of integers
//Returns:      integerToReturn 
//Calls:        Nothing               
//Globals:      None
double kronecker_delta(int firstInteger, int secondInteger) {
    
    int integerToReturn = 0;
    
    if (firstInteger == secondInteger) {
        
        integerToReturn = 1;
        
    } else {
        
        integerToReturn = 0;
        
    }
    
    return integerToReturn;

}//end of kronecker_delta function
//******************************************************************************
//Function:     reset_eam_values 
//Description:  Initializes to zero all the parameters used in the EAM calculations.
//              This function must be called after all the calculations have finished
//              for each atom in the chain.
//Parameters:   None  
//Returns:      Nothing
//Calls:        Important EAM parameters
//Globals:      None
void reset_eam_values() {
    
    totalEAMValue.totalEnergyPhi = 0;
    totalEAMValue.totalElectronDensityRho = 0;
    totalEAMValue.totalEnergyF = 0;
    
    totalEAMValue.totalPhiDerivative = 0;
    totalEAMValue.totalRhoDerivative = 0;
    totalEAMValue.totalEnergyFDerivative = 0;
    
    totalEAMValue.totalPhiSecondDerivative = 0;
    totalEAMValue.totalRhoSecondDerivative = 0;
    totalEAMValue.totalEnergyFSecondDerivative = 0;
    
    totalEAMValue.totalPhiMixDerivative = 0;
    totalEAMValue.totalRhoMixDerivative = 0;
    
    totalEAMValue.totalPhiPotential = 0;
    totalEAMValue.totalRhoPotential = 0;
    
}//end of reset_eam_values function
//******************************************************************************
//Function:     energy_F_calculations
//Description:  Calculates the total energy as a function of the sum of the electron
//              densities as well as the first and second derivatives of this 
//              total energy with respect to the total electron density. If FDE = 1, 
//              the derivatives are calculated by a forward difference. If FDE = 2, 
//              the derivatives are calculated by a backward difference. If FDE = 3,
//              the derivatives are calculated by a central difference. The units
//              of this total energy as a function of total electron density are 
//              as follows: (amu * angstroms^2) / picoseconds^2. 
//Parameters:   filePosition - The position of the pointer in the txt files
//              fde          - Designates which difference scheme will be used to 
//                             calculate the derivatives. 
//Returns:      Nothing  
//Calls:        energyF
//              totalDensity
//              std::abs
//Globals:      None
void energy_F_calculations(int filePosition, int fde) {
    
    totalEAMValue.totalEnergyF = EAMValue[filePosition].energyF;
    
    if (fde == 1) {
        
        totalEAMValue.totalEnergyFDerivative = ((EAMValue[filePosition + 1].energyF - 
                EAMValue[filePosition].energyF) / (std::abs(EAMValue[filePosition + 1].totalDensity - 
                EAMValue[filePosition].totalDensity)));
        
        totalEAMValue.totalEnergyFSecondDerivative = (((- EAMValue[filePosition + 3].energyF) + 
                (4 * EAMValue[filePosition + 2].energyF) - (5 * EAMValue[filePosition + 1].energyF) + 
                (2 * EAMValue[filePosition].energyF)) / (pow(std::abs(EAMValue[filePosition + 1].totalDensity - 
                EAMValue[filePosition].totalDensity),2)));
        
    } else if (fde == 2) {
        
        totalEAMValue.totalEnergyFDerivative = ((EAMValue[filePosition].energyF - 
                EAMValue[filePosition - 1].energyF) / (std::abs(EAMValue[filePosition].totalDensity - 
                EAMValue[filePosition - 1].totalDensity)));
        
        totalEAMValue.totalEnergyFSecondDerivative = (((2 * EAMValue[filePosition].energyF) - 
                (5 * EAMValue[filePosition - 1].energyF) + (4 * EAMValue[filePosition - 2].energyF) - 
                (EAMValue[filePosition - 3].energyF)) / (pow(std::abs(EAMValue[filePosition].totalDensity - 
                EAMValue[filePosition - 1].totalDensity),2)));
        
    } else if (fde == 3) {

        totalEAMValue.totalEnergyFDerivative = ((EAMValue[filePosition + 1].energyF - 
                EAMValue[filePosition - 1].energyF) / (2 * (std::abs(EAMValue[filePosition + 1].totalDensity - 
                EAMValue[filePosition].totalDensity))));
        
        totalEAMValue.totalEnergyFSecondDerivative = (((EAMValue[filePosition + 1].energyF) - 
                (2 * EAMValue[filePosition].energyF) + (EAMValue[filePosition - 1].energyF)) / 
                (pow(std::abs(EAMValue[filePosition + 1].totalDensity - EAMValue[filePosition].totalDensity),2)));
        
    }
       
}//end of energy_F_calculations
//******************************************************************************
//Function:     calculate_eam_energy
//Description:  Finds the position in the EAM txt files of the energy value given
//              the position of the total electron density value. Sets a pointer
//              at the correct position and calls the energy_F_calculations() 
//              function to perform the necessary calculations. This function will
//              continue to loop through the txt files until the correct position
//              is found. 
//Parameters:   None
//Returns:      Nothing
//Calls:        totalElectronDensityRho
//              totalDensity
//              energy_F_calculations()
//Globals:      txtFileLength
void calculate_eam_energy(int left, int right) {
    
    double totalRho = totalEAMValue.totalElectronDensityRho;
    
    right = right - 1;
    
    if ((totalRho > EAMValue[left].totalDensity) && (totalRho < EAMValue[right].totalDensity)) {
                 
        if (totalRho == EAMValue[right].totalDensity) {
            
            int fde = 3;
            int filePosition = right;
                     
            energy_F_calculations(filePosition, fde);
            
        } else if (totalRho == EAMValue[left].totalDensity) {
            
            int fde = 3;
            int filePosition = left;
            
            energy_F_calculations(filePosition, fde);
                
        } else if ((totalRho < EAMValue[right].totalDensity) &&
                (totalRho > EAMValue[right-1].totalDensity)) {
                     
            double rightDifference = EAMValue[right].totalDensity - totalRho;
            double leftDifference = totalRho - EAMValue[right-1].totalDensity;
                     
            if (rightDifference <= leftDifference) {
                  
                int fde = 3;
                int filePosition = right;
                
                energy_F_calculations(filePosition, fde);
                         
            } else if (leftDifference < rightDifference) {
                 
                int fde = 3;
                int filePosition = right - 1;
                
                energy_F_calculations(filePosition, fde);
                         
            }
                     
        } else if ((totalRho > EAMValue[left].totalDensity) &&
                (totalRho < EAMValue[left+1].totalDensity)) {
                     
            double rightDifference = EAMValue[left+1].totalDensity - totalRho;
            double leftDifference = totalRho - EAMValue[left].totalDensity;
                     
            if (rightDifference <= leftDifference) {
                  
                int fde = 3;
                int filePosition = left+1;
                
                energy_F_calculations(filePosition, fde);
                         
            } else if (leftDifference < rightDifference) {
                 
                int fde = 3;
                int filePosition = left;
                
                energy_F_calculations(filePosition, fde);
                         
            }
                     
        } else {
            
            int middlePosition = (right - left) / 2;
            
            calculate_eam_energy(left, middlePosition);
            calculate_eam_energy(middlePosition, right);
            
        }
                 
    } else if (totalRho <= EAMValue[0].totalDensity) {
            
        int fde = 1; 
        int filePosition = 0;
            
        energy_F_calculations(filePosition, fde);
                     
    } else if (totalRho >= EAMValue[txtFileLength - 1].totalDensity) {
            
        int fde = 2;
        int filePosition = txtFileLength - 1;
        
        energy_F_calculations(filePosition, fde);
                     
    }
        
}//end of calculate_eam_energy function 
//******************************************************************************
//Function:     elasticity_potential_terms
//Description:  Calculates the derivative of the rho term with respect to distance
//              as well as the derivative of the phi term with respect to distance.
//              These values are then multiplied by the magnitude of the distance
//              to obtain the "work" terms which makes up the potential part of 
//              the elasticity tensor. The sum of these terms is calculated for 
//              each atom in the chain. If FDE = 1, the derivatives are calculated 
//              by a forward difference. If FDE = 2, the derivatives are calculated 
//              by a backward difference. If FDE = 3, the derivatives are calculated 
//              by a central difference. The phi potential terms have units of energy
//              (amu * angstroms^2) / picoseconds^2. The rho potential terms have
//              some unit of electron density.             
//Parameters:   filePosition     - The position of the pointer in the txt files
//              periodicDistance - The distance between the two atoms in question.
//                                 This distance can be positive or negative. 
//              fde              - Designates which difference scheme will be used to 
//                                 calculate the derivatives. 
//Returns:      Nothing
//Calls:        energyPhi
//              distance
//              electronDensity
//              std::abs
//Globals:      None
void elasticity_potential_terms(int filePosition, double periodicDistance, int fde) {
    
    if (fde == 1) {
        
        totalEAMValue.totalPhiPotential += (((EAMValue[filePosition + 1].energyPhi - 
                EAMValue[filePosition].energyPhi) / (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance))) * (periodicDistance / std::abs(periodicDistance)) * 
                std::abs(EAMValue[filePosition].distance));
                          
        totalEAMValue.totalRhoPotential += (((EAMValue[filePosition + 1].electronDensity - 
                EAMValue[filePosition].electronDensity) / (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance))) * (periodicDistance / std::abs(periodicDistance)) * 
                std::abs(EAMValue[filePosition].distance));
           
    } else if (fde == 2) {
        
        totalEAMValue.totalPhiPotential += (((EAMValue[filePosition].energyPhi - 
                EAMValue[filePosition - 1].energyPhi) / (std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance))) * (periodicDistance / std::abs(periodicDistance)) * 
                std::abs(EAMValue[filePosition].distance));
                          
        totalEAMValue.totalRhoPotential += (((EAMValue[filePosition].electronDensity - 
                EAMValue[filePosition - 1].electronDensity) / (std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance))) * (periodicDistance / std::abs(periodicDistance)) * 
                std::abs(EAMValue[filePosition].distance));
        
    } else if (fde == 3) {
        
        totalEAMValue.totalPhiPotential += (((EAMValue[filePosition + 1].energyPhi - 
                EAMValue[filePosition - 1].energyPhi) / (2 * (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance)))) * (periodicDistance / std::abs(periodicDistance)) * 
                std::abs(EAMValue[filePosition].distance));
                          
        totalEAMValue.totalRhoPotential += (((EAMValue[filePosition + 1].electronDensity - 
                EAMValue[filePosition - 1].electronDensity) / (2 * (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance)))) * (periodicDistance / std::abs(periodicDistance)) * 
                std::abs(EAMValue[filePosition].distance));
        
    }
    
}//end of elasticity_potential_terms function
//******************************************************************************
//Function:     rho_phi_mix_derivative
//Description:  Calculates the derivative of the rho term with respect to distance
//              as well as the derivative of the phi term with respect to distance.
//              These values are then multiplied by the second derivative of the 
//              magnitude of the distance r_ij. This entire value is then multiplied
//              by the square of the absolute distance between the particles to obtain
//              the correct units for the bond-stiffness term used in the elasticity 
//              tensor calculation. The sum of these terms is calculated for 
//              each atom in the chain. If FDE = 1, the derivatives are calculated 
//              by a forward difference. If FDE = 2, the derivatives are calculated 
//              by a backward difference. If FDE = 3, the derivatives are calculated 
//              by a central difference. The phi terms have units of energy
//              (amu * angstroms^2) / picoseconds^2. The rho terms have some unit 
//              of electron density.             
//Parameters:   filePosition - The position of the pointer in the txt files 
//              fde          - Designates which difference scheme will be used to 
//                                 calculate the derivatives. 
//Returns:      Nothing
//Calls:        energyPhi
//              distance
//              electronDensity
//              std::abs
//Globals:      None   
void rho_phi_mix_derivative(int filePosition, int fde) {
    
    if (fde == 1) {
        
        totalEAMValue.totalPhiMixDerivative += (((EAMValue[filePosition + 1].energyPhi - 
                EAMValue[filePosition].energyPhi) / (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance))) * ((1 / std::abs(EAMValue[filePosition].distance)) - 
                (pow(EAMValue[filePosition].distance, 2) / pow(std::abs(EAMValue[filePosition].distance),3))) * 
                (pow(std::abs(EAMValue[filePosition].distance), 2)));
                          
        totalEAMValue.totalRhoMixDerivative += (((EAMValue[filePosition + 1].electronDensity - 
                EAMValue[filePosition].electronDensity) / (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance))) * ((1 / std::abs(EAMValue[filePosition].distance)) - 
                (pow(EAMValue[filePosition].distance, 2) / pow(std::abs(EAMValue[filePosition].distance),3))) * 
                (pow(std::abs(EAMValue[filePosition].distance), 2)));
           
    } else if (fde == 2) {
        
        totalEAMValue.totalPhiMixDerivative += (((EAMValue[filePosition].energyPhi - 
                EAMValue[filePosition - 1].energyPhi) / (std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance))) * ((1 / std::abs(EAMValue[filePosition].distance)) - 
                (pow(EAMValue[filePosition].distance, 2) / pow(std::abs(EAMValue[filePosition].distance),3))) * 
                (pow(std::abs(EAMValue[filePosition].distance), 2)));
                          
        totalEAMValue.totalRhoMixDerivative += (((EAMValue[filePosition].electronDensity - 
                EAMValue[filePosition - 1].electronDensity) / (std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance))) * ((1 / std::abs(EAMValue[filePosition].distance)) - 
                (pow(EAMValue[filePosition].distance, 2) / pow(std::abs(EAMValue[filePosition].distance),3))) * 
                (pow(std::abs(EAMValue[filePosition].distance), 2)));
        
    } else if (fde == 3) {
        
        totalEAMValue.totalPhiMixDerivative += (((EAMValue[filePosition + 1].energyPhi - 
                EAMValue[filePosition - 1].energyPhi) / (2 * (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance)))) * ((1 / std::abs(EAMValue[filePosition].distance)) - 
                (pow(EAMValue[filePosition].distance, 2) / pow(std::abs(EAMValue[filePosition].distance),3))) * 
                (pow(std::abs(EAMValue[filePosition].distance), 2)));
                          
        totalEAMValue.totalRhoMixDerivative += (((EAMValue[filePosition + 1].electronDensity - 
                EAMValue[filePosition - 1].electronDensity) / (2 * (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance)))) * ((1 / std::abs(EAMValue[filePosition].distance)) - 
                (pow(EAMValue[filePosition].distance, 2) / pow(std::abs(EAMValue[filePosition].distance),3))) * 
                (pow(std::abs(EAMValue[filePosition].distance), 2)));
        
    }
    
}//end of rho_phi_mix_derivative function
//******************************************************************************
//Function:     rho_phi_second_derivative
//Description:  Calculates the second derivative of the rho term with respect to distance
//              as well as the second derivative of the phi term with respect to distance.
//              This quantity is then multiplied by the square of the absolute 
//              distance between the particles to obtain the correct units for the 
//              bond-stiffness term used in the elasticity tensor calculation. 
//              The sum of these terms is calculated for each atom in the chain. 
//              If FDE = 1, the derivatives are calculated by a forward difference. 
//              If FDE = 2, the derivatives are calculated by a backward difference. 
//              If FDE = 3, the derivatives are calculated by a central difference. 
//              The phi terms have units of energy (amu * angstroms^2) / picoseconds^2. 
//              The rho terms have some unit of electron density.             
//Parameters:   filePosition     - The position of the pointer in the txt files 
//              periodicDistance - The distance between the two atoms in question.
//                                 This distance can be positive or negative.
//              fde              - Designates which difference scheme will be used to 
//                                 calculate the derivatives. 
//Returns:      Nothing
//Calls:        energyPhi
//              distance
//              electronDensity
//              std::abs
//Globals:      None
void rho_phi_second_derivative(int filePosition, double periodicDistance, int fde) {
    
    if (fde == 1) {
        
        totalEAMValue.totalPhiSecondDerivative += ((((- EAMValue[filePosition + 3].energyPhi) + 
                (4 * EAMValue[filePosition + 2].energyPhi) - (5 * EAMValue[filePosition + 1].energyPhi) + 
                (2 * EAMValue[filePosition].energyPhi)) / (pow(std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance),2))) * (pow(periodicDistance / std::abs(periodicDistance),2)) * 
                (pow(std::abs(EAMValue[filePosition].distance),2)));
                          
        totalEAMValue.totalRhoSecondDerivative += ((((- EAMValue[filePosition + 3].electronDensity) + 
                (4 * EAMValue[filePosition + 2].electronDensity) - (5 * EAMValue[filePosition + 1].electronDensity) + 
                (2 * EAMValue[filePosition].electronDensity)) / (pow(std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance),2))) * (pow(periodicDistance / std::abs(periodicDistance),2)) * 
                (pow(std::abs(EAMValue[filePosition].distance),2)));
        
    } else if (fde == 2) {
        
        totalEAMValue.totalPhiSecondDerivative += ((((2 * EAMValue[filePosition].energyPhi) - 
                (5 * EAMValue[filePosition - 1].energyPhi) + (4 * EAMValue[filePosition - 2].energyPhi) - 
                (EAMValue[filePosition - 3].energyPhi)) / (pow(std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance),2))) * (pow(periodicDistance / std::abs(periodicDistance),2)) * 
                (pow(std::abs(EAMValue[filePosition].distance),2)));
                          
        totalEAMValue.totalRhoSecondDerivative += ((((2 * EAMValue[filePosition].electronDensity) - 
                (5 * EAMValue[filePosition - 1].electronDensity) + (4 * EAMValue[filePosition - 2].electronDensity) - 
                (EAMValue[filePosition - 3].electronDensity)) / (pow(std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance),2))) * (pow(periodicDistance / std::abs(periodicDistance),2)) * 
                (pow(std::abs(EAMValue[filePosition].distance),2)));
            
    } else if (fde == 3) {
        
        totalEAMValue.totalPhiSecondDerivative += ((((EAMValue[filePosition + 1].energyPhi) - 
                (2 * EAMValue[filePosition].energyPhi) + (EAMValue[filePosition - 1].energyPhi)) / 
                (pow(std::abs(EAMValue[filePosition + 1].distance - EAMValue[filePosition].distance),2))) * 
                (pow(periodicDistance / std::abs(periodicDistance),2)) * 
                (pow(std::abs(EAMValue[filePosition].distance),2)));
                          
        totalEAMValue.totalRhoSecondDerivative += ((((EAMValue[filePosition + 1].electronDensity) - 
                (2 * EAMValue[filePosition].electronDensity) + (EAMValue[filePosition - 1].electronDensity)) / 
                (pow(std::abs(EAMValue[filePosition + 1].distance - EAMValue[filePosition].distance),2))) * 
                (pow(periodicDistance / std::abs(periodicDistance),2)) * 
                (pow(std::abs(EAMValue[filePosition].distance),2)));
        
    }
        
}//end of rho_phi_second_derivative function
//******************************************************************************
//Function:     rho_phi_derivative
//Description:  Calculates the derivative of the rho term with respect to distance
//              as well as the derivative of the phi term with respect to distance. 
//              The sum of these terms is calculated for each atom in the chain. 
//              If FDE = 1, the derivatives are calculated by a forward difference. 
//              If FDE = 2, the derivatives are calculated by a backward difference. 
//              If FDE = 3, the derivatives are calculated by a central difference. 
//              The phi terms have units of force (amu * angstroms) / picoseconds^2. 
//              The rho terms have some unit of electron density / angstrom.             
//Parameters:   filePosition     - The position of the pointer in the txt files 
//              periodicDistance - The distance between the two atoms in question.
//                                 This distance can be positive or negative.
//              fde              - Designates which difference scheme will be used to 
//                                 calculate the derivatives. 
//Returns:      Nothing
//Calls:        energyPhi
//              distance
//              electronDensity
//              std::abs
//Globals:      None
void rho_phi_derivative(int filePosition, double periodicDistance, int fde) {
    
    if (fde == 1) {
        
        totalEAMValue.totalPhiDerivative += (((EAMValue[filePosition + 1].energyPhi - 
                EAMValue[filePosition].energyPhi) / (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance))) * (periodicDistance / std::abs(periodicDistance)));
                          
        totalEAMValue.totalRhoDerivative += (((EAMValue[filePosition + 1].electronDensity - 
                EAMValue[filePosition].electronDensity) / (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance))) * (periodicDistance / std::abs(periodicDistance)));
           
    } else if (fde == 2) {
        
        totalEAMValue.totalPhiDerivative += (((EAMValue[filePosition].energyPhi - 
                EAMValue[filePosition - 1].energyPhi) / (std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance))) * (periodicDistance / std::abs(periodicDistance)));
                          
        totalEAMValue.totalRhoDerivative += (((EAMValue[filePosition].electronDensity - 
                EAMValue[filePosition - 1].electronDensity) / (std::abs(EAMValue[filePosition].distance - 
                EAMValue[filePosition - 1].distance))) * (periodicDistance / std::abs(periodicDistance)));
        
    } else if (fde == 3) {
        
        totalEAMValue.totalPhiDerivative += (((EAMValue[filePosition + 1].energyPhi - 
                EAMValue[filePosition - 1].energyPhi) / (2 * (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance)))) * (periodicDistance / std::abs(periodicDistance)));
                          
        totalEAMValue.totalRhoDerivative += (((EAMValue[filePosition + 1].electronDensity - 
                EAMValue[filePosition - 1].electronDensity) / (2 * (std::abs(EAMValue[filePosition + 1].distance - 
                EAMValue[filePosition].distance)))) * (periodicDistance / std::abs(periodicDistance)));
        
    }
       
}//end of rho_phi_derivative function
//******************************************************************************
//Function:     rho_phi_calculation 
//Description:  Calculates the sum of the pair potentials and the electron densities
//              for each atom in the chain. 
//Parameters:   filePosition - The position of the pointer in the txt files.
//Returns:      Nothing 
//Calls:        energyPhi
//              electronDensity
//Globals:      None
void rho_phi_calculation(int filePosition) {
    
    totalEAMValue.totalEnergyPhi += EAMValue[filePosition].energyPhi;
    totalEAMValue.totalElectronDensityRho += EAMValue[filePosition].electronDensity;
    
}//end of rho_phi_calculation function
//******************************************************************************
//Function:     calculate_eam_values
//Description:  Finds the position in the EAM txt files of the electron density and 
//              pair potential values given the position of the distance value. 
//              Sets a pointer at the correct position and calls the relevant functions 
//              to perform the necessary calculations. This function will continue 
//              to loop through the txt files until the correct position is found. 
//Parameters:   periodicDistance - The distance between the two atoms in question.
//                                 This distance can be positive or negative.
//Returns:      Nothing
//Calls:        distance
//              rho_phi_calculation()
//              rho_phi_derivative()
//              rho_phi_second_derivative()
//              rho_phi_mix_derivative()
//              elasticity_potential_terms()
//Globals:      txtFileLength                  
void calculate_eam_values(double periodicDistance, int left, int right) {
        
    right = right - 1;
    
    if ((std::abs(periodicDistance) > EAMValue[left].distance) && 
            (std::abs(periodicDistance) < EAMValue[right].distance)) {
                 
        if (std::abs(periodicDistance) == EAMValue[right].distance) {
            
            int fde = 3;
            int filePosition = right;
            
            rho_phi_calculation(filePosition);
            rho_phi_derivative(filePosition, periodicDistance, fde);
            rho_phi_second_derivative(filePosition, periodicDistance, fde);
            rho_phi_mix_derivative(filePosition, fde);
            elasticity_potential_terms(filePosition, periodicDistance, fde);
        
        } else if (std::abs(periodicDistance) == EAMValue[left].distance) {
            
            int fde = 3;
            int filePosition = left;
            
            rho_phi_calculation(filePosition);
            rho_phi_derivative(filePosition, periodicDistance, fde);
            rho_phi_second_derivative(filePosition, periodicDistance, fde);
            rho_phi_mix_derivative(filePosition, fde);
            elasticity_potential_terms(filePosition, periodicDistance, fde);
   
        } else if ((std::abs(periodicDistance) < EAMValue[right].distance) &&
                (std::abs(periodicDistance) > EAMValue[right-1].distance)) {
                     
            double rightDifference = EAMValue[right].distance - std::abs(periodicDistance);
            double leftDifference = std::abs(periodicDistance) - EAMValue[right-1].distance;
                     
            if (rightDifference <= leftDifference) {
                
                int fde = 3;
                int filePosition = right;
            
                rho_phi_calculation(filePosition);
                rho_phi_derivative(filePosition, periodicDistance, fde);
                rho_phi_second_derivative(filePosition, periodicDistance, fde);
                rho_phi_mix_derivative(filePosition, fde);
                elasticity_potential_terms(filePosition, periodicDistance, fde);
                         
            } else if (leftDifference < rightDifference) {
                
                int fde = 3;
                int filePosition = right - 1;
                
                rho_phi_calculation(filePosition);
                rho_phi_derivative(filePosition, periodicDistance, fde);
                rho_phi_second_derivative(filePosition, periodicDistance, fde);
                rho_phi_mix_derivative(filePosition, fde);
                elasticity_potential_terms(filePosition, periodicDistance, fde);
                         
            }
                     
        } else if ((std::abs(periodicDistance) > EAMValue[left].distance) &&
                (std::abs(periodicDistance) < EAMValue[left+1].distance)) {
                     
            double rightDifference = EAMValue[left+1].distance - std::abs(periodicDistance);
            double leftDifference = std::abs(periodicDistance) - EAMValue[left].distance;
                     
            if (rightDifference <= leftDifference) {
                
                int fde = 3;
                int filePosition = left + 1;
            
                rho_phi_calculation(filePosition);
                rho_phi_derivative(filePosition, periodicDistance, fde);
                rho_phi_second_derivative(filePosition, periodicDistance, fde);
                rho_phi_mix_derivative(filePosition, fde);
                elasticity_potential_terms(filePosition, periodicDistance, fde);
                         
            } else if (leftDifference < rightDifference) {
                
                int fde = 3;
                int filePosition = left;
                
                rho_phi_calculation(filePosition);
                rho_phi_derivative(filePosition, periodicDistance, fde);
                rho_phi_second_derivative(filePosition, periodicDistance, fde);
                rho_phi_mix_derivative(filePosition, fde);
                elasticity_potential_terms(filePosition, periodicDistance, fde);
                         
            }
                     
        } else {
                     
            int middlePosition = (right - left) / 2;
            
            calculate_eam_values(periodicDistance, left, middlePosition);
            calculate_eam_values(periodicDistance, middlePosition, right);
                         
        } 
    
    } else if (std::abs(periodicDistance) <= EAMValue[0].distance) {
            
        int fde = 1;
        int filePosition = 0;
            
        rho_phi_calculation(filePosition);
        rho_phi_derivative(filePosition, periodicDistance, fde);
        rho_phi_second_derivative(filePosition, periodicDistance, fde);
        rho_phi_mix_derivative(filePosition, fde);
        elasticity_potential_terms(filePosition, periodicDistance, fde);
                     
    } else if (std::abs(periodicDistance) >= EAMValue[txtFileLength - 1].distance) {
                     
        int fde = 2;
        int filePosition = txtFileLength - 1;
            
        rho_phi_calculation(filePosition);
        rho_phi_derivative(filePosition, periodicDistance, fde);
        rho_phi_second_derivative(filePosition, periodicDistance, fde);
        rho_phi_mix_derivative(filePosition, fde);
        elasticity_potential_terms(filePosition, periodicDistance, fde);
                     
    }

}//end of calculate_eam_values function
//******************************************************************************
//Function:     periodic_BC
//Description:  Utilizes periodic boundary conditions to calculate the spacing
//              between atoms which are farther apart than one half times the 
//              box length. This is accomplished by "mirroring" the atoms on the
//              opposite side of the current atom. If the atoms are not farther 
//              apart than one half times the box length, the function returns 
//              the original displacement. These displacements are dimensional 
//              and have units of angstroms.
//Parameters:   distance - Distance between the present atom the the atom either
//                         to its left or right. 
//Returns:      mirroredDistance 
//Calls:        std::abs
//              simValue.strainLength
//Globals:      None
double periodic_BC(double distance) {

    double mirroredDistance = 0;
    
    double absDistance = std::abs(distance);

    if (absDistance >= (simValue.strainLength / 2)) {
           
        if (distance <= 0) {
            
            mirroredDistance = distance + simValue.strainLength;
            
        } else if (distance > 0) {
            
            mirroredDistance = distance - simValue.strainLength;
            
        }
                    
    } else {
        
        mirroredDistance = distance;
        
    }

    return mirroredDistance;

}//end of periodic_BC function
//******************************************************************************
//Function:     right_atomic_distance
//Description:  Calculates the displacement between the current atom and the atoms
//              to the right of the current atom. These distances must be less than
//              the cutoff radius. If the atom is the last atom in the chain, the 
//              function calculates the difference in the first atom's position
//              and the last atom's position. If the present atom + the pair atom
//              is greater than the total number of atoms - 1, the function loops
//              back around to the beginning of the chain. 
//Parameters:   presentAtom - Atom the program is currently simulating
//              pairAtom    - Atom interacting with the current atom; used in the 
//                            pair potential. 
//Returns:      rightPeriodicDistance 
//Calls:        The new position of the relevant atoms
//              periodic_BC()
//Globals:      totalNumberOfAtoms
double right_atomic_distance(int presentAtom, int pairAtom) {
    
    double rightDistance = 0;
    double rightPeriodicDistance = 0;
    
    if (presentAtom == (totalNumberOfAtoms - 1)) {
        
        rightDistance = alAtom[presentAtom].newPosition - 
                alAtom[pairAtom - 1].newPosition;
        
    } else if ((presentAtom + pairAtom) > (totalNumberOfAtoms - 1)) {
                
        rightDistance = alAtom[presentAtom].newPosition - 
                alAtom[(presentAtom + pairAtom) - totalNumberOfAtoms].newPosition;
                
    } else {
                
        rightDistance = alAtom[presentAtom].newPosition - 
                alAtom[presentAtom + pairAtom].newPosition;
                
    }
    
    rightPeriodicDistance = periodic_BC(rightDistance);
    
    return rightPeriodicDistance;
        
}//end of right_atomic_distance function
//******************************************************************************
//Function:     left_atomic_distance
//Description:  Calculates the displacement between the current atom and the atoms
//              to the left of the current atom. These distances must be less than
//              the cutoff radius. If the atom is the first atom in the chain, the 
//              function calculates the difference in the first atom's position
//              and the last atom's position. If the present atom - the pair atom
//              is less than the first atom in the chain, the function loops
//              back around to the end of the chain. 
//Parameters:   presentAtom - Atom the program is currently simulating
//              pairAtom    - Atom interacting with the current atom; used in the 
//                            pair potential. 
//Returns:      leftPeriodicDistance 
//Calls:        The new position of the relevant atoms
//              periodic_BC()
//Globals:      totalNumberOfAtoms
double left_atomic_distance(int presentAtom, int pairAtom) {
    
    double leftDistance = 0;
    double leftPeriodicDistance = 0;
    
    if (presentAtom == 0) {
        
        leftDistance = alAtom[presentAtom].newPosition - 
                alAtom[totalNumberOfAtoms - pairAtom].newPosition;
        
    } else if ((presentAtom - pairAtom) < 0) {
                
        leftDistance = alAtom[presentAtom].newPosition - 
                alAtom[(presentAtom - pairAtom) + totalNumberOfAtoms].newPosition;
                   
    } else {
                
        leftDistance = alAtom[presentAtom].newPosition - 
                alAtom[presentAtom - pairAtom].newPosition;
                
    }
   
    leftPeriodicDistance = periodic_BC(leftDistance);
    
    return leftPeriodicDistance;
    
}//end of left_atomic_distance function
//******************************************************************************
//Function:     eam_potential
//Description:  Calculates the energy between two given atoms using the Embedded
//              Atom Model Method. This energy is approximated as an interatomic
//              potential.
//Parameters:   None
//Returns:      eamPotential
//Calls:        totalEnergyF
//              totalEnergyPhi
//Globals:      None
double eam_potential() {
    
    double eamPotential = 0;
    
    eamPotential = totalEAMValue.totalEnergyF + (0.5 * totalEAMValue.totalEnergyPhi);
    
    return eamPotential;
    
}//end of eam_potential function
//******************************************************************************
//Function:     eam_force
//Description:  Calculates the force between two given atoms using the Embedded
//              Atom Model Method. The force is defined as the negative gradient 
//              (in our 1D case, derivative) of the potential function.      
//Parameters:   None   
//Returns:      eamForce
//Calls:        totalEnergyFDerivative
//              totalRhoDerivative
//              totalPhiDerivative
//Globals:      None  
double eam_force() {

    double eamForce = 0;
    
    eamForce = - ((totalEAMValue.totalEnergyFDerivative * totalEAMValue.totalRhoDerivative)
            + (0.5 * totalEAMValue.totalPhiDerivative));
    
    return eamForce;
    
}//end of eam_force function
//******************************************************************************
//Function:     elasticity_tensor_potential
//Description:  Calculates the potential part of the elasticity tensor. This will 
//              be summed for all atoms at every time step in a later function.
//Parameters:   None     
//Returns:      potentialTerm
//Calls:        totalEnergyFDerivative
//              totalRhoPotential
//              totalPhiPotential
//Globals:      None   
double elasticity_tensor_potential() {

    double potentialTerm = 0;
    
    potentialTerm = ((totalEAMValue.totalEnergyFDerivative * totalEAMValue.totalRhoPotential)
            + (0.5 * totalEAMValue.totalPhiPotential));
    
    return potentialTerm;
    
}//end of elasticity_tensor_potential function
//******************************************************************************
//Function:     eam_bond_stiffness
//Description:  Calculates the bond stiffness part of the elasticity tensor. This 
//              will be summed for all atoms at every time step in a later function.
//Parameters:   None    
//Returns:      eamBondStiffness
//Calls:        totalEnergyFSecondDerivative
//              totalRhoPotential
//              totalEnergyFDerivative
//              totalRhoSecondDerivative
//              totalRhoMixDerivative
//              totalPhiSecondDerivative
//              totalPhiMixDerivative
//Globals:      None       
double eam_bond_stiffness() {
    
    double eamBondStiffness = 0;
    
    eamBondStiffness = ((totalEAMValue.totalEnergyFSecondDerivative * pow(totalEAMValue.totalRhoPotential, 2))
            + (totalEAMValue.totalEnergyFDerivative * totalEAMValue.totalRhoSecondDerivative)
            + (totalEAMValue.totalEnergyFDerivative * totalEAMValue.totalRhoMixDerivative) 
            + (0.5 * totalEAMValue.totalPhiSecondDerivative)
            + (0.5 * totalEAMValue.totalPhiMixDerivative));
    
    return eamBondStiffness;
  
}//end of eam_bond_stiffness function
//******************************************************************************
//Function:     normal_distribution
//Description:  Creates a normal distribution with a mean of zero and a standard
//              deviation of one using the c++ library. 
//Parameters:   None
//Returns:      A number within a normal distribution with the parameters 
//              described above. 
//Calls:        steady_clock and default_random_engine from the c++ library.
//Globals:      None
double normal_distribution() {

    double mean = 0;
    double standardDeviation = 1;
    
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    float sample;
    std::normal_distribution<float> d(mean, standardDeviation); 
    sample = d(gen); 
        
    return sample;    

}//end of normal_distribution function
//******************************************************************************
//Function:     thermal_vibration_velocity
//Description:  Finds the vibrational velocity (angstroms / picosecond) 
//              due to thermal fluctuations for the Langevin thermostat. 
//Parameters:   None
//Returns:      thermalVibrationVelocity
//Calls:        behindShock.temperature
//              normal_distribution()
//Globals:      boltzmannConstant
//              frictionCoef
//              copperMass
//              simValue.timeStep
double thermal_vibration_velocity() {

    double thermalVibrationVelocity = 0;

    thermalVibrationVelocity = sqrt((simValue.timeStep * boltzmannConstant * 
            behindShock.temperature * behindShock.frictionCoefficient) / aluminumMass) 
            * normal_distribution();

    return thermalVibrationVelocity;

}//end of thermal_vibration_forces function
//******************************************************************************
//Function:     write_to_csv_file
//Description:  Creates seven csv files. These files contain the positions,
//              accelerations, and velocities of the given number of atoms for
//              the given number of time steps. The function takes the present 
//              atom argument from compute_parameters and appends the displacement, 
//              acceleration, and velocity of that object in the respective files. 
//              Also appends the zeroTemp component, potential component, and 
//              kinetic component of the elasticity tensor to its corresponding file.
//              Finally, appends the total number of atoms and the modulus of 
//              elasticity to a file. 
//Parameters:   presentAtom    -  Atom the program is currently simulating
//              atomParameter  -  Either the present position, acceleration, or
//                                velocity of the given atom.
//              presentTime    -  Current time in simulation
//              flag           -  Integer indicating whether we're dealing with 
//                                1 = position, 2 = acceleration, or 3 = velocity.
//Returns:      Nothing
//Calls:        Nothing
//Globals:      totalNumberOfAtoms
void write_to_csv_file(int presentAtom, double atomParameter, double presentTime, int flag) {

    if (flag == 1) {

        FILE * positions;
        
        if (presentAtom == 0) {
            
            positions = fopen("Positions3000-1.csv", "a");
            fprintf(positions, "%0.6f", presentTime);
            fprintf(positions, ",");
            fprintf(positions, "%0.6f", atomParameter);
            fprintf(positions, ",");
            fclose(positions);
            
        } else if (presentAtom == (totalNumberOfAtoms - 1)) {

            positions = fopen("Positions3000-1.csv", "a");
            fprintf(positions, "%0.6f", atomParameter);
            fprintf(positions, "\n");
            fclose(positions);

        } else {

            positions = fopen("Positions3000-1.csv", "a");
            fprintf(positions, "%0.6f", atomParameter);
            fprintf(positions, ",");
            fclose(positions);

        }

    } else if (flag == 2) {

        FILE * accelerations;
        
        if (presentAtom == 0) {
            
            accelerations = fopen("Accelerations3000-1.csv", "a");
            fprintf(accelerations, "%0.6f", presentTime);
            fprintf(accelerations, ",");
            fprintf(accelerations, "%0.6f", atomParameter);
            fprintf(accelerations, ",");
            fclose(accelerations);

        } else if (presentAtom == (totalNumberOfAtoms - 1)) {

            accelerations = fopen("Accelerations3000-1.csv", "a");
            fprintf(accelerations, "%0.6f", atomParameter);
            fprintf(accelerations, "\n");
            fclose(accelerations);

        } else {

            accelerations = fopen("Accelerations3000-1.csv", "a");
            fprintf(accelerations, "%0.6f", atomParameter);
            fprintf(accelerations, ",");
            fclose(accelerations);

        }

    } else if (flag == 3) {

        FILE * velocities;
        
        if (presentAtom == 0) {
            
            velocities = fopen("Velocities3000-1.csv", "a");
            fprintf(velocities, "%0.6f", presentTime);
            fprintf(velocities, ",");
            fprintf(velocities, "%0.6f", atomParameter);
            fprintf(velocities, ",");
            fclose(velocities);

        } else if (presentAtom == (totalNumberOfAtoms - 1)) {

            velocities = fopen("Velocities3000-1.csv", "a");
            fprintf(velocities, "%0.6f", atomParameter);
            fprintf(velocities, "\n");
            fclose(velocities);

        } else {

            velocities = fopen("Velocities3000-1.csv", "a");
            fprintf(velocities, "%0.6f", atomParameter);
            fprintf(velocities, ",");
            fclose(velocities);

        }

    } else if (flag == 4) {
        
        FILE * zeroTempComponent;
        
        zeroTempComponent = fopen("zeroTempComponent3000-1.csv", "a");
        fprintf(zeroTempComponent, "%0.6f", atomParameter);
        fprintf(zeroTempComponent, "\n");
        fclose(zeroTempComponent);
        
    } else if (flag == 5) {
        
        FILE * potentialComponent;
        
        potentialComponent = fopen("potentialComponent3000-1.csv", "a");
        fprintf(potentialComponent, "%0.6f", atomParameter);
        fprintf(potentialComponent, "\n");
        fclose(potentialComponent);
        
    } else if (flag == 6) {
        
        FILE * kineticComponent;
        
        kineticComponent = fopen("kineticComponent-1.csv", "a");
        fprintf(kineticComponent, "%0.6f", atomParameter);
        fprintf(kineticComponent, "\n");
        fclose(kineticComponent);
        
    } else if (flag == 7) {
        
        FILE * elasticityModulus;
        
        elasticityModulus = fopen("ElasticityModulus-1.csv", "a");
        fprintf(elasticityModulus, "%d", totalNumberOfAtoms);
        fprintf(elasticityModulus, ",");
        fprintf(elasticityModulus, "%0.6f", atomParameter);
        fprintf(elasticityModulus, "\n");
        fclose(elasticityModulus);
            
    }

}//end of write_to_csv_file function
//******************************************************************************
//Function:     verlet_velocity
//Description:  Uses the Verlet integration algorithm to find the velocity of 
//              each atom in the linear chain for the present time step.  If the 
//              present time = 0, then the velocity of each atom in the chain is 
//              equal to the mean particle velocity. Otherwise, the 
//              velocity is calculated using the velocity verlet algorithm.
//              The Brownian thermostat is used in this discretization. 
//              The function then appends this velocity value to the relevant file.
//              All velocities have units of angstroms/picosecond.
//Parameters:   presentTime -  Current time in simulation
//              timeStep    -  Time passed for each iteration
//Returns:      Nothing
//Calls:        behindShock.particleVelocity
//              halfStepVelocity
//              newAcceleration
//              thermal_vibration_forces()
//              write_to_csv_file()
//Globals:      totalNumberOfAtoms
//              frictionCoef
//              copperMass
void verlet_velocity(double presentTime, double timeStep) {

    int flag = 3;
        
    for (int presentAtom = 0; presentAtom < totalNumberOfAtoms;
            presentAtom = presentAtom + 1) {

        if (presentTime == 0) {

            alAtom[presentAtom].newVelocity = behindShock.particleVelocity;

        } else {
                
            alAtom[presentAtom].newVelocity = alAtom[presentAtom].halfStepVelocity 
                    + ((0.5 * timeStep) * (alAtom[presentAtom].newAcceleration - 
                    (behindShock.frictionCoefficient * alAtom[presentAtom].halfStepVelocity))) + 
                    alAtom[presentAtom].thermalVelocity;
                
        }
            
    }
        
}// end of verlet_velocity function
//******************************************************************************
//Function:     verlet_acceleration
//Description:  Calculates the acceleration of each atom in the chain by finding 
//              force between pairs of atoms which have a radial distance less than
//              the cutoff radius. The function calculate_eam_values() is called
//              to calculate the relevant quantities used in the force calculation.
//              The calculate_eam_energy() function is then called to find the energy
//              required to place the given atom into the electron cloud. The function
//              then calls eam_force() and divides this by the atom's mass to find 
//              the particle acceleration. The function then appends this acceleration 
//              value to the relevant file. This acceleration has the following 
//              dimensions: angstroms / picoseconds^2. All values used in the eam 
//              potential calculation are then reset to zero. 
//Parameters:   presentTime - Current time in simulation
//Returns:      Nothing
//Calls:        left_atomic_distance()
//              right_atomic_distance()
//              std::abs
//              calculate_eam_values()
//              calculate_eam_energy()
//              eam_force()
//              write_to_csv_file()
//              reset_eam_values()
//Globals:      totalNumberOfAtoms
//              cutoffRadius
//              copperMass
void verlet_acceleration(double presentTime) {

    int flag = 2;
    
    double leftPeriodicDistance = 0;
    double rightPeriodicDistance = 0;
    
    double particleAcceleration = 0;

    for (int presentAtom = 0; presentAtom < totalNumberOfAtoms; presentAtom++) {
        
        for (int pairAtom = 1; pairAtom <= totalNumberOfAtoms; pairAtom++) {

            leftPeriodicDistance = left_atomic_distance(presentAtom, pairAtom);
            rightPeriodicDistance = right_atomic_distance(presentAtom, pairAtom);
            
            if ((std::abs(leftPeriodicDistance) <= cutoffRadius) || 
                    (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                        (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                    calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
                    calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);                
   
                } else if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                        (std::abs(rightPeriodicDistance) > cutoffRadius)) {   
                    
                    calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
   
                } else if ((std::abs(leftPeriodicDistance) > cutoffRadius) && 
                        (std::abs(rightPeriodicDistance) <= cutoffRadius)) {  
                    
                    calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
                    
                }
                       
            } else {
                    
                break;
                    
            }
            
        }
        
        calculate_eam_energy(0, txtFileLength);
        
        particleAcceleration = eam_force() / aluminumMass;
        
        alAtom[presentAtom].newAcceleration = particleAcceleration;
        
        reset_eam_values();
            
    }
    
}//end of verlet_acceleration function
//******************************************************************************
//Function:     verlet_periodic_position
//Description:  Keeps the periodic boxes from 'drifting.' In other words, if the 
//              0th atom becomes negative, every atom in the chain is shifted to 
//              the right. If the last atom passes the length of the chain, every
//              atom in the chain is shifted to the left. 
//Parameters:   None
//Returns:      Nothing 
//Calls:        strainLength
//              strainSpacing
//              newPosition
//Globals:      totalNumberOfAtoms
void verlet_periodic_position(double presentTime) {
       
    if ((alAtom[0].newPosition) < 0 && (alAtom[totalNumberOfAtoms - 1].newPosition > simValue.strainLength)) {
           
        double firstPosition = alAtom[totalNumberOfAtoms - 1].newPosition - simValue.strainLength;
        double lastPosition = alAtom[0].newPosition + simValue.strainLength;
           
        alAtom[0].newPosition = firstPosition;
        alAtom[totalNumberOfAtoms - 1].newPosition = lastPosition;
           
    } else if (alAtom[0].newPosition < 0) {
           
        double lastPosition = alAtom[0].newPosition + simValue.strainLength;
                
        for (int presentAtom = 0; presentAtom < totalNumberOfAtoms; 
                presentAtom = presentAtom + 1) {
                
            if (presentAtom == (totalNumberOfAtoms - 1)) {
                    
                alAtom[presentAtom].newPosition = lastPosition;
                    
            } else {
                
                alAtom[presentAtom].newPosition = alAtom[presentAtom + 1].newPosition;
                
            }
                
        }
            
    } else if (alAtom[totalNumberOfAtoms - 1].newPosition > (simValue.strainLength)) {
            
        double firstPosition = alAtom[totalNumberOfAtoms - 1].newPosition - simValue.strainLength;
               
        for (int presentAtom = (totalNumberOfAtoms - 1); presentAtom >= 0; 
                presentAtom = presentAtom - 1) {
                
            if (presentAtom == 0) {
                    
                alAtom[0].newPosition = firstPosition;
                    
            } else {
                
                alAtom[presentAtom].newPosition = alAtom[presentAtom - 1].newPosition;
                
            }
                
        }

    }
    
}//end of verlet_periodic_position function
//******************************************************************************
//Function:     verlet_position
//Description:  Finds the positions for the atoms in the linear chain by calling 
//              the relevant functions. These functions will perform a Verlet 
//              integration to obtain the relevant positions. The position of each
//              atom is defined as its absolute distance from the origin. 
//              If the algorithm is just starting (the present time is 0), the 
//              position of each atom is equal to its position in the chain times 
//              the equilibrium distance between each atom. Otherwise, the 
//              position of each atom is calculated using Verlet integration.
//              These positions are then appended to the respective csv file. 
//              All positions have units of angstroms. 
//Parameters:   presentTime - Current time in simulation
//              timeStep    - Time passed for each iteration
//Returns:      Nothing 
//Calls:        strainSpacing
//              currentPosition
//              halfStepVelocity
//              write_to_csv_file()
//Globals:      totalNumberOfAtoms
void verlet_position(double presentTime, double timeStep) {

    int flag = 1;
        
    for (int presentAtom = 0; presentAtom < totalNumberOfAtoms;
            presentAtom = presentAtom + 1) {

        if (presentTime == 0) {

            alAtom[presentAtom].newPosition = (presentAtom * simValue.strainSpacing) + 
                    (simValue.strainSpacing / 2);

        } else {
                
            alAtom[presentAtom].newPosition = alAtom[presentAtom].currentPosition + 
                    (timeStep * alAtom[presentAtom].halfStepVelocity);
                
        }

    }
          
}//end of verlet_position function
//******************************************************************************
//Function:     verlet_half_step_velocity
//Description:  Finds the half-step velocity for the atoms in the linear chain
//              by calling the relevant functions. This half-step velocity function
//              is part of the expanded velocity Verlet integration algorithm. 
//              If the algorithm is just starting (the present time is 0), the 
//              half-step velocity of each atom is 0. Otherwise, the half-step
//              velocity of each atom is calculated using the Verlet integration
//              techniques. All velocities have units of angstroms/picosecond
//Parameters:   presentTime - Current time in simulation
//              timeStep    - Time passed for each iteration
//Returns:      Nothing 
//Calls:        currentVelocity
//              currentAcceleration
//              behindShock.particleVelocity
//              thermal_vibration_forces()
//Globals:      totalNumberOfAtoms
//              frictionCoef
//              copperMass
void verlet_half_step_velocity(double presentTime, double timeStep) {
        
    for (int presentAtom = 0; presentAtom < totalNumberOfAtoms;
            presentAtom = presentAtom + 1) {

        if (presentTime == 0) {

            alAtom[presentAtom].halfStepVelocity = 0;

        } else {
            
            alAtom[presentAtom].thermalVelocity = thermal_vibration_velocity();
                
            alAtom[presentAtom].halfStepVelocity = alAtom[presentAtom].currentVelocity 
                    + ((0.5 * timeStep) * (alAtom[presentAtom].currentAcceleration - 
                    (behindShock.frictionCoefficient * (alAtom[presentAtom].currentVelocity - 
                    behindShock.particleVelocity)))) + alAtom[presentAtom].thermalVelocity;
                
        }

    }   

}//end of verlet_half_step_velocity function
//******************************************************************************
//Function:     potential_covariance_terms
//Description:  Computes the first and second terms of the covariance of the 
//              potential part of the virial stress tensor. The first term is 
//              a double summation involving the product of the force produced by 
//              EAM and the distances between the two atoms. A time average 
//              is taken of this value over the entire length of the simulation. 
//              The second term is merely a single summation of the same product. 
//              The values from each of these terms are placed into two different
//              arrays. Each array has a length that is equal to the equilibration
//              time of the simulation. The units of the first array are 
//              (amu^2 * angstroms^4) / (picoseconds^4) while the units of the 
//              second array are (amu * angstroms^2) / (picoseconds^2). 
//Parameters:   timeStep - Dimensional time passed in picoseconds for each 
//                         iteration.
//Returns:      Nothing
//Calls:        left_atomic_distance()
//              right_atomic_distance()
//              std::abs
//              calculate_eam_values()
//              calculate_eam_energy()
//              elasticity_tensor_potential()
//              reset_eam_values()
//              simValue.strainLength
//              write_to_csv_file()
//Globals:      totalNumberOfAtoms
//              cutoffRadius
//              firstCovarianceTerms
//              secondCovarianceTerms
//              alattice
//              convertToGigaPascals
void potential_covariance_terms(int timeStep) {
    
    double firstSumWork = 0;
    double secondSumWork = 0;
    double potentialTerm = 0;
    
    int flag = 5;

    if (timeStep >= equilibrationTime) {

        for (int presentAtom = 0; presentAtom < totalNumberOfAtoms; presentAtom++) {
            
            for (int pairAtom = 1; pairAtom <= totalNumberOfAtoms; pairAtom++) {
            
                double leftPeriodicDistance = left_atomic_distance(presentAtom, pairAtom);
                double rightPeriodicDistance = right_atomic_distance(presentAtom, pairAtom);
            
                if ((std::abs(leftPeriodicDistance) <= cutoffRadius) || 
                        (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                    if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) > cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) > cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
                    
                    }
                       
                } else {
                    
                    break;
            
                }
            
            }
        
            calculate_eam_energy(0, txtFileLength);
        
            double totalWork = elasticity_tensor_potential();
           
            firstSumWork += totalWork;
        
            reset_eam_values();
        
        }
        
        for (int presentAtom = 0; presentAtom < totalNumberOfAtoms; presentAtom++) {
            
            for (int pairAtom = 1; pairAtom <= totalNumberOfAtoms; pairAtom++) {
            
                double leftPeriodicDistance = left_atomic_distance(presentAtom, pairAtom);
                double rightPeriodicDistance = right_atomic_distance(presentAtom, pairAtom);
            
                if ((std::abs(leftPeriodicDistance) <= cutoffRadius) || 
                        (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                    if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) > cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) > cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
                    
                    }
                       
                } else {
                    
                    break;
            
                }
            
            }
        
            calculate_eam_energy(0, txtFileLength);
        
            double totalWork = elasticity_tensor_potential();
           
            double newTotalWork = totalWork * firstSumWork;
        
            secondSumWork += newTotalWork;
        
            reset_eam_values();
        
        }
        
        potentialTerm = secondSumWork - pow(firstSumWork, 2);
        
        //Units = (amu^2 * angstroms^4) / (picoseconds^4)
        firstCovarianceTerms[timeStep - equilibrationTime] = secondSumWork;
    
        //Units = (amu * angstroms^2) / (picoseconds^2)
        secondCovarianceTerms[timeStep - equilibrationTime] = firstSumWork;
        
        double potentialStress = (potentialTerm / (simValue.strainLength * pow(alattice,2))) *
                convertToGigaPascals;
        
        if ((timeStep % 500) == 0) {
            
            write_to_csv_file(1, potentialStress, 1, flag); 
            
        }
    
    }

}//end of potential_covariance_terms function
//******************************************************************************
//Function:     total_potential_stress_covariance
//Description:  Calculates the time average for the terms in both the first
//              and second potential stress covariance terms arrays. It then
//              uses these time averages to calculate the covariance of the 
//              potential parts of the instantaneous virial stresses. The covariance
//              has the following units: (amu^2 * angstroms^4) / (picoseconds^4).
//Parameters:   None              
//Returns:      totalPotentialCovariance 
//Calls:        Nothing                
//Globals:      timeAfterEquilibration
//              firstCovarianceTerms
//              secondCovarianceTerms
double total_potential_stress_covariance() {
    
    double firstSum = 0;
    double secondSum = 0;
    
    double firstTimeAverage = 0;
    double secondTimeAverage = 0;
    
    double totalPotentialCovariance = 0;
    
    for (int i = 0; i < timeAfterEquilibration; i++) {
        
        firstSum += firstCovarianceTerms[i];
        secondSum += secondCovarianceTerms[i];
  
    }
    
    firstTimeAverage = firstSum / timeAfterEquilibration;
    secondTimeAverage = secondSum / timeAfterEquilibration;
    
    //Units = (amu^2 * angstroms^4) / (picoseconds^4)
    totalPotentialCovariance = firstTimeAverage - pow(secondTimeAverage, 2);

    return totalPotentialCovariance;
        
}//end of total_potential_stress_covariance function
//******************************************************************************
//Function:     zero_temp_summation_components
//Description:  Calculates the bond stiffness component as well as the potential 
//              component in the zero temperature (Born) term of the elasticity
//              tensor. Then calculates the instantaneous value of the zero temp
//              component. Has the following units: (amu * square angstroms) / 
//              (square picoseconds).
//Parameters:   timeStep - Dimensional time passed in picoseconds for each 
//                         iteration.              
//Returns:      Nothing    
//Calls:        left_atomic_distance()
//              right_atomic_distance()
//              std::abs
//              calculate_eam_values()
//              calculate_eam_energy()
//              eam_bond_stiffness()
//              reset_eam_values()
//              elasticity_tensor_potential()
//              simValue.strainLength
//              write_to_csv_file()
//Globals:      totalNumberOfAtoms
//              cutoffRadius
//              firstCovarianceTerms
//              secondCovarianceTerms
//              alattice
//              convertToGigaPascals
void zero_temp_summation_components(int timeStep) {
    
    double bondStiffness = 0;
    double potentialWork = 0;
    
    int csvFlag = 4;
        
    if (timeStep >= equilibrationTime) {
        
        for (int presentAtom = 0; presentAtom < totalNumberOfAtoms; presentAtom++) {
            
            for (int pairAtom = 1; pairAtom <= totalNumberOfAtoms; pairAtom++) {
                    
                double leftPeriodicDistance = left_atomic_distance(presentAtom, pairAtom);
                double rightPeriodicDistance = right_atomic_distance(presentAtom, pairAtom);
            
                if ((std::abs(leftPeriodicDistance) <= cutoffRadius) || 
                        (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                    if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) > cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) > cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
                    
                    }
                       
                } else {
                    
                    break;
            
                }
   
            }
        
            calculate_eam_energy(0, txtFileLength);
        
            double kappaValue = 2 * eam_bond_stiffness();
           
            bondStiffness += kappaValue;
        
            reset_eam_values();
        
        }
          
        for (int presentAtom = 0; presentAtom < totalNumberOfAtoms; presentAtom++) {
            
            for (int pairAtom = 1; pairAtom <= totalNumberOfAtoms; pairAtom++) {
            
                double leftPeriodicDistance = left_atomic_distance(presentAtom, pairAtom);
                double rightPeriodicDistance = right_atomic_distance(presentAtom, pairAtom);
            
                if ((std::abs(leftPeriodicDistance) <= cutoffRadius) || 
                        (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                    if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) <= cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) > cutoffRadius)) {
                    
                        calculate_eam_values(leftPeriodicDistance, 0, txtFileLength);
   
                    } else if ((std::abs(leftPeriodicDistance) > cutoffRadius) && 
                            (std::abs(rightPeriodicDistance) <= cutoffRadius)) {
                    
                        calculate_eam_values(rightPeriodicDistance, 0, txtFileLength);
                    
                    }
                       
                } else {
                    
                    break;
            
                }
            
            }
        
            calculate_eam_energy(0, txtFileLength);
        
            double totalWork = elasticity_tensor_potential();
           
            potentialWork += totalWork;
        
            reset_eam_values();
        
        } 
 
        double zeroTempEnergyComponent = (0.25 * bondStiffness) - (0.5 * potentialWork); 
        
        //Units = (amu * angstroms^2) / (picoseconds^2)
        zeroTemperatureTerms[timeStep - equilibrationTime] = zeroTempEnergyComponent;
        
        double zeroTempStress = (zeroTempEnergyComponent / (simValue.strainLength * pow(alattice,2))) *
                convertToGigaPascals;
        
        if ((timeStep % 500) == 0) {
        
            write_to_csv_file(1, zeroTempStress , 1, csvFlag);
            
        }

    }

}//end of zero_temp_summation_components function
//******************************************************************************
//Function:     kinetic_term
//Description:  Calculates the kinetic component of the elasticity tensor. This 
//              component has units (amu * angstroms^2) / (picoseconds^2).
//Parameters:   None              
//Returns:      kineticTerm 
//Calls:        behindShock.temperature
//              kronecker_delta()
//              simValue.strainLength
//              write_to_csv_file()
//Globals:      totalNumberOfAtoms
//              boltzmannConstant
//              alattice
//              convertToGigaPascals
double kinetic_term() {
    
    double kineticTerm = 0;
    
    int flag = 6;
    
    int i = 1;
    int j = 1;
    int k = 1;
    int l = 1;
    
    int deltaIK = kronecker_delta(i,k);
    int deltaJL = kronecker_delta(j,l);
    int deltaIL = kronecker_delta(i,l);
    int deltaJK = kronecker_delta(j,k);
    
    kineticTerm = (totalNumberOfAtoms * boltzmannConstant * behindShock.temperature) * 
            ((deltaIK * deltaJL) + (deltaIL * deltaJK));
    
    double kineticStress = (kineticTerm / (simValue.strainLength * pow(alattice,2))) *
            convertToGigaPascals;
    
    write_to_csv_file(1, kineticStress, 1, flag);
    
    return kineticTerm;
            
}//end of kinetic_term function
//******************************************************************************
//Function:     potential_term
//Description:  Calculates the potential component of the elasticity tensor.  
//              This component has units (amu * angstroms^2) / (picoseconds^2). 
//Parameters:   None              
//Returns:      potentialTerm
//Calls:        behindShock.temperature
//              total_potential_stress_covariance()
//Globals:      boltzmannConstant
double potential_term() {
    
    double potentialTerm = 0;
    
    potentialTerm = (total_potential_stress_covariance()) / (4 * boltzmannConstant * behindShock.temperature);
            
    return potentialTerm;

}//end of potential_term function
//******************************************************************************
//Function:     zero_temp_term
//Description:  Calculates the zero temperature component of the elasticity tensor.
//              This component has units (amu * angstroms^2) / (picoseconds^2).
//Parameters:   None              
//Returns:      zeroTempTimeAverage
//Calls:        zeroTemperatureTerms                
//Globals:      timeAfterEquilibration
double zero_temp_term() {
    
    double zeroTempSum = 0;
    double zeroTempTimeAverage = 0;
    
    for (int i = 0; i < timeAfterEquilibration; i++) {
        
        zeroTempSum += zeroTemperatureTerms[i];
        
    }
    
    zeroTempTimeAverage = zeroTempSum / timeAfterEquilibration;

    return zeroTempTimeAverage;
    
}//end of zero_temp_term function
//******************************************************************************
//Function:     elasticity_tensor
//Description:  Calculates the elasticity tensor of the linear chain which is 
//              a function of the zero temperature elasticity component, the 
//              potential stress component, and the kinetic stress component.
//              This elasticity tensor has units amu / (angstroms * picoseconds^2).
//Parameters:   None             
//Returns:      elasticityTensor  
//Calls:        simValue.strainLength
//              zero_temp_term()
//              potential_term()
//              kinetic_term()
//Globals:      alattice
double elasticity_tensor() {
    
    double elasticityTensor = 0;
    
    elasticityTensor = (zero_temp_term() - potential_term() + kinetic_term()) / 
            (simValue.strainLength * pow(alattice,2));
    
    //Units = amu / (angstroms * picoseconds^2)
    return elasticityTensor;

}//end of elasticity_tensor function
//******************************************************************************
//Function:     end_simulation  
//Description:  Ends the Verlet velocity integration methods after the last time 
//              value is reached. Calculates the required quantities. 
//Parameters:   None
//Returns:      Nothing
//Calls:        elasticity_tensor()
//              write_to_csv_file()
//Globals:      convertToGigaPascals
void end_simulation() {
    
    int flag = 7;
    
    double elasticityTensor = elasticity_tensor() * convertToGigaPascals;
        
    printf("%0.12f", elasticityTensor);
    
    write_to_csv_file(1, elasticityTensor, 1, flag);
  
}//end of end_simulation function
//******************************************************************************
//Function:     end_time_step
//Description:  Sets the current parameter to the new parameter after each time step.
//Parameters:   None
//Returns:      Nothing
//Calls:        newPosition
//              newAcceleration
//              newVelocity
//Globals:      totalNumberOfAtoms
void end_time_step() {

    for (int presentAtom = 0; presentAtom < totalNumberOfAtoms;
            presentAtom = presentAtom + 1) {

        alAtom[presentAtom].currentPosition = alAtom[presentAtom].newPosition;
        alAtom[presentAtom].currentAcceleration = alAtom[presentAtom].newAcceleration;
        alAtom[presentAtom].currentVelocity = alAtom[presentAtom].newVelocity;
        
    }
        
}//end of end_time_step function
//******************************************************************************
//Function:     call_csv_files
//Description:  Write each of the new variables to the appropriate csv files.
//Parameters:   None
//Returns:      Nothing
//Calls:        newPosition
//              newAcceleration
//              newVelocity
//              write_to_csv_file()
//Globals:      None
void call_csv_files(double presentTime, int timeStep) {
    
    if ((timeStep % 500) == 0) {
            
        for (int currentAtom = 0; currentAtom < totalNumberOfAtoms;
                currentAtom = currentAtom + 1) {
        
            double position = alAtom[currentAtom].currentPosition;
            double acceleration = alAtom[currentAtom].currentAcceleration;
            double velocity = alAtom[currentAtom].currentVelocity;
        
            write_to_csv_file(currentAtom, position, presentTime, 1);
            write_to_csv_file(currentAtom, acceleration, presentTime, 2);
            write_to_csv_file(currentAtom, velocity, presentTime, 3);
        
        }
        
    }
    
}//end of call_csv_files
//******************************************************************************
//Function:     call_time_average_values
//Description:  Calls the relevant functions that will need to be time-averaged. 
//Parameters:   integerTimeStep - Dimensional time passed in picoseconds for each 
//                                iteration.
//Returns:      Nothing
//Calls:        potential_covariance_terms()
//              zero_temp_summation_components()
//Globals:      None
void call_time_average_values(int integerTimeStep) {

    potential_covariance_terms(integerTimeStep);
    zero_temp_summation_components(integerTimeStep);
    
}//end of call_time_average_values function
//******************************************************************************
//Function:     begin_simulation
//Description:  Begins the simulation on all the atoms in the linear chain. 
//              First we define the time step, which is an integer value. 
//              We then iterate from the 0th atom in the chain to the last atom in the 
//              chain. Next, we call the relevant updating functions and increase
//              the time step. After the loop ends, we end the simulation.
//Parameters:   None
//Returns:      Nothing
//Calls:        verlet_position()
//              verlet_half_step_velocity()
//              verlet_acceleration()
//              verlet_velocity()
//              call_time_average_values()
//              end_time_step()
//              end_simulation()
//Globals:      simValue.totalTime 
//              simValue.timeStep
void begin_simulation() {
    
    int integerTimeStep = 0;
        
    for (double time = 0; time <= simValue.totalTime; time = time + simValue.timeStep) {
            
        verlet_half_step_velocity(time, simValue.timeStep);
        verlet_position(time, simValue.timeStep);
        verlet_acceleration(time);
        verlet_velocity(time, simValue.timeStep);
        verlet_periodic_position(time);
        
        call_time_average_values(integerTimeStep);
        call_csv_files(time, integerTimeStep);
        end_time_step();
            
        integerTimeStep = integerTimeStep + 1;
              
    } 
    
    end_simulation();

}//end of begin_simulation function
//******************************************************************************
//Function:     main
//Description:  Calls the relevant functions to run the simulation.  
//Parameters:   None
//Returns:      Nothing
//Calls:        simulation_values()
//              initial_conditions()
//              shock_equation()
//              mass_equation()
//              momentum_equation()
//              energy_equation()
//              stress_equation()
//              gruneisen_constant()
//              temperature_equation()
//              length_and_spacing()
//              eam_structure_array()
//              begin_simulation()
//Globals:      None
int main() {

    simulation_values();
    initial_conditions();
    shock_equation();
    mass_equation();
    momentum_equation();
    temperature_equation();
    length_and_spacing();
    eam_structure_array();

    begin_simulation(); 

}//end of main
//******************************************************************************
//******************************************************************************
//End of Program
