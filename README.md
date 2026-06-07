# Conformal Invariance and Schramm-Loewner Evolution
----------------------------------------------------

Adaptive Kennedy algorithm to generate SLE traces using the Vertical slit method. 

Based on Kennedy 2009 (https://arxiv.org/abs/0909.2438). 

See Kennedy_README.md for more comprehensive description.


----------------------------------------------------

A main.py script to control all the analysis scripts. 
Contain both field, geometrical and SLE analysis.
The scrpts for analysis and plotting are seperated.
Scripts are enabled by True/False. 

The isolines/contours are extracted from field using Extract_isolines.py
    Field analysis:
        Structure Factor
        Autocorrelation
        Radius of Gyration
        Field PDF

    Geometrical analysis of interfaces:
        Yardstick method
        Winding Statistics

    SLE analysis of interfaces:
    Driving function statistics and LPP require transformation onto the upper half plane.
    Used Schwarz-Christoffel transformation.
        Variance Driving function 
        Driving function PDF
        Left-Passage Probability
        Test for Markov process by Correlation

----------------------------------------------------

Script to animate phase fields data.

----------------------------------------------------

Ginzburg-Landau free energy, described by double well potential.

----------------------------------------------------

Field theory continuum model simmulations:
    Passive models:
        Model A (Cahn Hillard)
        Model B (Cahn Hillard)

    Active models:
        Active Model A (Cahn Hillard) 
        Active Model B (Cahn Hillard) 
        Active Model B+ (Cahn Hillard)
        Active Model AB (Cahn Hillard)  

        Active Model H (Cahn Hillard) 
        Active Model J (Cahn Hillard) 

        Potts Model B
        Toner-Tu Model
        KPZ Model

----------------------------------------------------

Conserved active models. All parallelized.
    Active Model B+ (Cahn Hillard)
    Active Model H (Cahn Hillard) 
    Potts Model B

    + a noise model

----------------------------------------------------

Voronoi model (SPV-model) to generate cell monolayer.

----------------------------------------------------

Generating percolation field. 

----------------------------------------------------

Left-Passage Probability(LPP) plot over different \kappa values.

----------------------------------------------------

Visualise frames from data. 

----------------------------------------------------