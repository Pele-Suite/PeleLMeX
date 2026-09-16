# Taylor-Green Vortex (pulled from PeleC)

This setup is one of the test problems outlined by [the High-Order CFD workshop](https://www.grc.nasa.gov/hiocfd/).

A complete description of the problem can be found
[here](https://www.grc.nasa.gov/hiocfd/wp-content/uploads/sites/22/case_c3.3.pdf) and
the reference data is found
[here](https://www.grc.nasa.gov/wp-content/uploads/sites/22/C3.3_datafiles.zip). More
details of the problem and methods used to obtain the reference data
can be found in Bull and Jameson (2014) 7th AIAA Theoretical Fluid
Mechanics Conference (doi: 10.2514/6.2014-3210) and DeBonis (2013)
51st AIAA Aerospace Sciences Meeting (doi:10.2514/6.2013-382).

To directly plot the evolution of integrated kinetic energy and enstrophy, use
the data provided in temporals/tempState with the help of the processTGdata.py
script.

Note that these commands provide dimensional results, with t\* = t/(L\_0/V\_0),
E\_k\* = E\_k / ( rho\_0 * V\_0 * V\_0) and psi\* = psi / (V\_0/L\_0)^2

PeleLMeX results can be found [here](https://pele-suite.github.io/PeleLMeX/manual/html/Validation.html#taylor-green-vortex-breakdown).
