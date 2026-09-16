## Single Droplet Evaporation Validation
This case compares results from PeleLMeX against experimental literature data. Additional details and case descriptions are provided in the PelePhysics documentation at [https://pele-suite.github.io/PelePhysics/Spray.html#single-droplet-tests](https://pele-suite.github.io/PelePhysics/Spray.html#single-droplet-tests)

There are two general input files:
* `single-drop-evap.inp`
* `single-drop-evap-manifold.inp`
Both are set up for the `WongLin` case with the original PeleMP liquid properties model.  The non-manifold case can be run by compiling `make TPL && make -j`, then running
~~~
mpirun -np 4 ./PeleLMeX2d.<comp>.MPI.liquid_fuels_nonreacting.2SprayFuel.SprayMP.ex single-drop-evap.inp
~~~

Similarly, the case can be run with the group contribution method (GCM) liquid properties model by compiling with `SPRAY_FUEL_NUM = 2` and `SPRAY_GCM = TRUE`, then running:
~~~
mpirun -np 4 ./PeleLMeX2d.<comp>.MPI.liquid_fuels_nonreacting.2SprayFuel.SprayGCM.ex single-drop-evap.inp FILE=sprayPropsGCM_heptane-decane.inp
~~~

All cases provided in the [PelePhysics documentation]((https://pele-suite.github.io/PelePhysics/Spray.html#single-droplet-tests)) can be run using the `Validate.py` script. Because many of the necessary options for each case must be set at compile time, the `Validate.py` script can be used to compile the code with the necessary options if a valid executable does not exist already.
The relevant compile time options are indicated in the executable name. The following options are available with the `Validate.py` script:

| Option | Description | Choices/Defaults |
| :----: | :---------- | :--------------- |
| `-c`   | Case name   | **_WongLin_**, Nomura, Daif, RungeHep, <br> RungeDec, RungeMix, RungeJP8|
| `-l`   | Liquid properties model | **_gcm_**, mp |
| `-p`   | $P_{sat}$ model for PeleMP | **_Antoine_**, Clausius-Clapeyron (or CC) |
| `-m`   | Use Manifold chemistry/EOS | |
| `--cmlm_path` | Path to CMLM install for<br>Manifold table generation | Only needed with `-m` if installed outside<br> of SingleDropEvap |
| `-b`   | Build executable for case | |
| `-d`   | Don't run new simulation | Use to plot previously computed data |
| `-n`   | Number of processors for<br> parallel runs | Any machine-valid integer, default is **_6_** |

For example, the following command runs the *Daif* case with the *PeleMP* liquid properties model with the *Clasius-Clapeyron* model for estimating saturated vapor pressure on 4 processors
~~~
python Validate.py -c Daif -l mp -p CC -n 4 -b
~~~

Users can compare the results from the various tests/configurations by running the `CompareResults.py` script with the desired `case_name`. The script will search the current directory for all available data for each case. For example:
~~~
python CompareResults.py -c Daif
~~~

Note that multicomponent evaporation is a work in progress as illustrated by the `RungeJP8` test case.

### Droplet Evaporation with Manifold-based Chemistry Models

This case setup also supports single droplet evaporation validation for spray modeling capability coupled
to manifold based chemistry models. These models replace the EOS and Transport property models from
PelePhysics with tabulated (or neural network) reduced-order representations. These models require additional
files containing the tabulated data and associated metadata, which are generated with the separate [CMLM
repository](https://github.com/NatLabRockies/cmlm). For example purposes, we include the necessary files to run the
`WongLin` case (`spray_wonglin.ctb` and `manifold_metadata_wonglin.text`). To run the sample case, which uses
PeleMP liquid properties with Antoine coefficients, first compile with `USE_MANIFOLD=TRUE`, `Manifold_Dim=1`,
`SPRAY_FUEL_NUM=1` and `SPRAY_GCM=FALSE`. Then run:
~~~
mpirun -np 4 ./<PeleLMeX_EXE> single-drop-evap-mp-manifold.inp
~~~

The `Validate.py` script can also be used to run any of the cases with manifold-based chemistry. This requires an installed version of CMLM (including dependencies), which can be obtained within this directory using:
~~~
git clone git@github.com:NatLabRockies/cmlm.git
pip install -e cmlm
~~~
Then run `Validate.py` with the option `--use_manifold` (or the shorthand `-m`) and the appropriate `--cmlm_path` if downloaded elsewhere. For example: 
~~~
python Validate.py -b -c Daif -m -n 4 --cmlm_path $PATH_TO_CMLM
~~~
Note that the key script within CMLM that generates the tables for spray vaporization cases is located at
`cmlm/run_scripts/ctable/create_spray_table_nd.py`.