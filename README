Setup
-----

This project requires [Dune](dune-project.org). Run `duneproject` with the following answers:

	1) "masterarbeit"
	2) "dune-common dune-logging dune-uggrid dune-geometry dune-grid dune-localfunctions dune-istl dune-functions"
	
The project uses DUNE version 2.9.1-1 and Alberta 3.1.0-2 as well as Eigen 3.4.0-2

Remarks
-------

The program uses concept from C++ 20 and requires a compiler supporting this feature, e.g. GCC 14.2.1.
Furthermore, to speed up assembly, the routines use OpenMP, which also requires a fitting compiler (among them GCC).
Unfortunately, the Alberta grid manager yields an error, hence for this case OpenMP needs to be disabled either 
in the CMakeLists.txt file or to only use one thread by `export OMP_NUM_THREADS=1`.

In the setup section of the `main` method in `masterarbeit.cc` are some examples how to use the various objects. For 
the `GridMethod` there are the following options (`N` refers to the recommended first command line argument, see below):

	- `Square::Standard` with `2 <= N <= 500`,
	- `Triangle::Standard` with `2 <= N <= 500`,
	- `Triangle::RedGreen` with `2 <= N <= 11`,
	- `Triangle::NonDelaunay` with `2 <= N <= 18`,
	- `Triangle::Bisection` with `5 <= N <= 353`.

The program accepts 3 commandline arguments:

	1) an argument to define the grid fineness with the recommended options depending on the grid, see above,
	2) a value for the damping factor `omega`. Unsed for the Newton method, and used as the initial value if the adaptive omega is enabled,
	3) the value `gamma` in the stabilisation term.
