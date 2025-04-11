#include <config.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include <algorithm>
#include <functional>
#include <utility>

#include <dune/geometry/quadraturerules.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/umfpack.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/UmfPackSupport>
#include <eigen3/unsupported/Eigen/SparseExtra>

#include "util.hpp"
#include "grids.hpp"
#include "stabilisation_term.hpp"
#include "norms.hpp"
#include "assembly.hpp"
#include "iterative_solvers.hpp"

using namespace Dune;

int main(int argc, char *argv[])
{
	// Set up MPI, if available
	MPIHelper::instance(argc, argv);

	const double PI = StandardMathematicalConstants<double>::pi();
	constexpr int Dim = 2;//\Omega\subset\mathbb R^{dim}, \Omega=(0,1)^2 (see below, other sets/dimensions probably wont work, see below or e.g. getSVector)

	//-----------Problem Setup-----------
	const double mu = 1;
	const double eps = 1e-5;

	//Diffusion tensor
	const double diffusionInfinityNorm = eps; //currently, needs to be calculated by hand, if global data is used.
	const auto diffusion = [eps](const FieldVector<double,Dim>& x) -> const FieldMatrix<double,Dim,Dim> { return {{eps,0},{0,eps}}; };

	//convective flow field
	const double betaInfinityNorm = 2; //currently, needs to be calculated by hand, if global data is used.
	//inner layer example
	//~ const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {-x[1],x[0]}; };
	//error behaviour example
	//~ const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {2-x[0],2*x[0]}; };
	//unfitting bounds example
	const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {2,-1}; };
	//disable convection
	//~ const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {0,0}; };

	constexpr int LagrangeOrder = 1;
	using GridMethod = Square::Standard;

	//righthand sides
	//rhs for unfitting bounds example: u(x,y) = sin(pi*x)*sin(pi*y) with constant beta, D and mu
	auto const sourceTerm = [=](const FieldVector<double,Dim>& x) -> const double {return (2.0*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]) + beta({0,0})[0]*PI*cos(PI*x[0])*sin(PI*x[1])+beta({0,0})[1]*PI*sin(PI*x[0])*cos(PI*x[1]);};
	//rhs for inner layer (zero function)
	//~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x){return 0;};
	//rhs for error behaviour: solution u(x,y)=sin(pi*x^2)*sin(pi*y^3), beta=(2-x,2*y) and constant D = eps * unit matrix/mu
	//~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x) { return eps*(4*PI*PI*std::pow(x[0],2)*std::sin(PI*std::pow(x[0],2))*std::sin(PI*std::pow(x[1],3)) + 9*PI*PI*std::pow(x[1],4)*std::sin(PI*std::pow(x[0],2))*std::sin(PI*std::pow(x[1],3))) + (x[0]*(2-x[0])-eps)*2*PI*std::cos(PI*std::pow(x[0],2))*std::sin(PI*std::pow(x[1],3)) + (0.5*x[1]*2*x[0]-eps)*6*PI*x[1]*std::sin(PI*std::pow(x[0],2))*std::cos(PI*std::pow(x[1],3)) + mu*std::sin(PI*std::pow(x[0],2))*std::sin(PI*std::pow(x[1],3));};

	//correct solution functions and the derivative
	//unfitting bounds
	auto const f = [=] (const auto& coords) { return std::sin(PI*coords[0])*std::sin(PI*coords[1]); };
	auto const Df = [=](const auto& coords) -> FieldVector<double,Dim> { return { PI*std::cos(PI*coords[0])*std::sin(PI*coords[1]), PI*std::sin(PI*coords[0])*std::cos(PI*coords[1]) }; };
	//error behaviour
	//~ auto const f = [=] (const auto& coords) { return std::sin(PI*std::pow(coords[0],2))*std::sin(PI*std::pow(coords[1],3)); };
	//~ auto const Df = [=](const auto& coords) -> FieldVector<double,Dim> { return { 2*PI*coords[0]*std::cos(PI*std::pow(coords[0],2))*std::sin(PI*std::pow(coords[1],3)), 3*PI*std::pow(coords[1],2)*std::sin(PI*std::pow(coords[0],2))*std::cos(PI*std::pow(coords[1],3)) }; };

	//bounds: kappaU = upper bound, kappaL = lower bound
	//inner layer: local bounds
	//~ const auto kappaU = [=] (const FieldVector<double,2>& x) -> double { return x[0]*x[0]+x[1]*x[1] < 0.25 ? 0.5 : 1; };
	//~ const auto kappaL = [=] (const FieldVector<double,2>& x) -> double { return x[0]*x[0]+x[1]*x[1] < 0.25 ? 0 : 0.5; };
	//unfitting bounds: polynomial bound
	//~ const auto kappaU = [=] (const FieldVector<double,2>& coords) { return std::pow(coords[0]-0.5,2)+std::pow(coords[1]-0.5,2); };
	//unfitting bounds: sine hills
	//~ const auto kappaU = [=] (const FieldVector<double,2>& coords) { return 0.25*std::sin(4*PI*coords[0])*std::sin(4*PI*coords[1])+0.5; };
	//unfitting bounds: flat
	//~ const auto kappaU = [=] (const FieldVector<double,2>& x) -> double { return 0.5; };
	//normal bounds
	const auto kappaU = [=] (const FieldVector<double,2>& x) -> double { return 1; };
	const auto kappaL = [=] (const FieldVector<double,2>& x) -> double { return 0; };

	//(Dirichlet) boundary conditions
	//mark where they are
	auto predicate = [](const auto x)
	{
		//all examples except inner layer
		const bool ret = 1e-5 > x[0] || x[0] > 0.99999 || 1e-5 > x[1] || x[1] > 0.99999; //everywhere
		//inner layer example
		//~ const bool ret = x[0] > 0.99999 || 1e-5 > x[1];
		return ret;
	};

	// Set Dirichlet values
	auto dirichletValues = [](const auto x) -> const double
	{
		//all examples except inner layer
		return 0;
		//inner layer example
		if( x[0] < 0.333333 and x[1] < 1e-5) {
			return 0;
		}
		else if ( 0.333333 < x[0] and x[0] < 0.6666667 and x[1] < 1e-5) {
			return 0.5;
		}
		else {
			return 1;
		}
	};
	//---------------End Problem Setup-----------------

	//read command line arguments
	if( argc < 4 ) {
		std::cerr << argv[0] << " <Edges> <omega> <CIP-gamma>" << std::endl;
		return -1;
	}
	const unsigned int edges = std::atoi(argv[1]);
	const double omega = std::atof(argv[2]);
	const double gamma = std::atof(argv[3]);

	std::cerr 	<< "Configuration: " << std::endl
				<< "\t" << "Grid: " <<	(std::is_same_v<GridMethod,Square::Standard> ? "Squares" : "Triangles") << std::endl
				<< "\t" << "\t" << "Refinement: " << (std::is_same_v<GridMethod,Square::Standard> || std::is_same_v<GridMethod,Triangle::Standard> ? "Standard" : (std::is_same_v<GridMethod,Triangle::Bisection> ? "Bisection" : (std::is_same_v<GridMethod,Triangle::RedGreen> ? "RedGreen" : "Non-Delaunay"))) << std::endl
				<< "\t" << "Lagrange Elements" << std::endl
				<< "\t" << "\t" << "Order = " << LagrangeOrder << std::endl
				<< "\t" << "gamma = " << gamma << std::endl
				<< "\t" << "omega = " << omega << std::endl;

	//////////////////////////////////
	//   Generate the grid
	//////////////////////////////////

	std::cerr << "Generating Grid" << std::endl;

	auto grid = GridGenerator<GridMethod, Dim>::generate( {0,0}, {1,1}, edges );
	using Grid = GridGenerator<GridMethod, Dim>::GridType;

	using GridView = Grid::LeafGridView;
	GridView gridView = grid->leafGridView();

	//calculate grid constant. note that this is not the diameter but the smallest distance of two corners. some weird choice
	double H;
	if constexpr(std::is_same_v<GridMethod,Triangle::Bisection> or std::is_same_v<GridMethod,Triangle::CrissCross> or std::is_same_v<GridMethod,Triangle::RedGreen>) {
		H = ([&](const auto& geom) { 	auto result = (geom.corner(0) - geom.corner(1)).two_norm();
																	for( int i = 2; i < geom.corners(); i++ ) {
																		result = std::min( result, (geom.corner(0)-geom.corner(i)).two_norm());
																	}
																	return result;
		})(gridView.begin<0>()->geometry());
	} else {
		H = 1.0 / static_cast<double>(edges);
	};
	const double Diameter = diameter(gridView.begin<0>()->geometry());

	std::cerr << "||H = " << H << std::endl;
	gridlayoutToFile( gridView, H );

	std::cerr << "Generating Grid End" << std::endl;  

	/////////////////////////////////////////////////////////
	//   Stiffness matrix and right hand side vector
	/////////////////////////////////////////////////////////

	using Matrix = BCRSMatrix<double>;
	using Vector = BlockVector<double>;

	Matrix stiffnessMatrix;
	Vector b;

	/////////////////////////////////////////////////////////
	//   Assemble the system
	/////////////////////////////////////////////////////////

	Functions::LagrangeBasis<GridView,LagrangeOrder> basis(gridView);

	const auto sVector = getSVector( basis, diffusion, beta, diffusionInfinityNorm, betaInfinityNorm, mu );

	std::cerr << "Assemble Problem" << std::endl;
	assembleProblem(basis, stiffnessMatrix, b, sourceTerm, diffusion, beta, mu);
	std::cerr << "Assemble Problem End" << std::endl;

	addCIP(basis,stiffnessMatrix,beta,gamma);

	// Evaluating the predicate will mark all Dirichlet degrees of freedom
	std::vector<bool> dirichletNodes;
	Functions::interpolate(basis, dirichletNodes, predicate);

	///////////////////////////////////////////
	//   Modify Dirichlet rows
	///////////////////////////////////////////
	std::cerr << "Modify Dirichlet Rows" << std::endl;
	// Loop over the matrix rows
	for (size_t i=0; i<stiffnessMatrix.N(); i++)
	{
		if (dirichletNodes[i])
		{
			auto cIt    = stiffnessMatrix[i].begin();
			auto cEndIt = stiffnessMatrix[i].end();
			// Loop over nonzero matrix entries in current row
			for (; cIt!=cEndIt; ++cIt)
			*cIt = (cIt.index()==i) ? 1.0 : 0.0;
		}
	}

	Functions::interpolate(basis,b,dirichletValues, dirichletNodes);

	const Vector Rhs(b);
	std::cerr << "Modify Dirichlet Rows End" << std::endl;


	//prepare vectors for non-const bounds
	Eigen::Array<double,Eigen::Dynamic,1> uKappaU(b.size());
	Eigen::Array<double,Eigen::Dynamic,1> uKappaL(b.size());
	Functions::interpolate(basis,uKappaU,kappaU);
	Functions::interpolate(basis,uKappaL,kappaL);

	///////////////////////////
	//   Compute solution
	///////////////////////////

	// Choose an initial iterate that fulfills the Dirichlet conditions
	Vector x(basis.size());
	x = b;

	{
		std::cerr << "Solving" << std::endl;
		const auto solverStart = std::chrono::high_resolution_clock::now();
		Dune::UMFPack<Matrix> solver(stiffnessMatrix, 0);

		// Object storing some statistics about the solving process
		InverseOperatorResult statistics;

		// Solve!
		solver.apply(x, b, statistics);
		const auto solverEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\tTook: " << std::chrono::duration<float,std::milli>(solverEnd-solverStart).count() << " ms." << std::endl;
		std::cerr << "Solving End" << std::endl;
	}

	std::cerr << "(Dune) ||u_h^0-f||: h = " << H << ", " << L2Norm( gridView, basis.localView(), x, f) << std::endl;

	//-----------------------------------------------
	//Eigen-Version of solving	

	std::cerr << "Transcribe to Eigen" << std::endl;
	auto [stiffnessEigen, RhsEigen] = transcribeDuneToEigen( stiffnessMatrix, Rhs, 85 ); //85 for occupation should be enough for all grids and lagrange elements up to 3
	std::cerr << "Transcribe to Eigen End" << std::endl;

	const Eigen::Vector<double,Eigen::Dynamic> u0 = transcribeDuneToEigen(x);
	const Eigen::Vector<double,Eigen::Dynamic> eigenSVector = transcribeDuneToEigen(sVector);

	//~ Eigen::SparseLU<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int> > solver;
	Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver;
	solver.analyzePattern(stiffnessEigen);
	solver.factorize(stiffnessEigen);

	const Eigen::Vector<double,Eigen::Dynamic> uEigen = solver.solve(RhsEigen);
	std::cerr << "(Eigen) ||u_h^0-f||: h = " << H << ", " << L2Norm( gridView, basis.localView(), uEigen, f) << std::endl;
	outputVector<double>(basis,uEigen,std::ios::trunc, "test_output_eigen");

	//for fem-only testing
	//~ return 0;

	//--------------------------------------
	const auto L2NormBind = [&gridView,&basis](const auto& u) { return L2Norm( gridView, basis.localView(), u  ); };
	const auto OutputMethodBind = [&basis](const auto& u, const std::ios::openmode mode, const std::string filename) { return outputVector<double>( basis, u, mode, filename ); };

	//Newton-Method
	Eigen::Vector<double,Eigen::Dynamic> eigenU
		= newtonMethod( stiffnessEigen, RhsEigen, u0, eigenSVector,uKappaU, uKappaL, L2NormBind, OutputMethodBind );
	std::cerr << "H = " << H << std::endl;
	std::cerr << "(Newton|Eigen) ||u^+-f||_L2: " << L2Norm( gridView, basis.localView(), eigenU.array().min(uKappaU).max(uKappaL).matrix(), f) << std::endl;
	std::cerr << "(Newton|Eigen) ||u^+-f||_A = " << ANorm( gridView, basis.localView(), eigenU.array().min(uKappaU).max(uKappaL).matrix(), diffusion, mu, f, Df ) << std::endl;
	std::cerr << "(Newton|Eigen) ||u^+-f||_CIP = " << cipNorm( basis, eigenU.array().min(uKappaU).max(uKappaL).matrix(), diffusion, beta,mu, gamma, f, Df ) << std::endl;
	std::cerr << "(Newton|Eigen) ||u^-||_s = " << sNorm(eigenU - eigenU.array().min(uKappaU).max(uKappaL).matrix(), sVector, diffusionInfinityNorm,betaInfinityNorm,mu,Diameter) << std::endl;

	//Example: write the solution of the newton iteration to a file
	//~ {
		//~ outputVector<double>(basis,eigenU,std::ios::trunc,"newton_u");
		//~ outputVector<double>(basis,eigenU.array().min(uKappaU).max(uKappaL).matrix(),std::ios::trunc,"newton_uplus");
		//~ outputVector<double>(basis,eigenU - eigenU.array().min(uKappaU).max(uKappaL).matrix(),std::ios::trunc,"newton_uminus");
	//~ }

	//Example: generate the data for the support of u_h^- to a file (e.g. figs. 4.14 and 4.15, right picture)
	//~ {
		//~ Eigen::Vector<double,Eigen::Dynamic> tmp(eigenU - eigenU.array().min(uKappaU).max(uKappaL).matrix());
		
		//~ for( int i=0; i < tmp.size(); i++ ) {
			//~ if( std::abs(tmp[i]) < 1e-5 ) {
				//~ tmp[i] = 0;
			//~ }
			//~ else {
				//~ tmp[i] = 1;
			//~ }
		//~ }
		
		//~ outputVector<double>(basis,tmp,std::ios::trunc, "newton_uminus_amplified");
	//~ }

	//Example: calculate the indicators
	//~ {
		//~ //1st indicator: number of affected nodes
		//~ Eigen::Vector<double,Eigen::Dynamic> tmp(eigenU - eigenU.array().min(uKappaU).max(uKappaL).matrix());
		
		//~ auto predicate = [](const auto x)
		//~ {
			//~ const bool ret = 1e-5 > x[0] || x[0] > 0.99999 || 1e-5 > x[1] || x[1] > 0.99999; //everywhere
			//~ return ret ? 1 : 0;
		//~ };

		//~ // Evaluating the predicate will mark all Dirichlet degrees of freedom
		//~ std::vector<int> boundaryNodes;
		//~ Functions::interpolate(basis, boundaryNodes, predicate);
		
		//~ const int NoBoundaryNodes = noNnzElements(boundaryNodes);
		//~ const int NoNonzeroNodes = noNnzElements(tmp);
		//~ std::cerr << "No inner non-zero nodes u^-: " << NoNonzeroNodes << " of total " << (tmp.size() - NoBoundaryNodes) << " nodes, i.e." << (100*static_cast<double>(NoNonzeroNodes)/(tmp.size() - NoBoundaryNodes)) << "%." << std::endl;
		
		//~ //2nd indicator
		//~ Eigen::Vector<double,Eigen::Dynamic> tmp2((eigenSVector.array()*tmp.array()).matrix());
		//~ const auto [max,maxId] = inftyNorm( tmp2 );
		//~ const auto [maxUPlus,maxUPlusID] = inftyNorm((stiffnessEigen * eigenU.array().min(uKappaU).max(uKappaL).matrix()).eval());
		//~ std::cerr  << "u values, u vs u^+: " << max << " vs. " << maxUPlus << ", i.e. " << (100.0*max / maxUPlus) << "%." << std::endl;
	//~ }

	//for newton-only
	//~ return 0;

	//----------- Dune: fixed point method
	const auto normalsolution(x);

	x = fixedpointMethod( x, stiffnessMatrix, Rhs, omega, sVector, uKappaU, uKappaL, L2NormBind, OutputMethodBind );
	Vector uplus(x.size()), uminus(x.size());

	for( int i = 0; i < x.size(); i++ ) {
		uplus[i] = std::clamp(x[i],uKappaL[i],uKappaU[i]);
		uminus[i] = x[i] - uplus[i];
	}

	std::cerr << "||u_0-f||_2: h = " << H << ", " << L2Norm( gridView, basis.localView(), normalsolution, f) << std::endl;

	std::cerr << "(Richard|Dune) ||u^+-f||_L2 = " << L2Norm( gridView, basis.localView(), uplus, f) << std::endl;

	BlockVector<double> tmp;
	Functions::interpolate(basis,tmp,f);
	tmp -= uplus;

	std::cerr << "(Richard|Dune) ||u^+-f||_A = " << ANorm( gridView, basis.localView(), uplus, diffusion,mu, f, Df ) << std::endl;
	std::cerr << "(Richard|Dune) ||u^+-f||_CIP = " << cipNorm( basis, uplus, diffusion,beta,mu, gamma, f, Df ) << std::endl;
	std::cerr << "(Richard|Dune) ||u^-||_s = " << sNorm(uminus, sVector, diffusionInfinityNorm,betaInfinityNorm,mu,Diameter) << std::endl;

	return 0;
}
