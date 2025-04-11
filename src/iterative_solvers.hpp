#pragma once

#include <iomanip>
#include <string>
#include <chrono>

#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/umfpack.hh>

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/UmfPackSupport>
#include <eigen3/unsupported/Eigen/SparseExtra>

#include "util.hpp"
#include "norms.hpp"

//Contains the 3 solver functions: the nonsmooth Newton method based on Eigen,
//and two fixed point functions doing the same but based on DUNE and Eigen,
//respectively. To vary the damping factor of the fixed point functions 
//there is a localOmegaAdaption method

//for the fixed-point methods, adapt the damping factor omega.
template < typename FltPrec >
bool localOmegaAdaption( FltPrec& localOmega, const FltPrec errOld, const FltPrec errNew ) {
	bool ret = false; //if true, the iteration is done again with the updated omega. if false, the step is accepted
	
	//disable adaption
	//~ return false;
	
	/*current strategy (more effective but harder to describe):
		- error increase: reduce omega by 25%
			- minimum omega depends on the region of the break condition:
			  error < 2e-7: 1e-5, else 1e-3
			  performs better from observations
			- if error increase exceeds 50% -> redo step with updated omega
		- error decrease by 1% -> increase omega by 10%. bound it by 1
	*/    
	constexpr char escape = 27;
	const std::string boldOn = "[1m";
	const std::string boldOff = "[0m";
	if(errNew  >= 1.0*errOld) {
		localOmega *= 0.75;
		if( localOmega < ( errOld < 2e-7 ? 1e-5 : 1e-3) ) {
			localOmega = ( errOld < 2e-7 ? 1e-5 : 1e-3);
			std::cout << escape << "[1m" << "\tLowered localOmega to the minimum " << localOmega << escape << "[0m" << std::endl;
			return false;
		}
		std::cout << escape << "[1m" << "\tLowered localOmega to " << localOmega << escape << "[0m";
		
		if(errNew >= 1.5*errOld) {
			std::cout << escape << "[1;31m" << " Step rejected." << escape << boldOff << std::endl;
			ret = true;
		}
		else {
			std::cout << std::endl;
			ret = false;
		}
	}
	else if( 0 < errOld - errNew and errNew / errOld < 1- 10e-2 ) {
		localOmega *= 1.1;
		if( localOmega > 1 )
			localOmega = 1;
		std::cout << escape << "[1;32m" << "\tIncreased localOmega to " << localOmega << escape << "[0m" << ", " << std::endl;
	}
	
	//------------------------------------------------------------------
	//strategy described in the thesis
	//	- reduce omega by 25% if the new error compared to the old error increased by at least 10%
	//		- redo the current step if this error exceeds 50%
	//	- increase omega by 10% if the new error is at least 1% better than the old one
	
	//~ if( errNew >= 1.1*errOld) {
		//~ localOmega *= 0.75;
		
		//~ if( errNew >= 1.5*errOld) {
			//~ ret = true;
		//~ }
	//~ }
	//~ else if( 0 < errOld - errNew and errNew < 0.99 * errOld ) {
		//~ localOmega *= 1.1;
	//~ }

	return ret;
}

//fixed point method to solve the nonlinear system; Eigen interpolation
//due to accuracy problems for fine grids (error increase from up to 1e8,
//fine grids is h=1/500 and smaller) not used
template < typename FltPrec, class Norm, class OutputMethod >
Eigen::Vector< FltPrec, Eigen::Dynamic > fixedpointMethod(
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& u0, //starting point = solution of usual Galerkin method
	const Eigen::SparseMatrix< FltPrec >& A, //stiffness matrix
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& Rhs, //righthand side
	const FltPrec omega, //damping factor
	const Eigen::Vector<FltPrec,Eigen::Dynamic> sVector, //representing the S-matrix
	const Eigen::Array<FltPrec,Eigen::Dynamic,1>& uKappaU, //vector representing the upper bound
	const Eigen::Array<FltPrec,Eigen::Dynamic,1>& uKappaL, //vector representing the lower bound
	const Norm L2Norm, //functor to calculate the L_2 norm
	const OutputMethod Output //functor to write any solution to a file
)
{
	std::cerr << "Fixedpoint Method with Eigen Interface" << std::endl;
	
	//enable / disable outputs in the loop of the iterative method
	constexpr bool DoOutput = false;
	
	const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	Eigen::Vector<FltPrec,Eigen::Dynamic> oldB(Rhs),newB(Rhs);
	Eigen::Vector<FltPrec,Eigen::Dynamic> u(u0);

	Eigen::SparseLU<Eigen::SparseMatrix<FltPrec>,Eigen::COLAMDOrdering<int> > solver;
	//~ Eigen::UmfPackLU<Eigen::SparseMatrix<FltPrec> > solver;
	
	if constexpr(DoOutput) Output(u0,std::ios::trunc,"output_u");
	if constexpr(DoOutput) Output(u0,std::ios::trunc,"output_uplus");

	const int MaxIterations = 10000;
	int n = MaxIterations;
	FltPrec localOmega = omega;
	
	solver.analyzePattern(A);
	solver.factorize(A);
	FltPrec l2errOld = std::numeric_limits<FltPrec>::infinity();
	do {
		const Eigen::Vector<FltPrec,Eigen::Dynamic> uplus = u.array().min(uKappaU).max(uKappaL).matrix();
		newB = A*u + localOmega*( Rhs - A*uplus - (sVector.array()*(u-uplus).array()).matrix() );
		const Eigen::Vector<FltPrec,Eigen::Dynamic> y = solver.solve(newB);
		const auto l2err = L2Norm((y-u).eval());
		if( localOmegaAdaption( localOmega, l2errOld, l2err ) ) continue;
		l2errOld = l2err;
		
		u = y.eval();
		
		if constexpr(DoOutput) Output(u, std::ios::app | std::ios::ate, "output_u");
		if constexpr(DoOutput) Output(uplus, std::ios::app | std::ios::ate, "output_uplus");
		
		std::cout << "Break-Condition: " << l2err << std::endl;
		if( std::isnan(l2err) or (--n <= 0) or l2err < 1e-8 ) break;
	}
	while( true );
	const auto fixedpointEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "Stopped after " << (MaxIterations - n) << " iterations (Max " << MaxIterations << ")." << std::endl;
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(fixedpointEnd-fixedpointStart).count() << " ms." << std::endl;
	
	return u;
}

//fixed point method using the DUNE framework
template < typename FltPrec, class Norm, class OutputMethod, typename VectorImpl > requires IsEnumeratable<VectorImpl>
Dune::BlockVector< FltPrec > fixedpointMethod(
	const Dune::BlockVector<FltPrec>& u0, //starting iteration = solution of the Galerkin approximation
	const Dune::BCRSMatrix< FltPrec >& A, //stiffness matrix
	const Dune::BlockVector<FltPrec>& Rhs, //load vector
	const FltPrec omega, //damping factor
	const Dune::BlockVector<FltPrec> sVector, //representing the S matrix
	const VectorImpl& uKappaU, //upper bounds at each lagrange node
	const VectorImpl& uKappaL, //lower bounds at each lagrange node
	const Norm L2Norm, //functor for L_2 norm calculation
	const OutputMethod Output  //functor to write a solution to a file
)
{
	std::cerr << "Fixedpoint Method with Dune Interface" << std::endl;
	
	//switch to enable / disable writing the results of each iteration
	//to a file
	constexpr bool DoOutput = false;
	
	using Vector = Dune::BlockVector<FltPrec>;
	using Matrix = Dune::BCRSMatrix<FltPrec>;
	
	if constexpr (DoOutput) Output(u0,std::ios::trunc,"output_u");
	if constexpr (DoOutput) Output(u0,std::ios::trunc,"output_uplus");

	const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	Dune::MatrixAdapter<Matrix,Vector,Vector> linearOperator(A);
	Dune::UMFPack<Matrix> solver(A, 0);

	// Object storing some statistics about the solving process
	Dune::InverseOperatorResult statistics;

	//x = u_h^n, y=u_h^{n+1}
	Vector 	x(u0.size()),
			uplus(u0.size()),
			uminus(u0),
			newB(u0.size()),//new rhs for the linear system
			y(u0.size());
	x = u0;
	
	FltPrec l2errOld = std::numeric_limits<FltPrec>::infinity();
	FltPrec localOmega = omega;

	const int MaxIterations = 10000;
	int n = MaxIterations;
	do {
		for( int i = 0; i < x.size(); i++ ) {
			uplus[i] = std::clamp(x[i],uKappaL[i],uKappaU[i]);
		}
		Vector aUplus(u0.size()), Au(u0.size());
		A.mv( x, Au );
		A.mv( uplus, aUplus );
		
		for( int i = 0; i < x.size(); i++ ) {
			uminus[i] = x[i] - uplus[i];
			newB[i] = Au[i] + localOmega * (Rhs[i] - aUplus[i] - sVector[i]*uminus[i]);
		}
		
		y = x;//start solving from the old position; probably unsed for direct solver
		solver.apply(y, newB, statistics);
		auto tmp(y); //to store the update u_h^{n+1} - u_h^n, needed for the break condition
		tmp -= x;
		auto l2err = L2Norm(tmp);
		std::cout << "Break-Condition (remaining: " << n << "): " << l2err << std::endl;
		
		if( localOmegaAdaption(localOmega,l2errOld,l2err) ) continue;
		l2errOld = l2err;
		
		if( std::isnan(l2err)or (--n <= 0) or l2err < 1e-8 ) break;
		x = y;
		
		if constexpr (DoOutput) {
			Output(x, std::ios::app | std::ios::ate, "output_u");
			Vector tmpUPlus(x.size());
			for( int i=0; i < tmpUPlus.size(); i++ ) {
				tmpUPlus[i] = std::clamp(y[i],uKappaL[i],uKappaU[i]);
			}
			if constexpr(DoOutput) {
				Output(tmpUPlus, std::ios::app | std::ios::ate, "output_uplus");
			}
		}
	}
	while( true );
	const auto fixedpointEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "Stopped after " << (MaxIterations - n) << " iterations (Max " << MaxIterations << ")." << std::endl;
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(fixedpointEnd-fixedpointStart).count() << " ms." << std::endl;
	
	return x;
}

//newton method to solve the nonlinear system
template < typename FltPrec, class OutputMethod >
Eigen::Vector<FltPrec,Eigen::Dynamic> newtonMethod(
	const Eigen::SparseMatrix<FltPrec>& A,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& b,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& u0,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& sVector,
	const Eigen::Array<FltPrec,Eigen::Dynamic,1> uKappaU, //upper bounds at each Lagrange node
	const Eigen::Array<FltPrec,Eigen::Dynamic,1> uKappaL,
	const auto normFunc,
	const OutputMethod Output
	)
{
	//enable / disable outputs of the iteration u_h^n and (u_h^n)^+
	constexpr bool doOutput = false;
	
	std::cerr << "Newton method" << std::endl;
	
	const auto newtonStart = std::chrono::high_resolution_clock::now();
	Eigen::Vector<FltPrec,Eigen::Dynamic> u(u0);
	//Bouligand section (V in the thesis), contains 0 and 1 on the diagonal
	//depending on whether the bounds are exceeded or not
	auto bouligand = Eigen::DiagonalMatrix<FltPrec,Eigen::Dynamic>(b.size());
	Eigen::SparseLU<Eigen::SparseMatrix<FltPrec>,Eigen::COLAMDOrdering<int> > solver;
	//use of umfpack showed worse results, although with DUNE it works
	//~ Eigen::UmfPackLU<Eigen::SparseMatrix<FltPrec> > solver;
	
	//-F from the thesis, eq. (3.3). we use -F(u) below
	const auto F = [&b,&A,&sVector,&uKappaU,&uKappaL](const Eigen::Matrix<FltPrec,Eigen::Dynamic,1>& u) {
		const auto uplus = u.array().min(uKappaU).max(uKappaL).matrix();
		return b - A*uplus - (sVector.array()*(u - uplus).array()).matrix();
	};
	
	int n = 0;
	const int MaxN = 1000;
	if constexpr (doOutput) Output( u, std::ios::trunc, "newton" );
	if constexpr (doOutput) Output( u.array().min(uKappaU).max(uKappaL).matrix(), std::ios::trunc, "newton_uplus" );
	do {
		if( n++ > MaxN ) break;
		
		bouligand.setIdentity();
		for( int i = 0; i < u.size(); i++ ) {
			//in the kink (non-differentiable points), choose either 0 or 1. 1st option: choose 0, second option (enabled): choose 1
			//~ if( u[i] < uKappaL[i]+std::numeric_limits<FltPrec>::epsilon() || u[i] > uKappaU[i]-std::numeric_limits<FltPrec>::epsilon() ) {
			if( u[i] < uKappaL[i]-std::numeric_limits<FltPrec>::epsilon() || u[i] > uKappaU[i]+std::numeric_limits<FltPrec>::epsilon() ) {
				bouligand.diagonal()[i] = 0;
			}
		}
		
		//way better (performance wise) implementation of (S - A) * bouligand - S = -A*bouligand + S*(bouligand - unitMatrix)
		//2nd summand only operates on the diagonal
		Eigen::SparseMatrix<FltPrec,Eigen::RowMajor> tmp = - A * bouligand;
		tmp.diagonal() += (sVector.array()*(bouligand.diagonal().array() - 1)).matrix();
		
		solver.analyzePattern(tmp);
		solver.factorize(tmp);
		const Eigen::Vector<FltPrec,Eigen::Dynamic> b = -F(u);
		const Eigen::Vector<FltPrec,Eigen::Dynamic> d = solver.solve(b);//calculate update
		
		const FltPrec l2err = normFunc(d);
		std::cerr << "Break condition: " << l2err << std::endl;
		if(l2err < 1e-8)
			break;
		
		u += d;
		if constexpr (doOutput) Output( u, std::ios::app | std::ios::ate, "newton" );
		if constexpr (doOutput) Output( u.array().min(uKappaU).max(uKappaL), std::ios::app | std::ios::ate, "newton_uplus" );
	} while( true );
	const auto newtonEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "Stopped after " << n << " iterations." << std::endl;
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(newtonEnd-newtonStart).count() << " ms." << std::endl;
	return u;
}
