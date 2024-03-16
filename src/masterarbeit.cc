#include <config.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>

#include <functional>
#include <utility>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/albertagrid.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bdmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/matrixmarket.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/IterativeLinearSolvers>

//~ #include <gmpxx.h>
//~ #include <eigen3/Eigen/Core>
//~ #include <boost/operators.hpp>

template < class T >
concept IsEnumeratable = requires(T v) {
	v[v.size()-1];
};

template <typename FltPrec, typename VectorImpl, typename Basis> requires IsEnumeratable<VectorImpl>
void outputVector(const Basis& basis, const VectorImpl& u, const std::ios::openmode mode = std::ios::app | std::ios::ate, std::string fileName = "output.txt") {
		Dune::BlockVector<double> xVals, yVals;
		Dune::Functions::interpolate(basis,xVals,[](auto x) { return x[0]; });
		Dune::Functions::interpolate(basis,yVals,[](auto x) { return x[1]; });
		
		assert(xVals.size() == yVals.size() && yVals.size() == u.size());
		
		std::ofstream outFile;
		outFile.open(fileName, mode | std::ios::out);
		if( mode & std::ios::app) {
			outFile << std::endl;
			outFile << std::endl;
		}
		
		for( int i = 0; i < u.size(); i++ ) {
			outFile << xVals[i] << '\t' << yVals[i] << '\t' << u[i] << std::endl;
		}
		outFile.close();
}

template < typename FltPrec, class Norm, class OutputMethod >
Eigen::Vector< FltPrec, Eigen::Dynamic > fixedpointMethod(
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& u0,
	const Eigen::SparseMatrix< FltPrec >& A,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& Rhs,
	const FltPrec omega,
	const FltPrec eps,
	const FltPrec H,
	const FltPrec UpperBound,
	const Norm L2Norm,
	const OutputMethod Output 
)
{
	std::cerr << "Fixedpoint Method with Eigen Interface" << std::endl;
	
	Eigen::Vector<FltPrec,Eigen::Dynamic> oldB(Rhs),newB(u0.size());
	Eigen::Vector<FltPrec,Eigen::Dynamic> u(u0);

	Eigen::ConjugateGradient<Eigen::SparseMatrix<FltPrec>,Eigen::Lower|Eigen::Upper> solver;
	solver.setTolerance(1e-9);
	solver.compute(A);
	
	Output(u0,std::ios::trunc,"output_u");
	Output(u0,std::ios::trunc,"output_uplus");

	const int MaxIterations = 300;
	int n = MaxIterations;
	const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	do {
		const double sFactor = eps + 2.0 * std::pow(H, 2.0);
		
		const Eigen::Vector<FltPrec,Eigen::Dynamic> uplus = u.cwiseMin(UpperBound).cwiseMax(0);
		newB = oldB + omega*( Rhs - A*uplus - sFactor*(u-uplus) );
		oldB = newB.eval(); //eval: avoid aliasing, see https://eigen.tuxfamily.org/dox/group__TopicAliasing.html
		
		const Eigen::Vector<FltPrec,Eigen::Dynamic> y = solver.solve(newB);
		
		
		//@Debug
		auto l2err = L2Norm((y-u).eval());
		std::cerr << "Break-Condition: " << l2err << std::endl;
		if( std::isnan(l2err)or (--n <= 0) or l2err < 1e-12 ) break;
		u = y.eval();
		Output(u, std::ios::app | std::ios::ate, "output_u");
		Output(uplus, std::ios::app | std::ios::ate, "output_uplus");
	}
	while( true );
	const auto fixedpointEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "Stopped after " << (MaxIterations - n) << " iterations (Max " << MaxIterations << ")." << std::endl;
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(fixedpointEnd-fixedpointStart).count() << " ms." << std::endl;
	
	return u;
}

template < typename FltPrec, class Norm, class OutputMethod >
Dune::BlockVector< FltPrec > fixedpointMethod(
	const Dune::BlockVector<FltPrec>& u0,
	const Dune::BCRSMatrix< FltPrec >& A,
	const Dune::BlockVector<FltPrec>& Rhs,
	const FltPrec omega,
	const FltPrec eps,
	const FltPrec H,
	const FltPrec UpperBound,
	const Norm L2Norm,
	const OutputMethod Output 
)
{
	std::cerr << "Fixedpoint Method with Dune Interface" << std::endl;
	
	using Vector = Dune::BlockVector<FltPrec>;
	using Matrix = Dune::BCRSMatrix<FltPrec>;
	
	Output(u0,std::ios::trunc,"output_u");
	Output(u0,std::ios::trunc,"output_uplus");

	Dune::MatrixAdapter<Matrix,Vector,Vector> linearOperator(A);
	// Sequential incomplete LU decomposition as the preconditioner
	Dune::SeqILU<Matrix,Vector,Vector> preconditioner(A,
										  1.0);  // Relaxation factor
	Dune::CGSolver<Vector> cg(linearOperator,
				  preconditioner,
				  1e-9, // Desired residual reduction factor
				  200,   // Maximum number of iterations
				  2);   // Verbosity of the solver

	// Object storing some statistics about the solving process
	Dune::InverseOperatorResult statistics;

	Vector 	x(u0.size()),
					uplus(u0.size()),
					uminus(u0),
					sVector(u0.size()),
					newB(u0.size()),
					y(u0.size());
	x = u0;
	
	const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	const int MaxIterations = 300;
	int n = MaxIterations;
	do {
		
		for( int i = 0; i < x.size(); i++ ) {
			uplus[i] = std::clamp(x[i],0.0,UpperBound);
		}
		Vector aUplus(u0.size()), Au(u0.size());
		A.mv( x, Au );
		A.mv( uplus, aUplus );
		const FltPrec sFactor = eps + 2.0 * std::pow(H, 2.0);
		
		for( int i = 0; i < x.size(); i++ ) {
			uminus[i] = x[i] - uplus[i];
			sVector[i] = sFactor * uminus[i];
			newB[i] = Au[i] + omega * (Rhs[i] - aUplus[i] - sVector[i]);
		}
		////@Debug
		//~ std::cout << "s-Multiplikator = " << sFactor << std::endl;
		//~ std::cout << "h = " << H << std::endl;
		//~ Vector diffRhsNewB(Rhs), diffXuPlus(x), diffAUAUplus(Au);
		//~ diffRhsNewB -= newB;
		//~ diffXuPlus -= uplus;
		//~ diffAUAUplus -= aUplus;
		
		//~ std::cout << "||u_minus|| = " << uminus.two_norm() << std::endl;
		//~ std::cout << "||sVector|| = " << sVector.two_norm() << std::endl;
		//~ auto [max, maxIdx] = inftyNorm( x );
		//~ std::cout << "||u||_\\infty = " << max << ",\tRhs = " << Rhs[maxIdx] << ",\tmatrix[idx,idx] = " << stiffnessMatrix[maxIdx][maxIdx] << std::endl;
		
		//~ std::cout	<< std::setw(15) << "x"
							//~ << std::setw(15) << "Au"
							//~ << std::setw(15) << "aUplus"
							//~ << std::setw(15) << "diff Au/Au+"
							//~ << std::setw(15) << "uminus"
							//~ << std::setw(15) << "uplus"
							//~ << std::setw(15) << "diff x/u+"
							//~ << std::setw(15) << "sVector"
							//~ << std::setw(15) << "Rhs"
							//~ << std::setw(15) << "newB"
							//~ << std::setw(15) << "diff"
							//~ << std::endl;
		//~ for( int i = 0; i < newB.size(); i++ ) {
			//~ std::cout << std::setw(15) << x[i]
								//~ << std::setw(15) << Au[i]
								//~ << std::setw(15) << aUplus[i]
								//~ << std::setw(15) << diffAUAUplus[i]
								//~ << std::setw(15) << uminus[i]
								//~ << std::setw(15) << uplus[i]
								//~ << std::setw(15) << diffXuPlus[i]
								//~ << std::setw(15) << sVector[i]
								//~ << std::setw(15) << Rhs[i]
								//~ << std::setw(15) << newB[i]
								//~ << std::setw(15) << diffRhsNewB[i]
								//~ << std::endl;
		//~ }
		y = x;

		// Solve!
		cg.apply(y, newB, statistics);
		
		//Dune
		auto tmp(y);
		tmp -= x;
		
		//@Debug
		auto l2err = L2Norm(tmp);
		std::cerr << "Break-Condition: " << l2err << std::endl;
		if( std::isnan(l2err)or (--n <= 0) or l2err < 1e-12 ) break;
		x = y;
		Output(x, std::ios::app | std::ios::ate, "output_u");
		Output(uplus, std::ios::app | std::ios::ate, "output_uplus");
	}
	while( true );
	const auto fixedpointEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "Stopped after " << (MaxIterations - n) << " iterations (Max " << MaxIterations << ")." << std::endl;
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(fixedpointEnd-fixedpointStart).count() << " ms." << std::endl;
	
	return x;
}

template < typename FltPrec, typename Basis>
Eigen::Vector<FltPrec,Eigen::Dynamic> newtonMethod(
	const Basis& basis,
	const Eigen::SparseMatrix<FltPrec>& A,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& b,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& u0,
	const FltPrec sFactor, const FltPrec a,
	const auto normFunc
	)
{
	Eigen::Vector<FltPrec,Eigen::Dynamic> u(u0);
	//~ const auto Identity = Eigen::Matrix<FltPrec,Eigen::Dynamic,Eigen::Dynamic>::Identity(b.size(),b.size());
	auto bouligand = Eigen::DiagonalMatrix<FltPrec,Eigen::Dynamic>(b.size());
	Eigen::BiCGSTAB<Eigen::SparseMatrix<FltPrec,Eigen::RowMajor> > solver;
	solver.setTolerance(1e-9);
	solver.setMaxIterations(1e3);
	
	const auto F = [&b,&A,a,sFactor](const Eigen::Matrix<FltPrec,Eigen::Dynamic,1>& u) {
		const auto uplus = u.cwiseMin(a).cwiseMax(0);
		return b - A*uplus - sFactor*(u - uplus);
	};
	
	
	int n = 0;
	std::cerr << "while loop" << std::endl;
		outputVector<FltPrec>( basis, u, std::ios::trunc, "newton" );
	do {
		n++;
		
		std::cerr << "\tLoop body" << std::endl;
		const auto setIdentity = std::chrono::high_resolution_clock::now();
		bouligand.setIdentity();
		const auto setIdentityEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\t\tSet Identity end: " << std::chrono::duration<float,std::milli>(setIdentityEnd-setIdentity).count() << std::endl;
		const auto updateBouligand = std::chrono::high_resolution_clock::now();
		for( int i = 0; i < u.size(); i++ ) {
			if( u[i] < -std::numeric_limits<FltPrec>::epsilon() || u[i] > a+std::numeric_limits<FltPrec>::epsilon() ) {
				bouligand.diagonal()[i] = 0;
			}
		}
		const auto updateBouligandEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\t\tUpdate bouligand end: " << std::chrono::duration<float,std::milli>(updateBouligandEnd-updateBouligand).count() << std::endl;
		const auto updateTmp = std::chrono::high_resolution_clock::now();
		
		//~ const Eigen::SparseMatrix<FltPrec> tmp = (sFactor * Identity - A) * bouligand - sFactor * Identity;
		Eigen::SparseMatrix<FltPrec,Eigen::RowMajor> tmp = - A * bouligand;
		tmp.diagonal() += (sFactor*(bouligand.diagonal().array() - 1)).matrix();
		const auto updateTmpEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\t\tUpdate tmp end: " << std::chrono::duration<float,std::milli>(updateTmpEnd - updateTmp).count() << std::endl;
		const auto compute = std::chrono::high_resolution_clock::now();
		solver.compute(tmp);
		const auto computeEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\t\tCompute end: " << std::chrono::duration<float,std::milli>(computeEnd - compute).count() << std::endl;
		const auto solve = std::chrono::high_resolution_clock::now();
		const Eigen::Vector<FltPrec,Eigen::Dynamic> d = solver.solve(-F(u));
		const auto solveEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\t\tSolve end: " << std::chrono::duration<float,std::milli>(solveEnd - solve).count() << " (" << solver.iterations() << " iterations / " << solver.error() << " )" << std::endl;
		
		const FltPrec l2err = normFunc(d);
		std::cerr << "Break condition: " << l2err << std::endl;
		if(l2err < 1e-12)
			break;
		
		const auto updateD = std::chrono::high_resolution_clock::now();
		u += d;
		const auto updateDEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\t\tUpdate u end: " << std::chrono::duration<float,std::milli>(updateDEnd - updateD).count() << std::endl;
		outputVector<FltPrec>( basis, u, std::ios::app | std::ios::ate, "newton" );
		std::cerr << "\tLoop body end" << std::endl;
	} while( true );
	std::cerr << "while loop end. " << n << " iterations." << std::endl;
	std::cout << u << std::endl;
	return u;
}

template < typename GridView >
void gridlayoutToFile( const GridView& gridView, const double H, const std::string filename = "gridlayout.gnuplot" ) {
	if( 1.0 / H > 20 ) return;
	
	std::ofstream gridLayout;
	gridLayout.open(filename, std::ios::trunc | std::ios::out);
	if( gridView.template begin<0>()->geometry().corners() != 4 ) {
		for( auto const& elem : elements(gridView ) ) {
			for( int i = 0; i < elem.geometry().corners(); i++ ) {
				const auto coordinate = elem.geometry().corner(i);
				
				gridLayout << coordinate << std::endl;
			}
			gridLayout << std::endl;
		}
	}
	else {
		for( auto const& elem : elements(gridView) ) {
			gridLayout << elem.geometry().corner(0) << std::endl;
			gridLayout << elem.geometry().corner(1) << std::endl;
			gridLayout << elem.geometry().corner(3) << std::endl;
			gridLayout << elem.geometry().corner(2) << std::endl;
			
			gridLayout << std::endl;
		}
	}
	gridLayout.close();
}

template < typename FltPrec, typename LocalView, typename GridView >
FltPrec ANorm( const GridView& gridView, LocalView localView,
	const Dune::BlockVector<FltPrec>& u,
	const Dune::FieldMatrix< FltPrec, LocalView::Element::dimension, LocalView::Element::dimension >& D,
	const FltPrec mu,
	const std::function<double(Dune::FieldVector<double,LocalView::Element::dimension>)> f,
	const std::function<Dune::FieldVector<double,LocalView::Element::dimension>(const Dune::FieldVector<double,LocalView::Element::dimension>)> Df
)
{
	constexpr int dim = LocalView::Element::dimension;
	
	FltPrec aNorm = 0;
	
	for( const auto& elem : elements(gridView) ) {
		localView.bind(elem);
		
		const auto& localFiniteElement = localView.tree().finiteElement();
		const int order = 4*localFiniteElement.localBasis().order();
		const auto& quadRule = Dune::QuadratureRules<FltPrec, dim>::rule(elem.type(), order);
		
		for( const auto& quadPoint : quadRule ) {
			const Dune::FieldVector<FltPrec,dim>& quadPos = quadPoint.position();
			
			const double integrationElement = elem.geometry().integrationElement(quadPos);
			
			std::vector<Dune::FieldVector<FltPrec,1> > shapeFunctionValues;
			localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues );
			
			const FltPrec functionValue = f(elem.geometry().global(quadPos));
			
			std::vector<Dune::FieldMatrix<FltPrec,1,dim> > referenceGradients;
			localFiniteElement.localBasis().evaluateJacobian(quadPos,
                                                     referenceGradients);

			const auto jacobian = elem.geometry().jacobianInverseTransposed(quadPos);
			//~ const auto jacobianT = elem.geometry().jacobianTransposed(quadPos);
			std::vector<Dune::FieldVector<FltPrec,dim> > gradients(referenceGradients.size());
			for (size_t i=0; i<gradients.size(); i++) {
				jacobian.mv(referenceGradients[i][0], gradients[i]);
			}
			
			//~ Dune::FieldVector<FltPrec,dim> gradf;
			//~ jacobianT.mv( Df( elem.geometry().global(quadPos) ), gradf );
			
			Dune::FieldVector<FltPrec,dim> gradSum; gradSum = 0;
			FltPrec shapeFunctionSum = 0;
			for( int i = 0; i < gradients.size(); i++ ) {
				const int globalIndex = localView.index(i);
				gradSum += u[globalIndex] * gradients[i];
				shapeFunctionSum += u[globalIndex] * shapeFunctionValues[i];
			}
			//~ gradSum -= gradf;
			gradSum -= Df( elem.geometry().global(quadPos) );
			Dune::FieldVector<FltPrec,dim> dGradSum;
			D.mv( gradSum, dGradSum );
			
			//~ std::cout << dGradSum * gradSum << '\t' << std::pow(shapeFunctionSum - functionValue,2) << '\t' << integrationElement << '\t' << quadPoint.weight() << std::endl;
			
			aNorm += quadPoint.weight() * integrationElement * ( dGradSum * gradSum + mu * std::pow(shapeFunctionSum - functionValue,2) );
		}
		
		
		localView.unbind();
	}
	
	return std::sqrt(aNorm);
}

//setting h=0 is a wrong value, but allows to omit the additional parameter,
//and is only neccessary for validating the general version
template < typename FltPrec, typename VectorImpl, typename LocalView, typename GridView > requires IsEnumeratable<VectorImpl>
FltPrec L2Norm( GridView gridView, LocalView localView, VectorImpl u, const FltPrec h = 0) {
	constexpr int dim = LocalView::Element::dimension;
	
	FltPrec l2err = 0;
	FltPrec l2err2 = 0;
	
	for( const auto& elem : elements(gridView) ) {
		localView.bind(elem);
		
		const auto& localFiniteElement = localView.tree().finiteElement();
		const int order = 4*localFiniteElement.localBasis().order();
		const auto& quadRule = Dune::QuadratureRules<FltPrec, dim>::rule(elem.type(), order);
		
		for( const auto& quadPoint : quadRule ) {
			const Dune::FieldVector<FltPrec,dim>& quadPos = quadPoint.position();
			
			const double integrationElement = elem.geometry().integrationElement(quadPos);
			std::vector<Dune::FieldVector<FltPrec,1> > shapeFunctionValues;
			localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues );

			double localU = 0;
			for( size_t p = 0; p < localFiniteElement.size(); p++ ) {
				const int globalIndex = localView.index(p);
				localU += shapeFunctionValues[p] * u[globalIndex];
			}
			l2err += std::pow(localU, 2) * quadPoint.weight() * integrationElement;
		}
		localView.unbind();
	}
	//only works for linear quadratic elements, usefull for comparison
	//~ for( const auto& elem : elements(gridView) ) {
		//~ localView.bind(elem);
		
		//~ const int noCorners = elem.geometry().corners();
		//~ for( int i = 0; i < noCorners; i++ ) {
			//~ const int localIndex = localView.tree().localIndex(i);
			//~ const int globalIndex = localView.index(i);
			//~ const auto coordinate = elem.geometry().corner(i);
			
			//~ l2err2 += std::pow(u[globalIndex], 2) * 0.25 * std::pow(h, 2);
		//~ }
		
		//~ localView.unbind();
	//~ }
	//~ std::cout << "||u||: l2err = " << std::setw(15) << l2err << ", l2err2 = " << std::setw(15) << l2err2 << std::endl;
	
	return std::sqrt(l2err);
}

template < typename FltPrec, typename VectorImpl, typename LocalView, typename GridView > requires IsEnumeratable<VectorImpl>
FltPrec L2Norm( GridView gridView, LocalView localView, const VectorImpl u,
								const std::function<double(Dune::FieldVector<double,LocalView::Element::dimension>)> f, const FltPrec h
) {
	constexpr int dim = LocalView::Element::dimension;
	
	FltPrec l2err = 0;
	FltPrec l2err2 = 0;
	
	for( const auto& elem : elements(gridView) ) {
		localView.bind(elem);
		
		const auto& localFiniteElement = localView.tree().finiteElement();
		const auto& order = 2 *localFiniteElement.localBasis().order();
		const auto& quadRule = Dune::QuadratureRules<FltPrec, dim>::rule(elem.type(), order);
		
		for( const auto& quadPoint : quadRule ) {
			const Dune::FieldVector<FltPrec,dim>& quadPos = quadPoint.position();
			
			const double integrationElement = elem.geometry().integrationElement(quadPos);
			std::vector<Dune::FieldVector<FltPrec,1> > shapeFunctionValues;
			localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues );
			
			const double functionValue = f(elem.geometry().global(quadPos));
			
			double localU = 0;
			for( size_t p = 0; p < localFiniteElement.size(); p++ ) {
				const int globalIndex = localView.index(p);
				localU += u[globalIndex] * shapeFunctionValues[p];
			}
			l2err += std::pow(localU - functionValue, 2) * quadPoint.weight() * integrationElement;
		}
		localView.unbind();
	}
	//~ for( const auto& elem : elements(gridView) ) {
		//~ localView.bind(elem);
		
		//~ const int noCorners = elem.geometry().corners();
		//~ for( int i = 0; i < noCorners; i++ ) {
			//~ const int localIndex = localView.tree().localIndex(i);
			//~ const int globalIndex = localView.index(i);
			//~ const auto coordinate = elem.geometry().corner(i);
			
			//~ l2err2 += std::pow(u[globalIndex] - f(coordinate), 2) * 0.25 * std::pow(h, 2);
		//~ }
		
		//~ localView.unbind();
	//~ }
	//~ std::cout << "||u-f||: l2err = " << std::setw(15) << l2err << ", l2err2 = " << std::setw(15) << l2err2 << std::endl;
	
	return std::sqrt(l2err);
}

template < typename FltPrec, int MantissaLength=500 >
std::pair< Eigen::SparseMatrix<FltPrec>, Eigen::Matrix<FltPrec,Eigen::Dynamic,1> >
transcribeDuneToEigen( Dune::BCRSMatrix<FltPrec> stiffnessMatrix, Dune::BlockVector< FltPrec > rhs, const int maxOccupationPerRow ) {
	Eigen::SparseMatrix< FltPrec > eigenMatrix( stiffnessMatrix.N(), stiffnessMatrix.M() );
	Eigen::Matrix< FltPrec, Eigen::Dynamic, 1 > eigenRhs( rhs.size() );
	
	//~ if( 1.0 / H > 20 ) return std::make_pair( eigenMatrix, eigenRhs );

	eigenMatrix.reserve(Eigen::VectorXi::Constant(stiffnessMatrix.M(),maxOccupationPerRow));

	const auto startLoop = std::chrono::high_resolution_clock::now();
	//~ for( size_t i=0; i < stiffnessMatrix.N(); i++ ) {
		//~ auto 			 cIt		= stiffnessMatrix[i].begin();
		//~ const auto cEndIt	= stiffnessMatrix[i].end();
		
		//~ for( ; cIt != cEndIt; ++cIt ) {
	for( auto row = stiffnessMatrix.begin(); row != stiffnessMatrix.end(); row++ ) {
		for( auto col = row->begin(); col != row->end(); col++ ) {
			//this loops over the whole matrix row, not only the non-zero entries
			//therefore checking
			if( std::abs(*col) < std::numeric_limits<double>::epsilon() ) continue;
			//~ if( std::abs(*cIt) < std::numeric_limits<double>::epsilon() ) continue;
			
			if constexpr (std::is_same_v< FltPrec, mpf_class >) {
				eigenMatrix.insert( row.index(), col.index() ) = mpf_class(*col, MantissaLength );
				//~ eigenMatrix.insert( i, cIt.index() ) = mpf_class(*cIt, MantissaLength );
			}
			else {
				eigenMatrix.insert( row.index(), col.index() ) = *col;
				//~ eigenMatrix.insert( i, cIt.index() ) = *cIt;
			}
		}
	}
	const auto startCompressing = std::chrono::high_resolution_clock::now();
	eigenMatrix.makeCompressed();
	const auto startRhs = std::chrono::high_resolution_clock::now();
	for( size_t i = 0; i < rhs.size(); i++ ) {
		if constexpr (std::is_same_v< FltPrec, mpf_class >) {
			eigenRhs[i]  = mpf_class( rhs[i], MantissaLength );
		}
		else {
			eigenRhs[i] = rhs[i];
		}
	}
	const auto done = std::chrono::high_resolution_clock::now();
	std::cerr << '\t' << "Matrix-Insert " << std::chrono::duration<float, std::milli>(startCompressing - startLoop).count() << "ms." << std::endl;
	std::cerr << '\t' << "Compressing " << std::chrono::duration<float, std::milli>(startRhs - startCompressing).count() << "ms." << std::endl;
	std::cerr << '\t' << "Rhs " << std::chrono::duration<float, std::milli>(done - startRhs).count() << "ms." << std::endl;
	
	return std::make_pair( eigenMatrix, eigenRhs );
}


	//Max. Element
template < typename FltPrec >
std::pair< FltPrec, size_t > inftyNorm( Dune::BlockVector< FltPrec > x ) {
	FltPrec max = std::abs(x[0]);
	size_t maxIdx = 0;
	
	for( int i = 1; i < x.size(); i++ ) {
		if( std::abs(x[i]) > max ) {
			max = std::abs(x[i]);
			maxIdx = i;
		}
	}
	
	return std::make_pair( max, maxIdx );
}

using namespace Dune;

// Compute the stiffness matrix for a single element
template<class LocalView, class Matrix, class Precision = typename Matrix::block_type>
void assembleElementStiffnessMatrix(const LocalView& localView,
                                    Matrix& elementMatrix,
                                    const FieldMatrix<Precision, LocalView::Element::dimension, LocalView::Element::dimension> D,
                                    const Precision mu)
{
  using Element = typename LocalView::Element;
  constexpr int dim = Element::dimension;
  auto element = localView.element();
  auto geometry = element.geometry();

  // Get set of shape functions for this element
  const auto& localFiniteElement = localView.tree().finiteElement();

  // Set all matrix entries to zero
  elementMatrix.setSize(localView.size(),localView.size());
  elementMatrix = 0;      // Fill the entire matrix with zeros

  // Get a quadrature rule
  int order = 4 * (localFiniteElement.localBasis().order());
  //@Debug
  //~ std::cout << "Order = " << order << std::endl;
  
  const auto& quadRule = QuadratureRules<Precision, dim>::rule(element.type(),
                                                            order);

  // Loop over all quadrature points
  for (const auto& quadPoint : quadRule)
  {

    // Position of the current quadrature point in the reference element
    const auto quadPos = quadPoint.position();

    // The transposed inverse Jacobian of the map from the reference element
    // to the grid element
    const auto jacobian = geometry.jacobianInverseTransposed(quadPos);

    // The determinant term in the integral transformation formula
    const auto integrationElement = geometry.integrationElement(quadPos);

    // The gradients of the shape functions on the reference element
    std::vector<FieldMatrix<Precision,1,dim> > referenceGradients;
    localFiniteElement.localBasis().evaluateJacobian(quadPos,
                                                     referenceGradients);

    // Compute the shape function gradients on the grid element
    // Dgradients: multiply the gradient w/ matrix D
    std::vector<FieldVector<Precision,dim> > gradients(referenceGradients.size());
    std::vector<FieldVector<Precision,dim> > Dgradients(referenceGradients.size());
    for (size_t i=0; i<gradients.size(); i++) {
      jacobian.mv(referenceGradients[i][0], gradients[i]);
      D.mv(gradients[i], Dgradients[i]);
		}
    
    //shape function values for the derivative-free term
    std::vector<FieldVector<Precision, 1>> shapeFunctionValues;  
    localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);
    
    //@Debug
    //~ std::vector<FieldVector<Precision, 1>> testShapeFunctionValues, testShapeFunctionValues2;  
    //~ localFiniteElement.localBasis().evaluateFunction(FieldVector<double,2>{0.5,0.5}, testShapeFunctionValues);
    //~ localFiniteElement.localBasis().evaluateFunction(FieldVector<double,2>{0.0,0.0}, testShapeFunctionValues2);
    //~ for( int i = 0; i < testShapeFunctionValues.size(); i++ ) {
			//~ std::cout << "i = " << i << '\t' << "testShapeFunctionValues[i] = " << testShapeFunctionValues[i] << '\t' << "testShapeFunctionValues2[i] = " << testShapeFunctionValues2[i] << std::endl;
		//~ }

    // Compute the actual matrix entries
    for (size_t p=0; p<elementMatrix.N(); p++)
    {
			//possibly different for vector-valued functions, i.e. localRow != p
      auto localRow = localView.tree().localIndex(p);
      assert(localRow == p);
      for (size_t q=0; q<elementMatrix.M(); q++)
      {
        auto localCol = localView.tree().localIndex(q);
        assert(localCol == q);
        elementMatrix[localRow][localCol] += (Dgradients[p] * gradients[q] + mu *  shapeFunctionValues[p] * shapeFunctionValues[q])
        //~ elementMatrix[localRow][localCol] += (gradients[p] * gradients[q] + mu *  shapeFunctionValues[p] * shapeFunctionValues[q])
                                    * quadPoint.weight() * integrationElement;
        //@Debug
        //~ if( p == 4 and q == 4 ) {
					//~ std::cout << "p=q=8:" << std::endl
										//~ << '\t'			<< "quadPos = " << quadPos << std::endl
										//~ << '\t'			<< "shapeFunctionVal[p] = " << shapeFunctionValues[p] << std::endl
										//~ << '\t'			<< "quadPoint.weight() = " << quadPoint.weight() << std::endl
										//~ << '\t'			<< "integrationElement = " << integrationElement << std::endl;
				//~ }
      }
    }
  }
  //@Debug
	//~ std::cout << "Matrix:" << std::endl;
	//~ for( int i = 0; i < elementMatrix.N(); i++ ) {
		//~ for( int j = 0; j < elementMatrix.M(); j++ ) {
			//~ std::cout << std::setw(12) << (elementMatrix.exists(i,j) ? elementMatrix[i][j] : 0);
		//~ }
		//~ std::cout << std::endl;
	//~ }
}


// Compute the source term for a single element
template<class LocalView>
void assembleElementVolumeTerm(
        const LocalView& localView,
        BlockVector<double>& localB,
        const std::function<double(FieldVector<double,
                                               LocalView::Element::dimension>)> volumeTerm)
{
  using Element = typename LocalView::Element;
  auto element = localView.element();
  constexpr int dim = Element::dimension;

  // Set of shape functions for a single element
  const auto& localFiniteElement = localView.tree().finiteElement();

  // Set all entries to zero
  localB.resize(localFiniteElement.size());
  localB = 0;

  // A quadrature rule
  int order = dim*20;
  const auto& quadRule = QuadratureRules<double, dim>::rule(element.type(), order);

  // Loop over all quadrature points
  for (const auto& quadPoint : quadRule)
  {
    // Position of the current quadrature point in the reference element
    const FieldVector<double,dim>& quadPos = quadPoint.position();

    // The multiplicative factor in the integral transformation formula
    const double integrationElement = element.geometry().integrationElement(quadPos);

    double functionValue = volumeTerm(element.geometry().global(quadPos));

    // Evaluate all shape function values at this point
    std::vector<FieldVector<double,1> > shapeFunctionValues;
    localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);

    // Actually compute the vector entries
    for (size_t p=0; p<localB.size(); p++)
    {
      auto localIndex = localView.tree().localIndex(p);
      localB[localIndex] += shapeFunctionValues[p] * functionValue
                          * quadPoint.weight() * integrationElement;
    }
  }
}

// Get the occupation pattern of the stiffness matrix
template<class Basis>
void getOccupationPattern(const Basis& basis, MatrixIndexSet& nb)
{
  nb.resize(basis.size(), basis.size());

  auto gridView = basis.gridView();

  // A loop over all elements of the grid
  auto localView = basis.localView();

  for (const auto& element : elements(gridView))
  {
    localView.bind(element);

    for (size_t i=0; i<localView.size(); i++)
    {
      // The global index of the i-th vertex of the element
      auto row = localView.index(i);

      for (size_t j=0; j<localView.size(); j++ )
      {
        // The global index of the j-th vertex of the element
        auto col = localView.index(j);
        nb.add(row,col);
      }
    }
  }
}

template<typename Basis, typename Element>
void processElement(const Basis& basis,
														Element& element,
                            BCRSMatrix<double>& matrix,
                            BlockVector<double>& b,
                            const std::function<
                                double(FieldVector<double,
                                                   Basis::GridView::dimension>)
                                               > volumeTerm, 
                            const FieldMatrix<double, 2, 2> D, //Vektorwertigkeit unschön. Basis::GridView::dimension?
                            const double mu)
{
	auto localView = basis.localView();
	localView.bind(element);

	Matrix<double> elementMatrix;
	assembleElementStiffnessMatrix(localView, elementMatrix, D, mu);

	for(size_t p=0; p<elementMatrix.N(); p++)
	{
		// The global index of the p-th degree of freedom of the element
		auto row = localView.index(p);

		for (size_t q=0; q<elementMatrix.M(); q++ )
		{
			// The global index of the q-th degree of freedom of the element
			auto col = localView.index(q);
			#pragma omp atomic
			matrix[row][col] += elementMatrix[p][q];
		}
	}

	// Now get the local contribution to the right-hand side vector
	BlockVector<double> localB;
	assembleElementVolumeTerm(localView, localB, volumeTerm);

	for (size_t p=0; p<localB.size(); p++)
	{
		// The global index of the p-th vertex of the element
		auto row = localView.index(p);
		#pragma omp atomic
		b[row] += localB[p];
	}
}

template<class Basis>
void assembleProblem(const Basis& basis,
                            BCRSMatrix<double>& matrix,
                            BlockVector<double>& b,
                            const std::function<
                                double(FieldVector<double,
                                                   Basis::GridView::dimension>)
                                               > volumeTerm, 
                            const FieldMatrix<double, 2, 2> D, //Vektorwertigkeit unschön. Basis::GridView::dimension?
                            const double mu)
{
  auto gridView = basis.gridView();

  // MatrixIndexSets store the occupation pattern of a sparse matrix.
  // They are not particularly efficient, but simple to use.
  MatrixIndexSet occupationPattern;
  getOccupationPattern(basis, occupationPattern);
  occupationPattern.exportIdx(matrix);

  // Set all entries to zero
  matrix = 0;

  // Set b to correct length
  b.resize(basis.dimension());

  // Set all entries to zero
  b = 0;

  // A loop over all elements of the grid
  auto localView = basis.localView();
  //@Test: as the elementMatrix stays the same for some grids, dont
  //recreate it each time
  //~ Matrix<double> elementMatrix;
  //~ const auto firstElement = *(gridView.template begin< 0 >());
  //~ localView.bind(firstElement);
  //~ assembleElementStiffnessMatrix(localView, elementMatrix, D, mu);

	std::cerr << "basis.dimension() = " << basis.dimension() << std::endl;

	const auto LastElement = gridView.template end<0>();
	
  //~ for (const auto& element : elements(gridView))
	#pragma omp parallel if(basis.dimension() > 1e4)
	#pragma omp single
	{
		for(auto element = gridView.template begin<0>(); element != LastElement; element++)
		//~ {
			#pragma omp task default(none) firstprivate(element) shared(D,mu,matrix,b,basis,volumeTerm)
				processElement( basis, *element,matrix,b,volumeTerm,D,mu );
			// Now let's get the element stiffness matrix
			// A dense matrix is used for the element stiffness matrix
			//~ localView.bind(element);

			//~ Matrix<double> elementMatrix;
			//~ assembleElementStiffnessMatrix(localView, elementMatrix, D, mu);

			//~ for(size_t p=0; p<elementMatrix.N(); p++)
			//~ {
				//~ // The global index of the p-th degree of freedom of the element
				//~ auto row = localView.index(p);

				//~ for (size_t q=0; q<elementMatrix.M(); q++ )
				//~ {
					//~ // The global index of the q-th degree of freedom of the element
					//~ auto col = localView.index(q);
					//~ #pragma omp atomic
					//~ matrix[row][col] += elementMatrix[p][q];
				//~ }
			//~ }

			//~ // Now get the local contribution to the right-hand side vector
			//~ BlockVector<double> localB;
			//~ assembleElementVolumeTerm(localView, localB, volumeTerm);

			//~ for (size_t p=0; p<localB.size(); p++)
			//~ {
				//~ // The global index of the p-th vertex of the element
				//~ auto row = localView.index(p);
				//~ #pragma omp atomic
				//~ b[row] += localB[p];
			//~ }
		//~ }
		#pragma omp taskwait
	}
}

class Triangle {
public:
	struct Bisection {};
	using CrissCross = Bisection;
	struct RedGreen {};
	struct Standard {};
};
class Square {
public:
	struct Standard {};
};

template < typename T, int dim>
class GridGenerator {};

template <int dim>
class GridGenerator<Square::Standard, dim> {
public:
	using GridType = Dune::YaspGrid<dim>;

	template <typename U = double>
	static
	auto generate(const std::array<U,dim> lowerLeftCorner, const std::array<U,dim> upperRightCorner, const uint edgeNumber) {
		return std::make_shared<GridType>( /*Dune::FieldVector<U,2>{lowerLeftCorner[0],lowerLeftCorner[1]},*/ //uncommenting breaks compiling, for an unkown reason
																				Dune::FieldVector<U,2>{upperRightCorner[0],upperRightCorner[1]},
																				std::array<int,dim>{ edgeNumber, edgeNumber } );
	}
};
template <int dim>
class GridGenerator<Triangle::Bisection, dim> {
public:
	using GridType = Dune::AlbertaGrid<dim,dim>;

	template < typename U = double >
	static
	auto generate(const std::array<U,dim> lowerLeftCorner, const std::array<U,dim> upperRightCorner, const uint edgeNumber) {
		auto grid = StructuredGridFactory<GridType>::createSimplexGrid( {lowerLeftCorner[0],lowerLeftCorner[1]}, {upperRightCorner[0],upperRightCorner[1]}, {1,1} );
		grid->globalRefine(edgeNumber);
		
		return grid;
	}
};
template <int dim>
class GridGenerator<Triangle::Standard, dim> {
public:
	using GridType = Dune::UGGrid<dim>;

	template < typename U = double >
	static
	auto generate(const std::array<U,dim> lowerLeftCorner, const std::array<U,dim> upperRightCorner, const uint edgeNumber) {
		auto grid = StructuredGridFactory<GridType >::createSimplexGrid( {lowerLeftCorner[0],lowerLeftCorner[1]}, {upperRightCorner[0],upperRightCorner[1]}, {edgeNumber,edgeNumber} );
		
		return grid;
	}
};
template <int dim>
class GridGenerator<Triangle::RedGreen, dim> {
public:
	using GridType = Dune::UGGrid<dim>;
	
	template < typename U = double >
	static
	auto generate(const std::array<U,dim> lowerLeftCorner, const std::array<U,dim> upperRightCorner, const uint edgeNumber) {
		//generate criss cross layout
		GridFactory<GridType> factory;
		factory.insertVertex({lowerLeftCorner[0]													,lowerLeftCorner[1]														});
		factory.insertVertex({0.5*(lowerLeftCorner[0]+upperRightCorner[0]),lowerLeftCorner[1]														});
		factory.insertVertex({upperRightCorner[0]													,lowerLeftCorner[1]														});
		factory.insertVertex({lowerLeftCorner[0]													, 0.5*(lowerLeftCorner[1]+upperRightCorner[1])});
		factory.insertVertex({0.5*(lowerLeftCorner[0]+upperRightCorner[0]), 0.5*(lowerLeftCorner[1]+upperRightCorner[1])});
		factory.insertVertex({upperRightCorner[0]													, 0.5*(lowerLeftCorner[1]+upperRightCorner[1])});
		factory.insertVertex({lowerLeftCorner[0]													, upperRightCorner[1]													});
		factory.insertVertex({0.5*(lowerLeftCorner[0]+upperRightCorner[0]), upperRightCorner[1]													});
		factory.insertVertex({upperRightCorner[0]													, upperRightCorner[1]													});
		
		factory.insertElement(GeometryTypes::simplex(2), {3,0,4});
		factory.insertElement(GeometryTypes::simplex(2), {0,1,4});
		factory.insertElement(GeometryTypes::simplex(2), {1,2,4});
		factory.insertElement(GeometryTypes::simplex(2), {2,5,4});
		factory.insertElement(GeometryTypes::simplex(2), {4,7,6});
		factory.insertElement(GeometryTypes::simplex(2), {3,4,6});
		factory.insertElement(GeometryTypes::simplex(2), {4,5,8});
		factory.insertElement(GeometryTypes::simplex(2), {4,8,7});
		
		auto grid = factory.createGrid();
		if( edgeNumber > 2)
			grid->globalRefine(edgeNumber-2);
		
		return grid;
	}
};

int main(int argc, char *argv[])
{
  // Set up MPI, if available
  MPIHelper::instance(argc, argv);

  const double PI = StandardMathematicalConstants<double>::pi();

  const double mu = 1;
  const double eps = 1e-5;
  const FieldMatrix<double, 2, 2> D = {{eps,0},{0,eps}};
  //~ const int edges = 100;
  if( argc < 4 ) {
		std::cerr << argv[0] << " <Edges> <omega> <upper bound>" << std::endl;
		return -1;
	}
  const unsigned int edges = std::atoi(argv[1]);
  const double omega = std::atof(argv[2]);
  const double UpperBound = std::atof(argv[3]);

  //////////////////////////////////
  //   Generate the grid
  //////////////////////////////////

	std::cerr << "Generating Grid" << std::endl;
  constexpr int dim = 2;
  using GridMethod = Triangle::Standard;
  
  auto grid =	 GridGenerator<GridMethod, dim>::generate( {0,0}, {1,1}, edges );
  using Grid = GridGenerator<GridMethod, dim>::GridType;

  using GridView = Grid::LeafGridView;
  GridView gridView = grid->leafGridView();
  
  const double H = ([&](const auto& geom) { auto result = (geom.corner(0) - geom.corner(1)).two_norm();
																						for( int i = 2; i < geom.corners(); i++ ) {
																							result = std::min( result, (geom.corner(0)-geom.corner(i)).two_norm());
																						}
																						return result;
	})(gridView.begin<0>()->geometry());
  std::cerr << "H = " << H << std::endl;
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

  Functions::LagrangeBasis<GridView,1> basis(gridView);

  auto sourceTerm = [=](const FieldVector<double,dim>& x){return (2.0*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]);};
  std::cerr << "Assemble Problem" << std::endl;
  assembleProblem(basis, stiffnessMatrix, b, sourceTerm, D, mu);
  std::cerr << "Assemble Problem End" << std::endl;

  // Determine Dirichlet dofs by marking all degrees of freedom whose Lagrange nodes
  // comply with a given predicate.
  auto predicate = [](auto x)
  {
		const bool ret = 1e-5 > x[0] || x[0] > 0.99999 || 1e-5 > x[1] || x[1] > 0.99999;
		//@Debug
		//~ std::cout << "x = " << x << " : " << (ret ? 1 : 0) << std::endl;
    return ret;
  };
 

  // Evaluating the predicate will mark all Dirichlet degrees of freedom
  std::vector<bool> dirichletNodes;
  Functions::interpolate(basis, dirichletNodes, predicate);
  //@Debug
  std::cout << "Dirichlet-Nodes: ";
  for( auto const& val :  dirichletNodes) std::cout << (val ? 1 : 0) << " ";
  std::cout << std::endl;

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

  // Set Dirichlet values
  auto dirichletValues = [](auto x)
  {
    return 0;
  };
  Functions::interpolate(basis,b,dirichletValues, dirichletNodes);
    
  const Vector Rhs(b);
  std::cerr << "Modify Dirichlet Rows End" << std::endl;

  ///////////////////////////
  //   Compute solution
  ///////////////////////////

  // Choose an initial iterate that fulfills the Dirichlet conditions
  Vector x(basis.size());
  x = b;

	{
		std::cerr << "Solving" << std::endl;
  // Turn the matrix into a linear operator
  MatrixAdapter<Matrix,Vector,Vector> linearOperator(stiffnessMatrix);

  // Sequential incomplete LU decomposition as the preconditioner
  SeqILU<Matrix,Vector,Vector> preconditioner(stiffnessMatrix,
                                              1.0);  // Relaxation factor

  // Preconditioned conjugate gradient solver
  CGSolver<Vector> cg(linearOperator,
                      preconditioner,
                      1e-5, // Desired residual reduction factor
                      20000,   // Maximum number of iterations
                      2);   // Verbosity of the solver

  // Object storing some statistics about the solving process
  InverseOperatorResult statistics;

  // Solve!
  cg.apply(x, b, statistics);
  std::cerr << "Solving End" << std::endl;
	}
	
	//-----------------------------------------------
	//Eigen-Version of solving	
	
	//~ using FLTPrec = mpf_class;
	std::cerr << "Transcribe to Eigen" << std::endl;
	using FLTPrec = double;
	auto [stiffnessEigen, RhsEigen] = transcribeDuneToEigen( stiffnessMatrix, Rhs, 25 ); //25 for occupation should be enough for all grids and lagrange elements up to 2
	std::cerr << "Transcribe to Eigen End" << std::endl;
	
	const Eigen::Vector<double,Eigen::Dynamic> u0 = ([&x]() {
		Eigen::Vector<double,Eigen::Dynamic> u0(x.size());
		for( int i = 0; i < x.size(); i++ ) { u0[i] = x[i]; }
		return u0;
	})();
	
	//Newton-Method
	const auto newtonStart = std::chrono::high_resolution_clock::now();
	Eigen::Vector<double,Eigen::Dynamic> eigenU
			//~ (x.size());
		= newtonMethod<double>( basis, stiffnessEigen, RhsEigen, u0, eps + 2.0 * std::pow(H, 2.0), UpperBound, [&gridView,&basis](const auto& u) { return L2Norm<double>(gridView,basis.localView(),u);} );
	const auto newtonEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(newtonEnd-newtonStart).count() << " ms." << std::endl;
	auto const f = [=] (const auto& coords) { return std::sin(PI*coords[0])*std::sin(PI*coords[1]); };
	std::cerr << "H = " << H << std::endl;
	std::cerr << "||u^+-f||_L2: " << L2Norm( gridView, basis.localView(), eigenU.cwiseMin(UpperBound).cwiseMax(0), f, H) << std::endl;
	
	
	//~ return 0;
	
	//~ Eigen::SparseLU<Eigen::SparseMatrix<FLTPrec>> solverEigen;
	//~ solverEigen.analyzePattern(stiffnessEigen);
	//~ solverEigen.factorize(stiffnessEigen);
	//~ Eigen::Matrix<FLTPrec,Eigen::Dynamic,1> u = solverEigen.solve(RhsEigen);
	//~ Eigen::Matrix<FLTPrec,Eigen::Dynamic,1> u(RhsEigen);
	std::cout	<< std::setw(20) << "Eigen solution"
						<< std::setw(20) << "Dune solution"
						<< std::setw(20) << "Eigen-Dune"
						<< std::endl;
	for( int i=0; i < x.size(); i++ ) {
		std::cout	
							//~ << std::setw(20) << u[i]
							<< std::setw(20) << x[i]
							//~ << std::setw(20) << (u[i]-x[i])
							<< std::endl;
	}
	
	if( 1.0 / H <= 20 ) {
		std::cout << stiffnessEigen << std::endl;
		std::cout << RhsEigen << std::endl;
	}
	
	//-----------
	const auto L2NormBind = [&gridView,&basis](const auto& u) { return L2Norm<double>( gridView, basis.localView(), u  ); };
	const auto OutputMethodBind = [&basis](const auto& u, const std::ios::openmode mode, const std::string filename) { return outputVector<double>( basis, u, mode, filename ); };
	Eigen::Vector<double,Eigen::Dynamic> eigenX = fixedpointMethod( u0, stiffnessEigen, RhsEigen, omega, eps, H, UpperBound, L2NormBind, OutputMethodBind );
	
	//-----------
	const auto normalSolution(x);
	
	std::cerr << "||eigenX-f||: h = " << H << ", " << L2Norm( gridView, basis.localView(), eigenX, [=](const auto& coords) { return std::sin(PI*coords[0])*std::sin(PI*coords[1]); }, H) << std::endl;
	std::cerr << "||normalsolution-f||: h = " << H << ", " << L2Norm( gridView, basis.localView(), normalSolution, [=](const auto& coords) { return std::sin(PI*coords[0])*std::sin(PI*coords[1]); }, H) << std::endl;
	
	//uncomment for error checking of usual fem
	//~ return 0;
	
	x = fixedpointMethod( normalSolution, stiffnessMatrix, Rhs, omega, eps, H, UpperBound, L2NormBind, OutputMethodBind );
	Vector uplus(x.size()), uminus(x.size());
	
	for( int i = 0; i < x.size(); i++ ) {
		uplus[i] = std::clamp(x[i],double(0),UpperBound);
		uminus[i] = x[i] - uplus[i];
	}
	
	//~ outputVector<double>(basis,normalSolution,std::ios::trunc,"output_u");
	//~ outputVector<double>(basis,normalSolution,std::ios::trunc,"output_uplus");
	
	//-----------------
	//Vector uplus(basis.size());
	//Vector uminus(x);
	//Vector sVector(basis.size());
	//Vector newB(basis.size());
	//Vector y(basis.size());

	////~ {
		////~ auto [max, maxIdx] = inftyNorm( x );
		////~ std::cout << "||u||_\\infty = " << max << ",\tRhs = " << Rhs[maxIdx] << ",\tmatrix[idx,idx] = " << stiffnessMatrix[maxIdx][maxIdx] << std::endl;
	////~ }
	
	
	//const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	//const int MaxIterations = 300;
	//int n = MaxIterations;
	//do {
		
		//for( int i = 0; i < x.size(); i++ ) {
			//uplus[i] = std::clamp(x[i],0.0,UpperBound);
		//}
		//Vector aUplus(basis.size()), Au(basis.size());
		//stiffnessMatrix.mv( x, Au );
		//stiffnessMatrix.mv( uplus, aUplus );
		//const double sFactor = eps + 2.0 * std::pow(H, 2.0);
		
		//for( int i = 0; i < x.size(); i++ ) {
			//uminus[i] = x[i] - uplus[i];
			//sVector[i] = sFactor * uminus[i];
			//newB[i] = Au[i] + omega * (Rhs[i] - aUplus[i] - sVector[i]);
		//}
		
		////@Debug
		////~ std::cout << "s-Multiplikator = " << sFactor << std::endl;
		////~ std::cout << "h = " << H << std::endl;
		////~ Vector diffRhsNewB(Rhs), diffXuPlus(x), diffAUAUplus(Au);
		////~ diffRhsNewB -= newB;
		////~ diffXuPlus -= uplus;
		////~ diffAUAUplus -= aUplus;
		
		////~ std::cout << "||u_minus|| = " << uminus.two_norm() << std::endl;
		////~ std::cout << "||sVector|| = " << sVector.two_norm() << std::endl;
		////~ auto [max, maxIdx] = inftyNorm( x );
		////~ std::cout << "||u||_\\infty = " << max << ",\tRhs = " << Rhs[maxIdx] << ",\tmatrix[idx,idx] = " << stiffnessMatrix[maxIdx][maxIdx] << std::endl;
		
		////~ std::cout	<< std::setw(15) << "x"
							////~ << std::setw(15) << "Au"
							////~ << std::setw(15) << "aUplus"
							////~ << std::setw(15) << "diff Au/Au+"
							////~ << std::setw(15) << "uminus"
							////~ << std::setw(15) << "uplus"
							////~ << std::setw(15) << "diff x/u+"
							////~ << std::setw(15) << "sVector"
							////~ << std::setw(15) << "Rhs"
							////~ << std::setw(15) << "newB"
							////~ << std::setw(15) << "diff"
							////~ << std::endl;
		////~ for( int i = 0; i < newB.size(); i++ ) {
			////~ std::cout << std::setw(15) << x[i]
								////~ << std::setw(15) << Au[i]
								////~ << std::setw(15) << aUplus[i]
								////~ << std::setw(15) << diffAUAUplus[i]
								////~ << std::setw(15) << uminus[i]
								////~ << std::setw(15) << uplus[i]
								////~ << std::setw(15) << diffXuPlus[i]
								////~ << std::setw(15) << sVector[i]
								////~ << std::setw(15) << Rhs[i]
								////~ << std::setw(15) << newB[i]
								////~ << std::setw(15) << diffRhsNewB[i]
								////~ << std::endl;
		////~ }
		
		//y = x;
		//MatrixAdapter<Matrix,Vector,Vector> linearOperator(stiffnessMatrix);
		//// Sequential incomplete LU decomposition as the preconditioner
		//SeqILU<Matrix,Vector,Vector> preconditioner(stiffnessMatrix,
                                              //1.0);  // Relaxation factor
		//CGSolver<Vector> cg(linearOperator,
                      //preconditioner,
                      //1e-9, // Desired residual reduction factor
                      //200,   // Maximum number of iterations
                      //2);   // Verbosity of the solver

  //// Object storing some statistics about the solving process
		//InverseOperatorResult statistics;
		//// Solve!
		//cg.apply(y, newB, statistics);
		
		////Dune
		//auto tmp(y);
		//tmp -= x;
		
		////@Debug
		//auto l2err = L2Norm(gridView, basis.localView(), tmp, 1.0/static_cast<double>(edges));
		//std::cerr << "Break-Condition: " << l2err << std::endl;
		//if( std::isnan(l2err)or (--n <= 0) or l2err < 1e-12 ) break;
		//x = y;
		//outputVector<double>(basis, x, std::ios::app | std::ios::ate, "output_u");
		//outputVector<double>(basis, uplus, std::ios::app | std::ios::ate, "output_uplus");
	//}
	//while( true );
	//const auto fixedpointEnd = std::chrono::high_resolution_clock::now();
	//std::cerr << "Stopped after " << (MaxIterations - n) << " iterations (Max " << MaxIterations << ")." << std::endl;
	//std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(fixedpointEnd-fixedpointStart).count() << " ms." << std::endl;
	
	//--------------
	//some norms
	{	
		auto const f = [=] (const auto& coords) { return std::sin(PI*coords[0])*std::sin(PI*coords[1]); };
		auto const Df = [=](const auto& coords) -> FieldVector<double,2> { return { PI*std::cos(PI*coords[0])*std::sin(PI*coords[1]), PI*std::sin(PI*coords[0])*std::cos(PI*coords[1]) }; };
		
		std::cerr << "||u_0-f||_2: h = " << H << ", " << L2Norm( gridView, basis.localView(), normalSolution, f, H) << std::endl;
	
		std::cerr << "||u_N-f||_2 = " << L2Norm( gridView, basis.localView(), x, f, H) << std::endl;
		std::cerr << "||u^+-f||_2 = " << L2Norm( gridView, basis.localView(), uplus, f, H) << std::endl;
		
		auto const ANormMaybeWorking = [&](BCRSMatrix<double>& A, BlockVector<double> u) { BlockVector<double> _tmp(u.size()); A.mv(u,_tmp); return std::sqrt(u*_tmp); };
		BlockVector<double> tmp;
		Functions::interpolate(basis,tmp,f);
		tmp -= uplus;
		
		std::cerr << "||u^+-u_0||_A = " << ANormMaybeWorking( stiffnessMatrix, tmp) << std::endl;
		std::cerr << "||u^+-f||_A = " << ANorm( gridView, basis.localView(), uplus, D, mu, f, Df ) << std::endl;
		
		BlockVector<double> tmp2(Rhs), tmp3(Rhs.size());
		stiffnessMatrix.mv(uplus, tmp3);
		tmp2 -= tmp3;
		for( int i = 0; i < tmp2.size(); i++ ) {
			tmp2[i] -= (eps + 2.0 * std::pow(H, 2.0)) * uminus[i];
		}
		std::cerr << "||Rhs - Auplus - s(uminus)||_\\infty = " << std::get<0>(inftyNorm(tmp2)) << std::endl;
		
		Eigen::Vector<double,Eigen::Dynamic> duneXtranscribed = ([](const auto& u){ Eigen::Vector<double,Eigen::Dynamic> tmp(u.size()); for( int i=0; i<u.size();i++ ) tmp[i]=u[i]; return tmp; })(x);
		std::cerr << "||u^+_eigen - u^+_dune||_L2 = " << L2Norm<double>( gridView, basis.localView(), eigenU.cwiseMin(UpperBound).cwiseMax(0)-duneXtranscribed.cwiseMin(UpperBound).cwiseMax(0) ) << std::endl;
	}
	
	return 0;
}
