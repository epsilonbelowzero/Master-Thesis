#pragma once

#include <fstream>
#include <iomanip>
#include <functional>

#include <limits>

#include <dune/istl/bvector.hh>

#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/geometry/quadraturerules.hh>

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>

//Contains a concept that works for both DUNE and Eigen vectors, functions
//to turn DUNE matrices / vectors into their Eigen pendant, output functions,
//a zero function, a diameter function to calculate a diameter from a simplex,
//a function to approximate the infinity norm over a simplex

//concept that both Dune::BlockVector and Eigen::Vector fit on for functions
//that work on both datatypes
template < class T >
concept IsEnumeratable = requires(T v) {
	v[v.size()-1];
};

constexpr
double zeroFunction(Dune::FieldVector<double,2>) {
	return 0;
}

//writes vector representing a function u over the grid to a file, either appending or truncating
template <typename FltPrec, typename VectorImpl, typename Basis> requires IsEnumeratable<VectorImpl>
void outputVector(const Basis& basis, const VectorImpl& u, const std::ios::openmode mode = std::ios::app | std::ios::ate, std::string fileName = "output.txt") {
	//interpolate the functions (x,y) -> x and (x,y) -> y to obtain the
	//x and y coordinates for u[i]
	Dune::BlockVector<double> xVals, yVals;
	Dune::Functions::interpolate(basis,xVals,[](auto x) { return x[0]; });
	Dune::Functions::interpolate(basis,yVals,[](auto x) { return x[1]; });
	
	assert(xVals.size() == yVals.size() && yVals.size() == u.size());
	
	std::ofstream outFile;
	outFile.open(fileName, mode | std::ios::out);
	//to use gnuplot's index keyword, seperate datasets by 2 newlines
	if( mode & std::ios::app) {
		outFile << std::endl;
		outFile << std::endl;
	}
	
	for( int i = 0; i < u.size(); i++ ) {
		outFile << xVals[i] << '\t' << yVals[i] << '\t' << u[i] << std::endl;
	}
	outFile.close();
}

//stores the grid layout to a file to be read and visualised by gnuplot
//with the command 'plot "gridlayout.gnuplot" with polygons'
template < typename GridView >
void gridlayoutToFile( const GridView& gridView, const double H, const std::string filename = "gridlayout.gnuplot" ) {
	//disabled for fine grids: takes too long for gnuplot and nothing to
	//see anyways
	if( 1.0 / H > 20 ) return;
	
	std::ofstream gridLayout;
	gridLayout.open(filename, std::ios::trunc | std::ios::out);
	//distinguish between triangles and rectangles
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
		//for rectangles we need a different order
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

//counts number of non-zero elements of a vector
//counts actually the number of zeros and substract this number from the
//total vector length.
//distinguishes between integral and floating point types
template < typename VectorImpl > requires IsEnumeratable<VectorImpl>
int noNnzElements( const VectorImpl& x ) {
	int n = 0;
	
	for( int i = 0; i < x.size(); i++ ) {
		if constexpr( std::is_integral_v<typename VectorImpl::value_type> ) {
			if( x[i] == 0)
				++n;
		}
		else {
			if( std::abs(x[i]) < std::numeric_limits<typename VectorImpl::value_type>::epsilon() ) {
				++n;
			}
		}
	}
	
	return x.size() - n;
}

//calculates the diamter of a simplex by calculating the distances between
//all corners and taking the maximum of those
template <class Geometry>
typename Geometry::ctype diameter(const Geometry& geom) {
	typename Geometry::ctype result = (geom.corner(0) - geom.corner(1)).two_norm();
	for( int i = 0; i < geom.corners(); i++ ) {
		for( int j = i+1; j < geom.corners(); j++ ) {
			result = std::max( result, (geom.corner(j)-geom.corner(i)).two_norm());
		}
	}
	
	return result;
}

//turns a DUNE matrix and vector into its fitting Eigen version
//do both in one function for stiffness matrix + right-hand side
template < typename FltPrec >
std::pair< Eigen::SparseMatrix<FltPrec>, Eigen::Matrix<FltPrec,Eigen::Dynamic,1> >
transcribeDuneToEigen( const Dune::BCRSMatrix<FltPrec>& stiffnessMatrix, const Dune::BlockVector<FltPrec>& rhs, const int maxOccupationPerRow ) {
	Eigen::SparseMatrix< FltPrec > eigenMatrix( stiffnessMatrix.N(), stiffnessMatrix.M() );
	Eigen::Matrix< FltPrec, Eigen::Dynamic, 1 > eigenRhs( rhs.size() );

	eigenMatrix.reserve(Eigen::VectorXi::Constant(stiffnessMatrix.M(),maxOccupationPerRow));

	for( auto row = stiffnessMatrix.begin(); row != stiffnessMatrix.end(); row++ ) {
		for( auto col = row->begin(); col != row->end(); col++ ) {
			//this loops over the whole matrix row, not only the non-zero entries
			//therefore checking
			if( std::abs(*col) < std::numeric_limits<FltPrec>::epsilon() ) continue;
			
			eigenMatrix.insert( row.index(), col.index() ) = *col;
		}
	}
	eigenMatrix.makeCompressed();
	for( size_t i = 0; i < rhs.size(); i++ ) {
		eigenRhs[i] = rhs[i];
	}
	
	return std::make_pair( eigenMatrix, eigenRhs );
}

//only a vector version: DUNE->Eigen
template < typename FltPrec >
Eigen::Vector<FltPrec,Eigen::Dynamic>
transcribeDuneToEigen( const Dune::BlockVector<FltPrec>& duneVec) {
	Eigen::Vector< FltPrec, Eigen::Dynamic > eigenVec( duneVec.size() );
	for( size_t i = 0; i < duneVec.size(); i++ ) {
		eigenVec[i] = duneVec[i];
	}
	
	return eigenVec;
}

//approximates the infinity norm of a function over an element
//by using a high order lagrange integration scheme and evaluates
//at the points the function
//furthermore, the corners are explicitely considered.
template < class Element, class FltPrec, class ReturnType, int DomainDim >
FltPrec getApproxMaximumNorm(
						const Element& element,
						const std::function<ReturnType(const Dune::FieldVector<FltPrec,DomainDim>)> f
)
{
	//~ constexpr int DomainDim = Element::dimension;
	//DomainDim == Element::dimension or Element::dimensionworld for elements / intersections
	
	//calculate the maximum depending on the function f:
	// - scalar function: absolute value of the result at pos x
	// - vector valued function: maximum of the absolute values at pos x
	// - matrix valued function: maximum of the absolute values of all matrix element at pos x
	const auto calcMaxAtPosition = [&f](const Dune::FieldVector<FltPrec,DomainDim> x) -> FltPrec {
		const ReturnType result = f(x);
		
		FltPrec ret = {0};
		if constexpr (std::is_integral_v<ReturnType> or std::is_floating_point_v<ReturnType> ) {
			ret = std::max(ret,std::abs(result));
		}
		else if constexpr(std::is_same_v<ReturnType,Dune::FieldVector<FltPrec,DomainDim> >) {
			for( int i=0; i < DomainDim; i++ ) {
				ret = std::max(ret,std::abs(result[i]));
			}
		}
		else if constexpr(std::is_same_v<ReturnType,Dune::FieldMatrix<FltPrec,DomainDim,DomainDim> >) {
			for( int i=0; i < DomainDim; i++ ) {
				for( int j=0; j < DomainDim; j++ ) {
					ret = std::max(ret,std::abs(result[i][j]));
				}
			}
		}
		
		return ret;
	};

	// A quadrature rule, maybe use localView.tree().finiteElement().localBasis().order()
	//high order for approximation of the infinity norm L^\infty
	const int order = 5;
	//Element::mydimension = 1 for intersections, = 2 for simplices
	const auto& quadRule = Dune::QuadratureRules<FltPrec, Element::mydimension>::rule(element.type(), order);
	
	FltPrec returnVal = {0};
	
	for (const auto& quadPoint : quadRule) {
		// Position of the current quadrature point in the reference element
		//this is 1d for intersection and 2d for elements
		const Dune::FieldVector<FltPrec,DomainDim> globalPos = element.geometry().global(quadPoint.position());
		
		returnVal = std::max(returnVal, calcMaxAtPosition(globalPos) );
	}
	
	//have a look at the corners too
	for( int i=0; i < element.geometry().corners(); i++ ) {
		returnVal = std::max(returnVal, calcMaxAtPosition(element.geometry().corner(i)));
	}
	
	return returnVal;
}
