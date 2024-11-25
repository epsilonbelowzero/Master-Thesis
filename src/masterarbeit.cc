#include <config.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>

#include <algorithm>
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
#include <dune/istl/umfpack.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/UmfPackSupport>
#include <eigen3/unsupported/Eigen/SparseExtra>

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

	//Max. Element
template < typename VectorImpl > requires IsEnumeratable<VectorImpl>
auto inftyNorm( const VectorImpl& x ) {
	auto max = std::abs(x[0]);
	size_t maxIdx = 0;
	
	for( int i = 1; i < x.size(); i++ ) {
		if( std::abs(x[i]) > max ) {
			max = std::abs(x[i]);
			maxIdx = i;
		}
	}
	
	return std::make_pair( max, maxIdx );
}

//counts number of non-zero elements of a vector
template < typename VectorImpl > requires IsEnumeratable<VectorImpl>
int noNnzElements( const VectorImpl& x ) {
	int n = 0;
	
	for( int i = 0; i < x.size(); i++ ) {
		if( std::abs(x[i]) < std::numeric_limits<typename VectorImpl::value_type>::epsilon() ) {
			++n;
		}
	}
	
	return x.size() - n;
}

//calculates the diamter of a simplex
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

template < typename FltPrec >
bool localOmegaAdaption( FltPrec& localOmega, const FltPrec errOld, const FltPrec errNew ) {
	bool ret = false;
	
	//disable adaption
	//~ return false;
	
	constexpr char escape = 27;

	if(errNew  > 0.5*errOld) {
		localOmega *= 0.5;
		std::cout << escape << "[1m" << "\tLowered localOmega to " << localOmega << escape << "[0m" << std::endl;
		
		ret = true;
	}
	if(errNew  < 0.125 * errOld) {
		localOmega *= 2;
		std::cout << escape << "[1m" << "\tIncreased localOmega to " << localOmega << escape << "[0m" << std::endl;
	}

	return ret;
}

//despite the name: the structure is to get the infinity norm of \mu, \beta and D
//over the set \omega_i
//untested!
//~ template <typename GridView, typename Element, typename GlobalPosition>
//~ bool sVectorRecursion( 	const GridView& gridView, 
												//~ const Element& center,
												//~ const Element& elevated,
												//~ const GlobalPosition& stop,
												//~ const int orientation, //1 = counter-clockwise, -1 = clockwise
												//~ std::unordered_set<Element::EntitySeed>& elementsForDiameters )
//~ {
	//~ using FltPrec = typename GridView::ctype;
	//~ constexpr Eps = std::numeric_limits<FltPrec>::epsilon();
	
	//~ if( (elevated.geometry().center() - stop).two_norm2() < Eps ) {
		//~ return false;
	//~ }
	
	//~ //works only for R^2, where intersections are lines
	//~ const auto checkGoodIntersection = [orientation,center&,elevated&](const auto& intersectionGeometry) {
		//~ const auto corner0Coords = intersectionGeometry.corner(0);
		//~ const auto corner1Coords = intersectionGeometry.corner(1);
		//~ const auto intersectionCoords = intersectionGeometry.center();
		//~ const auto centerCoords = center.geometry().center();
		//~ const auto elevatedCoords = elevated.geometry().center();
		
		//~ bool result = false;
		//~ //distance check: ensure, that there is exactly one corner of the intersection line identical to any of the `center`-elements corners
		//~ //version 1: use center.center() + diameter. might fail for triangles
		//~ result = (corner0Coords - center.geometry().center()).two_norm < 0.5*diameter(center.geometry());
		//~ result = result != (corner1Coords - center.geometry().center()).two_norm < 0.5*diameter(center.geometry());//`!=` serves as xor in this context
		
		//~ //version 2: check, whether there is a corner of element `center`, that's identical with either corner0 or corner1 of the intersection-line
		//~ //exact, but more expensive
		//~ //need to loop over all corners for the case of the intersection between center & elevated; for the first corner result becomes true, but this is reversed when the 2nd corner is considered
		//for( int i=0; i < center.geometry().corners(); i++ ) {
			//const auto cornerICoords = center.geometry().corner(i);
			//result = result != (corner0Coords - cornerICoords).two_norm2() < Eps;
			//result = result != (corner1Coords - cornerICoords).two_norm2() < Eps;
		//}
		
		//~ //orientation check
		//~ //ensure that the intersection "is left of the line elevatedCoords-centerCoords". For this reason, compute the scalar product of `intersectionCoords-elevatedCoords`
		//~ //and the cross-product (0,0,1)^T \times `elevatecCoords-centerCoords`.
		//~ const auto vec = elevatedCoords - centerCoords;
		//~ const FltPrec sp = (intersectionCoords - elevatedCoords) * Dune::FieldVector<FltPrec,2>{-vec[1],vec[0]};
		//~ result = result and (orientation * sp > 0);
		
		//~ return result;
	//~ }
	
	//~ for( const auto& intersection : intersections( gridView, elevated ) ) {
		//~ if( checkGoodIntersection(intersection.geometry()) ) {
			//~ if( intersection.boundary() ) {
				//~ return true; //getting around the centerElement `center` with given orientation `orientation` hit the boundary, so start again with the other orientation
			//~ }
			
			//~ elementsForDiameters.insert(diameter(intersection.outside().seed()));
			
			//~ return sVectorRecursion( gridView, center, intersection.outside(), stop, orientation, diameters );
		//~ }
	//~ }
	
	//~ std::cerr << "Error" << std::endl;
	//~ exit(10);
//~ }

//despite the name: the structure is to get the infinity norm of \mu, \beta and D
//over the set \omega_i
//~ template < class Basis >
//~ Dune::BlockVector<typename Basis::GridView::ctype> getSVector(
	//~ const Basis& basis, 
	//~ const Dune::FieldMatrix< typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension >& D,
	//~ const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension> beta,
	//~ const typename Basis::GridView::ctype mu)
//~ {
	//~ using FltPrec = typename Basis::GridView::ctype;
	
	//~ const auto gridView = basis.gridView();
	//~ auto viewElem = basis.localView();
	//~ auto viewOther = basis.localView();
	
	//~ Dune::BlockVector<FltPrec> result(basis.dimension());
	
	//~ for( const auto& elem : elements(gridView) ) {
		//~ viewElem.bind(elem);
		//~ for( const auto& other : elements(gridView) ) {
			//~ viewOther.bind(other);
			
			//~ assert( viewElem.size() == viewOther.size() );
			
			//~ std::vector<int> globalElem, globalOther;
			
			//~ for( int i=0; i < viewElem.size(); i++ ) {
				//~ globalElem.push_back( viewElem.index(i) );
				//~ globalOther.push_back( viewOther.index(i) );
			//~ }
			
			//~ const auto searchResult = std::find_first_of( globalElem.begin(), globalElem.end(), globalOther.begin(), globalOther.end() );
			
			//~ if( searchResult != globalElem.end() ) { //match, i.e. elem & other share at least one DOF
				//~ for( const auto& globalIdx : globalElem ) {
					//~ const FltPrec otherDiam = diameter(elem.geometry());
					//~ result[globalIdx] = std::max( result[globalIdx], otherDiam );
				//~ }
			//~ }
			
			//~ viewOther.unbind();
		//~ }
		//~ viewElem.unbind();
	//~ }
	
	//~ for( int i=0; i < result.size(); i++ ) {
		//~ result[i] = D.infinity_norm() + beta.infinity_norm() * result[i] + mu * result[i] * result[i];
	//~ }
	
	//~ return result;
//~ }

//despite the name: the structure is to get the infinity norm of \mu, \beta and D
//over the set \omega_i
//~ template < class Basis >
//~ Dune::BlockVector<typename Basis::GridView::ctype> getSVector(
	//~ const Basis& basis, 
	//~ const Dune::FieldMatrix< typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension >& D,
	//~ const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension> beta,
	//~ const typename Basis::GridView::ctype mu)
//~ {
	//~ using FltPrec = typename Basis::GridView::ctype;
	
	//~ const auto gridView = basis.gridView();
	//~ auto viewElem = basis.localView();
	
	//~ std::vector<std::unordered_set<Basis::LocalView::Element::EntitySeed>> elementsForDiameter;
	//~ Dune::BlockVector<FltPrec> result(basis.dimension());
	
	//~ for( const auto& elem : elements(gridView) ) {
		//~ viewElem.bind(elem);
		
		//~ std::unordered_set<Basis::LocalView::Element::EntitySeed> localElementsForDiameter;
		//~ for( const auto& intersection : intersections(gridView,elem) ) {
			//~ if( intersection.boundary() )
				//~ continue;
			
			//~ bool result = sVectorRecursion( gridView, elem, intersection.outside(),intersection.outside().geometry().center(),1,localElementsForDiameter );
			//~ if( result ) {
				//~ result = sVectorRecursion( gridView, elem, intersection.outside(),intersection.outside().geometry().center(),-1,localElementsForDiameter );
				
				//~ //if 1 boundary w/ orienation +1 was hit, we should hit the boundary again w/ orientation -1
				//~ assert(result);
			//~ }
			
			//~ break;
		//~ }
		
		//~ for( int i=0; i < viewElem.size(); i++ ) {
			//~ elementsForDiameter[viewElement.index(i)].insert(localElementsForDiameter.begin(),localElementsForDiameter.end());
		//~ }
	//~ }
	
	//~ for( int i=0; i < result.size(); i++ ) {
		//~ const int n = elementsForDiameter[i].size();
		//~ FltPrec sum = 0;
		//~ for( const auto& elem : elementsForDiameter[i] ) {
			//~ sum += diameter(gridView.entity(elem).geometry());
		//~ }
		//~ result[i] = D.infinity_norm() + beta.infinity_norm() * result[i] + mu * result[i] * result[i];
	//~ }
	
	//~ return result;
//~ }

template < class Basis >
Dune::BlockVector<typename Basis::GridView::ctype> getSVector(
	const Basis& basis, 
	const typename Basis::GridView::ctype diffusionInfinityNorm,
	const typename Basis::GridView::ctype betaInfinityNorm,
	const typename Basis::GridView::ctype mu)
{
	//generate the s-Vector in 3 steps:
	//	1) loop over all elements, and count the numbers of elements each
	//		 mesh node is corner of, and sum up the diameters of those elements;
	//		 identify mesh node as a corner of an element
	//	2) calculate the value at each mesh node (= corner of an element)
	//		 (i.e. contributions of \mathcal D, \beta, \mu and \mathfrak h)
	//	3) linear interpolate the values of each mesh grid (=corner of an
	//		 element) for the remaining lagrange node
	//@TODO current implementation assumes that \mathcal D, \beta have constant infinity norm
	
	using FltPrec = typename Basis::GridView::ctype;
	Dune::BlockVector<FltPrec> xVals, yVals; //save x/y coordinate for lagrange node i
	Dune::Functions::interpolate(basis,xVals,[](auto x) { return x[0]; });
	Dune::Functions::interpolate(basis,yVals,[](auto x) { return x[1]; });
	
	Dune::BlockVector<FltPrec> result(basis.dimension());
	std::vector<int> noParticipatingElems(basis.dimension());//no=Number; needed for average calculation
	auto localView = basis.localView();
	
	//detect a mesh node (=corner of an element) by comparing local coordinates
	const auto isCorner = [](const FltPrec x, const FltPrec y) -> bool {
		constexpr FltPrec Limit = 1e-15; //std::numeric_limits<FltPrec>::epsilon() doesn't work - some corners are then not detected with the approach below
		return	(std::abs(	x) < Limit && std::abs(	 y) < Limit) ||	//bottom left corner
						(std::abs(1-x) < Limit && std::abs(	 y) < Limit) ||	//bottom right corner
						(std::abs(	x) < Limit && std::abs(1-y) < Limit) ||	//top left corner
						(std::abs(1-x) < Limit && std::abs(1-y) < Limit);		//quads only, top right corner
	};
	
	//step 1
	for( const auto& elem : elements(basis.gridView()) ) {
		localView.bind(elem);
		
		const FltPrec diam = diameter(elem.geometry());
		int noCornersDetected = 0;
		for( int i=0; i < localView.size(); i++ ) {
			
			const auto localCoord = elem.geometry().local({xVals[localView.index(i)], yVals[localView.index(i)]});
			if( isCorner(localCoord[0],localCoord[1]) ) {
				result[localView.index(i)] += diam;
				noParticipatingElems[localView.index(i)]++;
				
				noCornersDetected++;
			}
		}
		assert(noCornersDetected == elem.geometry().corners());
	}
	
	//step 2
	for( int i=0;i<result.size(); i++) {
		if(noParticipatingElems[i] == 0) continue; //skip Lagrange nodes that are no mesh nodes
		
		const FltPrec h_i = result[i] / noParticipatingElems[i];
		result[i] = diffusionInfinityNorm + h_i * betaInfinityNorm + h_i * h_i * mu;
	}
	
	//step 3
	for( const auto& elem : elements(basis.gridView()) ) {
		localView.bind(elem);
		
		//get the (global) indices of current elements corners
		std::vector<int> corners;
		for( int i=0; i < localView.size(); i++ ) {
			const auto localCoord = elem.geometry().local({xVals[localView.index(i)], yVals[localView.index(i)]});
			if( isCorner(localCoord[0],localCoord[1]) ) {
				corners.push_back(localView.index(i));
			}
		}
		assert(corners.size() == elem.geometry().corners());
		
		//linear interpolate for the remaining lagrange nodes
		for( int i=0; i < localView.size(); i++ ) {
			const int globalI = localView.index(i);
			//corners are already done; skip them
			if( std::find(corners.begin(),corners.end(),globalI) == corners.end() ) {
				const auto localCoord = elem.geometry().local({xVals[globalI], yVals[globalI]});
				assert(0 <= localCoord[0] <= 1);
				assert(0 <= localCoord[1] <= 1);
				assert(0 <= localCoord[0] + localCoord[1] <= 1);
				//linear interpolation of the corresponding corner values
				result[globalI] = (1-localCoord[0]-localCoord[1]) * result[corners[0]] + localCoord[0] * result[corners[1]] + localCoord[1] * result[corners[2]];
			}
		}
	}
	
	return result;
}

template < typename FltPrec, class Norm, class OutputMethod >
Eigen::Vector< FltPrec, Eigen::Dynamic > fixedpointMethod(
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& u0,
	const Eigen::SparseMatrix< FltPrec >& A,
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& Rhs,
	const FltPrec omega,
	const Eigen::Vector<FltPrec,Eigen::Dynamic> sVector,
	const FltPrec H,
	const Eigen::Array<FltPrec,Eigen::Dynamic,1>& uKappaU,
	const Eigen::Array<FltPrec,Eigen::Dynamic,1>& uKappaL,
	const Norm L2Norm,
	const OutputMethod Output 
)
{
	std::cerr << "Fixedpoint Method with Eigen Interface" << std::endl;
	
	constexpr bool DoOutput = true;
	
	Eigen::Vector<FltPrec,Eigen::Dynamic> oldB(Rhs),newB(Rhs);
	Eigen::Vector<FltPrec,Eigen::Dynamic> u(u0);

	//~ Eigen::ConjugateGradient<Eigen::SparseMatrix<FltPrec>,Eigen::Lower|Eigen::Upper> solver;
	//~ solver.setTolerance(1e-9);
	//~ solver.compute(A);
	//~ Eigen::SparseLU<Eigen::SparseMatrix<FltPrec>,Eigen::COLAMDOrdering<int> > solver;
	Eigen::UmfPackLU<Eigen::SparseMatrix<FltPrec> > solver;
	
	if constexpr(DoOutput) Output(u0,std::ios::trunc,"output_u");
	if constexpr(DoOutput) Output(u0,std::ios::trunc,"output_uplus");

	const int MaxIterations = 10000;
	int n = MaxIterations;
	FltPrec localOmega = omega;
	const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	solver.analyzePattern(A);
	solver.factorize(A);
	FltPrec l2errOld = std::numeric_limits<FltPrec>::infinity();
	do {
		const auto updateUplusStart = std::chrono::high_resolution_clock::now();
		const Eigen::Vector<FltPrec,Eigen::Dynamic> uplus = u.array().min(uKappaU).max(uKappaL).matrix();
		const auto updateRhsStart = std::chrono::high_resolution_clock::now();
		newB = A*u + localOmega*( Rhs - A*uplus - (sVector.array()*(u-uplus).array()).matrix() );
		const auto solverStart = std::chrono::high_resolution_clock::now();
		const Eigen::Vector<FltPrec,Eigen::Dynamic> y = solver.solve(newB);
		const auto solverEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tu+: " << std::chrono::duration<float,std::milli>(updateRhsStart-updateUplusStart).count() << " ms." << std::endl;
		std::cout << "\trhs: " << std::chrono::duration<float,std::milli>(solverStart-updateRhsStart).count() << " ms." << std::endl;
		std::cout << "\tSolver: " << std::chrono::duration<float,std::milli>(solverEnd-solverStart).count() << " ms." << std::endl;
		
		//@Debug
		const auto computeNormStart = std::chrono::high_resolution_clock::now();
		const auto l2err = L2Norm((y-u).eval());
		if( localOmegaAdaption( localOmega, l2errOld, l2err ) ) continue;
		l2errOld = l2err;
		const auto computeNormEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tNorm: " << std::chrono::duration<float,std::milli>(computeNormEnd-computeNormStart).count() << " ms." << std::endl;
		const auto updateUStart = std::chrono::high_resolution_clock::now();
		u = y.eval();
		const auto updateUEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tUpdate u: " << std::chrono::duration<float,std::milli>(updateUEnd-updateUStart).count() << " ms." << std::endl;
		if constexpr(DoOutput) Output(u, std::ios::app | std::ios::ate, "output_u");
		if constexpr(DoOutput) Output(uplus, std::ios::app | std::ios::ate, "output_uplus");
		std::cout << "Break-Condition: " << l2err << std::endl;
		if( std::isnan(l2err) or (--n <= 0) or l2err < 1e-8 ) break; //1e-12 w/ \beta==0, 1e-8 otherwise
	}
	while( true );
	const auto fixedpointEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "Stopped after " << (MaxIterations - n) << " iterations (Max " << MaxIterations << ")." << std::endl;
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(fixedpointEnd-fixedpointStart).count() << " ms." << std::endl;
	
	return u;
}

template < typename FltPrec, class Norm, class OutputMethod, typename VectorImpl > requires IsEnumeratable<VectorImpl>
Dune::BlockVector< FltPrec > fixedpointMethod(
	const Dune::BlockVector<FltPrec>& u0,
	const Dune::BCRSMatrix< FltPrec >& A,
	const Dune::BlockVector<FltPrec>& Rhs,
	const FltPrec omega,
	const Dune::BlockVector<FltPrec> sVector, //@TODO
	const FltPrec H,
	const VectorImpl& uKappaU, //upper bounds at each lagrange node
	const VectorImpl& uKappaL, //lower bounds at each lagrange node
	const Norm L2Norm,
	const OutputMethod Output 
)
{
	std::cerr << "Fixedpoint Method with Dune Interface" << std::endl;
	
	constexpr bool DoOutput = false;
	
	using Vector = Dune::BlockVector<FltPrec>;
	using Matrix = Dune::BCRSMatrix<FltPrec>;
	
	if constexpr (DoOutput) Output(u0,std::ios::trunc,"output_u");
	if constexpr (DoOutput) Output(u0,std::ios::trunc,"output_uplus");

	Dune::MatrixAdapter<Matrix,Vector,Vector> linearOperator(A);
	// Sequential incomplete LU decomposition as the preconditioner
	//~ Dune::SeqILU<Matrix,Vector,Vector> preconditioner(A,
										  //~ 1.0);  // Relaxation factor
	//~ Dune::SeqJac<Matrix,Vector,Vector> preconditioner(A, 1, 1.0);
	//~ Dune::CGSolver<Vector> solver(linearOperator,
				  //~ preconditioner,
				  //~ 1e-9, // Desired residual reduction factor
				  //~ 200,   // Maximum number of iterations
				  //~ 2);   // Verbosity of the solver
	Dune::UMFPack<Matrix> solver(A, 0);

	// Object storing some statistics about the solving process
	Dune::InverseOperatorResult statistics;

	Vector 	x(u0.size()),
					uplus(u0.size()),
					uminus(u0),
					newB(u0.size()),
					y(u0.size());
	x = u0;
	
	const int NnzRhs = noNnzElements(Rhs);
	const int NoInnerNods = NnzRhs; //only works for dirichlet boundary conditions and non-zero rhs-function
	const int NoBoundaryNodes = Rhs.size() - NnzRhs;
	
	FltPrec l2errOld = std::numeric_limits<FltPrec>::infinity();
	FltPrec localOmega = omega;
	
	const auto fixedpointStart = std::chrono::high_resolution_clock::now();
	const int MaxIterations = 10000;
	int n = MaxIterations;
	do {
		const auto uplusStart = std::chrono::high_resolution_clock::now();
		for( int i = 0; i < x.size(); i++ ) {
			uplus[i] = std::clamp(x[i],uKappaL[i],uKappaU[i]);
		}
		const auto uplusEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tu+: " << std::chrono::duration<float,std::milli>(uplusEnd-uplusStart).count() << " ms." << std::endl;
		Vector aUplus(u0.size()), Au(u0.size());
		const auto oldRhsStart = std::chrono::high_resolution_clock::now();
		A.mv( x, Au );
		const auto auplusStart = std::chrono::high_resolution_clock::now();
		A.mv( uplus, aUplus );
		const auto auplusEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\told rhs: " << std::chrono::duration<float,std::milli>(auplusStart-oldRhsStart).count() << " ms." << std::endl;
		std::cout << "\tAu+: " << std::chrono::duration<float,std::milli>(auplusEnd-auplusStart).count() << " ms." << std::endl;
		
		const auto newRhsStart = std::chrono::high_resolution_clock::now();
		for( int i = 0; i < x.size(); i++ ) {
			uminus[i] = x[i] - uplus[i];
			newB[i] = Au[i] + localOmega * (Rhs[i] - aUplus[i] - sVector[i]*uminus[i]);
		}
		const auto newRhsEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tNew Rhs: " << std::chrono::duration<float,std::milli>(newRhsEnd-newRhsStart).count() << " ms." << std::endl;
		
		//~ const auto [max,maxId] = inftyNorm( sVector );
		//~ std::cerr << "max s / max b / ration: " << max << '\t' << Rhs[maxId] << '\t' << max / Rhs[maxId] << std::endl;
		//~ const int NnzS = noNnzElements(sVector);
		//~ std::cerr << "nnz rhs / nnz s / ratio (inner): " << NnzRhs << '\t' << NnzS << '\t' << static_cast<double>(NnzS) / (sVector.size() - NoBoundaryNodes) << std::endl;
		
		const auto updateInitialGuessStart = std::chrono::high_resolution_clock::now();
		y = x;
		const auto updateInitialGuessEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tUpdate Initial Guess: " << std::chrono::duration<float,std::milli>(updateInitialGuessEnd-updateInitialGuessStart).count() << " ms." << std::endl;

		// Solve!
		const auto solveStart = std::chrono::high_resolution_clock::now();
		solver.apply(y, newB, statistics);
		const auto solveEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tSolve: " << std::chrono::duration<float,std::milli>(solveEnd-solveStart).count() << " ms." << std::endl;
		
		//Dune
		const auto computeDeltaStart = std::chrono::high_resolution_clock::now();
		auto tmp(y);
		tmp -= x;
		const auto computeDeltaEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tCompute Delta: " << std::chrono::duration<float,std::milli>(computeDeltaEnd-computeDeltaStart).count() << " ms." << std::endl;
		
		//@Debug
		const auto normStart = std::chrono::high_resolution_clock::now();
		auto l2err = L2Norm(tmp);
		const auto normEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tCompute Norm: " << std::chrono::duration<float,std::milli>(normEnd-normStart).count() << " ms." << std::endl;
		std::cout << "Break-Condition: " << l2err << std::endl;
		
		if( localOmegaAdaption(localOmega,l2errOld,l2err) ) continue;
		l2errOld = l2err;
		
		if( std::isnan(l2err)or (--n <= 0) or l2err < 1e-8 ) break; //1e-12 if beta==0, 1e-8 otherwise
		const auto updateXStart = std::chrono::high_resolution_clock::now();
		x = y;
		const auto updateXEnd = std::chrono::high_resolution_clock::now();
		std::cout << "\tUpdate Solution: " << std::chrono::duration<float,std::milli>(updateXEnd-updateXStart).count() << " ms." << std::endl;
		
		if constexpr (DoOutput) Output(x, std::ios::app | std::ios::ate, "output_u");
		if constexpr (DoOutput) {
			Vector tmpUPlus(x.size());
			for( int i=0; i < tmpUPlus.size(); i++ ) {
				tmpUPlus[i] = std::clamp(y[i],uKappaL[i],uKappaU[i]);
			}
			Output(tmpUPlus, std::ios::app | std::ios::ate, "output_uplus");
		}
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
	const Eigen::Vector<FltPrec,Eigen::Dynamic>& sVector,
	const FltPrec Diameter,
	const Eigen::Array<FltPrec,Eigen::Dynamic,1> uKappaU, //upper bounds at each Lagrange node
	const Eigen::Array<FltPrec,Eigen::Dynamic,1> uKappaL,
	const auto normFunc
	)
{
	constexpr bool nanCheck = false;
	constexpr bool conditionNumberCheck = false;
	constexpr bool solveableCheck = false;
	constexpr bool doOutput = true;
	constexpr bool doDurations = false;
	
	Eigen::Vector<FltPrec,Eigen::Dynamic> u(u0);
	auto bouligand = Eigen::DiagonalMatrix<FltPrec,Eigen::Dynamic>(b.size());
	//~ Eigen::BiCGSTAB<Eigen::SparseMatrix<FltPrec,Eigen::RowMajor> > solver;
	//~ solver.setTolerance(1e-9);
	//~ solver.setMaxIterations(1e3);
	//~ Eigen::SparseLU<Eigen::SparseMatrix<FltPrec>,Eigen::COLAMDOrdering<int> > solver;
	Eigen::UmfPackLU<Eigen::SparseMatrix<FltPrec> > solver;
	
	const auto F = [&b,&A,&sVector,Diameter,&uKappaU,&uKappaL](const Eigen::Matrix<FltPrec,Eigen::Dynamic,1>& u) {
		const auto uplus = u.array().min(uKappaU).max(uKappaL).matrix();
		return b - A*uplus - (sVector.array()*(u - uplus).array()).matrix();
	};
	
	//~ const int NnzRhs = noNnzElements(b);
	//~ const int NoInnerNods = NnzRhs; //only works for dirichlet boundary conditions and non-zero rhs-function
	//~ const int NoBoundaryNodes = b.size() - NnzRhs;
	
	int n = 0;
	std::cerr << "while loop" << std::endl;
	if constexpr (doOutput) outputVector<FltPrec>( basis, u, std::ios::trunc, "newton" );
	if constexpr (doOutput) outputVector<FltPrec>( basis, u.array().min(uKappaU).max(uKappaL).matrix(), std::ios::trunc, "newton_uplus" );
	do {
		n++;
		
		std::cerr << "\tLoop body" << std::endl;
		const auto setIdentity = std::chrono::high_resolution_clock::now();
		bouligand.setIdentity();
		const auto setIdentityEnd = std::chrono::high_resolution_clock::now();
		if constexpr(doDurations) {
			std::cout << "\t\tSet Identity end: " << std::chrono::duration<float,std::milli>(setIdentityEnd-setIdentity).count() << std::endl;	
		}
		const auto updateBouligand = std::chrono::high_resolution_clock::now();
		nanCheck and std::cout << "\t\tNaN Check 1: " << u.hasNaN() << std::endl;
		for( int i = 0; i < u.size(); i++ ) {
			if( u[i] < uKappaL[i]-std::numeric_limits<FltPrec>::epsilon() || u[i] > uKappaU[i]+std::numeric_limits<FltPrec>::epsilon() ) {
				bouligand.diagonal()[i] = 0;
			}
			//switch from bouligand to clark?
			//~ if( 	(0-std::numeric_limits<FltPrec>::epsilon() < u[i] && u[i] < 0+std::numeric_limits<FltPrec>::epsilon()) 
			   //~ || (kappa-std::numeric_limits<FltPrec>::epsilon() < u[i] && u[i] < kappa+std::numeric_limits<FltPrec>::epsilon()) )
			//~ {
				//~ bouligand.diagonal()[i] = 0.25;
			//~ }
		}
		const auto updateBouligandEnd = std::chrono::high_resolution_clock::now();
		if constexpr(doDurations) {
			std::cout << "\t\tUpdate bouligand end: " << std::chrono::duration<float,std::milli>(updateBouligandEnd-updateBouligand).count() << std::endl;
		}
		const auto updateTmp = std::chrono::high_resolution_clock::now();
		
		//way better (performance wise) implementation of (sFactor* 1 - A) * bouligand - sFactor * 1
		Eigen::SparseMatrix<FltPrec,Eigen::RowMajor> tmp = - A * bouligand;
		nanCheck and std::cerr << "\t\tNaN Check 1.25: " << Eigen::MatrixXd(A).hasNaN() << std::endl;
		nanCheck and std::cerr << "\t\tNaN Check 1.5: " << Eigen::MatrixXd(bouligand).hasNaN() << std::endl;
		nanCheck and std::cerr << "\t\tNaN Check 2: " << Eigen::MatrixXd(tmp).hasNaN() << std::endl;
		tmp.diagonal() += (sVector.array()*(bouligand.diagonal().array() - 1)).matrix();
		nanCheck and std::cerr << "\t\tNaN Check 3: " << Eigen::MatrixXd(tmp).hasNaN() << std::endl;
		if constexpr(solveableCheck) {
			std::cout << "tmp eigenvals: " << Eigen::MatrixXd(tmp).eigenvalues() << std::endl;
			std::cout << "tmp det: " << Eigen::MatrixXd(tmp).determinant() << std::endl;
			std::cout << "tmp: " << tmp << std::endl;
			std::cout << "rhs: " << -F(u) << std::endl;
		}
		const auto updateTmpEnd = std::chrono::high_resolution_clock::now();
		if constexpr(doDurations) {
			std::cout << "\t\tUpdate tmp end: " << std::chrono::duration<float,std::milli>(updateTmpEnd - updateTmp).count() << std::endl;
		}
		const auto compute = std::chrono::high_resolution_clock::now();
		//~ solver.compute(tmp); //iterative solvers
		solver.analyzePattern(tmp); //direct solver
		solver.factorize(tmp);		  //direct solver
		if constexpr(conditionNumberCheck) {
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(tmp);
			std::cerr << "\t\t\tCondition number = " << (svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1)) << std::endl;;
			std::cout << "\t\t\t\tlargest = " << svd.singularValues()(0) << std::endl;
			std::cout << "\t\t\t\tsmallest = " << svd.singularValues()(svd.singularValues().size()-1) << std::endl;
			std::cout << "\t\t\t\tall = " << svd.singularValues() << std::endl;
		}
		const auto computeEnd = std::chrono::high_resolution_clock::now();
		if constexpr(doDurations) {
			std::cout << "\t\tCompute end: " << std::chrono::duration<float,std::milli>(computeEnd - compute).count() << std::endl;
		}
		const auto solve = std::chrono::high_resolution_clock::now();
		const Eigen::Vector<FltPrec,Eigen::Dynamic> b = -F(u);
		const Eigen::Vector<FltPrec,Eigen::Dynamic> d = solver.solve(b);
		nanCheck and std::cerr << "\t\tNaN Check 4: " << (-F(u)).hasNaN() << std::endl;
		nanCheck and std::cerr << "\t\tNaN Check 5: " << d.hasNaN() << std::endl;
		const auto solveEnd = std::chrono::high_resolution_clock::now();
		if constexpr(doDurations) {
			std::cout << "\t\tSolve end: " << std::chrono::duration<float,std::milli>(solveEnd - solve).count() << " (" << /*solver.iterations() << " iterations / " << solver.error() << " )" <<*/ std::endl;
		}
		
		//~ const auto sVector=sFactor*(u-u.cwiseMin(a).cwiseMax(0));
		//~ const auto [max,maxId] = inftyNorm( sVector );
		//~ std::cerr << "max s / max b / ration: " << max << '\t' << b[maxId] << '\t' << max / b[maxId] << std::endl;
		//~ const int NnzS = noNnzElements(sVector);
		//~ std::cerr << "nnz rhs / nnz s / ratio (inner): " << NnzRhs << '\t' << NnzS << '\t' << static_cast<double>(NnzS) / (u.size() - NoBoundaryNodes) << std::endl;
		
		const FltPrec l2err = normFunc(d);
		std::cerr << "Break condition: " << l2err << std::endl;
		if(l2err < 1e-8)//w/o convection/cip: 1e-12
			break;
		
		const auto updateD = std::chrono::high_resolution_clock::now();
		u += d;
		nanCheck and std::cout << "\t\tNaN Check 5: " << u.hasNaN() << std::endl;
		const auto updateDEnd = std::chrono::high_resolution_clock::now();
		if constexpr(doDurations) {
			std::cout << "\t\tUpdate u end: " << std::chrono::duration<float,std::milli>(updateDEnd - updateD).count() << std::endl;
		}
		if constexpr (doOutput) outputVector<FltPrec>( basis, u, std::ios::app | std::ios::ate, "newton" );
		if constexpr (doOutput) outputVector<FltPrec>( basis, u.array().min(uKappaU).max(uKappaL), std::ios::app | std::ios::ate, "newton_uplus" );
		std::cerr << "\tLoop body end" << std::endl;
	} while( true );
	std::cerr << "while loop end. " << n << " iterations." << std::endl;
	//~ std::cout << u << std::endl;
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

template < typename LocalView, typename GridView, typename VectorImpl > requires IsEnumeratable<VectorImpl>
typename GridView::ctype ANorm( const GridView& gridView, LocalView localView,
	const VectorImpl& u,
	 const std::function<
					Dune::FieldMatrix<typename GridView::ctype, GridView::dimension, GridView::dimension>(
						const Dune::FieldVector<typename GridView::ctype, GridView::dimension>
					)
				> diffusion,
	const typename GridView::ctype diffusionInfinityNorm,
	const typename GridView::ctype mu,
	const std::function<typename GridView::ctype(Dune::FieldVector<typename GridView::ctype,GridView::dimension>)> f,
	const std::function<Dune::FieldVector<typename GridView::ctype,GridView::dimension>(const Dune::FieldVector<typename GridView::ctype,GridView::dimension>)> Df
)
{
	constexpr int dim = LocalView::Element::dimension;
	using FltPrec = typename GridView::ctype;
	
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
			std::vector<Dune::FieldVector<FltPrec,dim> > gradients(referenceGradients.size());
			for (size_t i=0; i<gradients.size(); i++) {
				jacobian.mv(referenceGradients[i][0], gradients[i]);
			}
			
			Dune::FieldVector<FltPrec,dim> gradSum; gradSum = 0;
			FltPrec shapeFunctionSum = 0;
			for( int i = 0; i < gradients.size(); i++ ) {
				const int globalIndex = localView.index(i);
				gradSum += u[globalIndex] * gradients[i];
				shapeFunctionSum += u[globalIndex] * shapeFunctionValues[i];
			}
			gradSum -= Df( elem.geometry().global(quadPos) );
			Dune::FieldVector<FltPrec,dim> dGradSum;
			const Dune::FieldMatrix<FltPrec,dim,dim> localD = diffusion(elem.geometry().global(quadPos));
			localD.mv( gradSum, dGradSum );
			
			aNorm += quadPoint.weight() * integrationElement * ( dGradSum * gradSum + mu * std::pow(shapeFunctionSum - functionValue,2) );
		}
		
		
		localView.unbind();
	}
	
	return std::sqrt(aNorm);
}

constexpr
double zeroFunction(Dune::FieldVector<double,2>) {
	return 0;
}

template < typename FltPrec, typename VectorImpl, typename LocalView, typename GridView > requires IsEnumeratable<VectorImpl>
FltPrec L2Norm( GridView gridView, LocalView localView, const VectorImpl u,
								const std::function<double(Dune::FieldVector<double,LocalView::Element::dimension>)> f = zeroFunction
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
	
	return std::sqrt(l2err);
}

//Eigen uses expression templates and therefore the appropriate call uses this version; const Eigen::Vector<FltPrec,Eigen::Dynamic> u doesn't work for this reason
//other possible solution: call .eval() on the corresponding vector before calling this function and use Eigen::Vector as type instead.
//due to more specialisation, dune-vectors use the function overload below
template < typename VectorImpl, typename FltPrec > requires IsEnumeratable<VectorImpl>
FltPrec sNorm(
	const VectorImpl& u,
	const FltPrec diffusionInfinityNorm,
	const FltPrec betaInfinityNorm,
	const FltPrec mu,
	const FltPrec H)
{
	return std::sqrt((diffusionInfinityNorm + betaInfinityNorm * H + mu * H * H) * u.dot(u));
}

template < typename FltPrec, int Dim >
FltPrec sNorm(
	const Dune::BlockVector<FltPrec> u,
	const Dune::FieldMatrix<FltPrec, Dim, Dim >& D,
	const FltPrec betaInfinityNorm,
	const FltPrec mu,
	const FltPrec H)
{
	return std::sqrt((D.infinity_norm() + betaInfinityNorm * H + mu * H * H) * (u*u));
}

template < typename FltPrec, int MantissaLength=500 >
std::pair< Eigen::SparseMatrix<FltPrec>, Eigen::Matrix<FltPrec,Eigen::Dynamic,1> >
transcribeDuneToEigen( const Dune::BCRSMatrix<FltPrec>& stiffnessMatrix, const Dune::BlockVector<FltPrec>& rhs, const int maxOccupationPerRow ) {
	Eigen::SparseMatrix< FltPrec > eigenMatrix( stiffnessMatrix.N(), stiffnessMatrix.M() );
	Eigen::Matrix< FltPrec, Eigen::Dynamic, 1 > eigenRhs( rhs.size() );

	eigenMatrix.reserve(Eigen::VectorXi::Constant(stiffnessMatrix.M(),maxOccupationPerRow));

	const auto startLoop = std::chrono::high_resolution_clock::now();
	for( auto row = stiffnessMatrix.begin(); row != stiffnessMatrix.end(); row++ ) {
		for( auto col = row->begin(); col != row->end(); col++ ) {
			//this loops over the whole matrix row, not only the non-zero entries
			//therefore checking
			if( std::abs(*col) < std::numeric_limits<double>::epsilon() ) continue;
			
			if constexpr (std::is_same_v< FltPrec, mpf_class >) {
				eigenMatrix.insert( row.index(), col.index() ) = mpf_class(*col, MantissaLength );
			}
			else {
				eigenMatrix.insert( row.index(), col.index() ) = *col;
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
	std::cout << '\t' << "Matrix-Insert " << std::chrono::duration<float, std::milli>(startCompressing - startLoop).count() << "ms." << std::endl;
	std::cout << '\t' << "Compressing " << std::chrono::duration<float, std::milli>(startRhs - startCompressing).count() << "ms." << std::endl;
	std::cout << '\t' << "Rhs " << std::chrono::duration<float, std::milli>(done - startRhs).count() << "ms." << std::endl;
	
	return std::make_pair( eigenMatrix, eigenRhs );
}

template < typename FltPrec >
Eigen::Vector<FltPrec,Eigen::Dynamic>
transcribeDuneToEigen( const Dune::BlockVector<FltPrec>& duneVec) {
	Eigen::Vector< FltPrec, Eigen::Dynamic > eigenVec( duneVec.size() );
	for( size_t i = 0; i < duneVec.size(); i++ ) {
		eigenVec[i] = duneVec[i];
	}
	
	return eigenVec;
}

using namespace Dune;

template < class Basis >
void addCIP(const Basis& basis,
						BCRSMatrix<typename Basis::GridView::ctype>& stiffnessMatrix,
						const typename Basis::GridView::ctype betaInfinityNorm,
						const typename Basis::GridView::ctype gamma)
{
	using FltPrec = typename Basis::GridView::ctype;
	
	const auto action = [&stiffnessMatrix](const int globalI,const int globalJ, const FltPrec contrib) {
		stiffnessMatrix[globalI][globalJ] += contrib;
	};
	addCIPImpl(basis,action,betaInfinityNorm,gamma);
}

template < class Basis, typename VectorImpl > requires IsEnumeratable<VectorImpl>
typename Basis::GridView::ctype cipNorm(	const Basis& basis,
								const VectorImpl& u,
								const std::function<
												typename Dune::FieldMatrix<typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension>(
													typename Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
												)
											> diffusion,
								const typename Basis::GridView::ctype diffusionInfinityNorm,
								const typename Basis::GridView::ctype betaInfinityNorm,
								const typename Basis::GridView::ctype mu,
								const typename Basis::GridView::ctype gamma,
								const std::function<double(Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>)> f,
								const std::function<Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>)> Df)
{
	//WORKS ONLY for functions of at least C^1!
	//i.e. [\grad u] = 0
	using FltPrec = typename Basis::GridView::ctype;
	
	FltPrec result = std::pow(ANorm(basis.gridView(),basis.localView(),u,diffusion,diffusionInfinityNorm,mu,f,Df),2);
	
	const auto action = [&u,&result](const int globalI,const int globalJ, const FltPrec contrib) {
		result += u[globalI] * u[globalJ] * contrib;
	};
	addCIPImpl(basis,action,betaInfinityNorm,gamma);

	if( result < 0 and std::abs(result) < std::numeric_limits<float>::epsilon() ) {
		result = std::abs(result);
	}
	
	return std::sqrt(result);
}

template < class Basis >
void addCIPImpl(const Basis& basis,
						const std::function<void(const int,const int,const typename Basis::GridView::ctype)> action,
						const typename Basis::GridView::ctype betaInfinityNorm,
						const typename Basis::GridView::ctype gamma)
{
	using FltPrec = typename Basis::GridView::ctype;
	constexpr int DomainDim = Basis::GridView::dimension;
	
	const auto evaluateIntegrand = [gamma,betaInfinityNorm](const FltPrec h_F, const FieldVector<FltPrec,DomainDim> diffIn, const FieldVector<FltPrec,DomainDim> diffOut) -> FltPrec {
		return gamma *betaInfinityNorm * (h_F*h_F) * (diffIn*diffOut);
	};
	
	const auto gridView = basis.gridView();
	auto localView = basis.localView();
	auto localViewOut = basis.localView();
	for( const auto& element : elements(gridView) ) {
		localView.bind(element);
		for( const auto& intersection : intersections(gridView, element) ) {
			if( intersection.boundary() ) {
				continue;
			}
			
			localViewOut.bind(intersection.outside());
			
			const FltPrec h_F = diameter(intersection.geometry());
			
			constexpr int intersectionDim = DomainDim - 1;
			
			//Q_i results in i, but highest order is 2i, reducted by 1 (gradients), doubled (product of two such functions)
			const int quadOrder = 2*(2*localView.tree().finiteElement().localBasis().order()-1);
			const auto& quadRule = QuadratureRules<FltPrec,intersectionDim>::rule(intersection.type(),quadOrder);
			const auto& localFiniteElement = localView.tree().finiteElement();
			
			auto const geometryIn = intersection.inside().geometry();
			auto const geometryIntersection = intersection.geometry();
			auto const geometryOut = intersection.outside().geometry();
			
			for( const auto& quadPoint : quadRule ) {
				const auto quadPos = quadPoint.position();
				const auto quadPosGlobal = geometryIntersection.global(quadPos);
				//switch to geometryIn(In|Out)side?
				//~ std::cout << "\t\t\t\tTest: " << intersection.geometryInInside().global(quadPos) << " vs. " << geometryIn.local(quadPosGlobal) << std::endl;
				//~ std::cout << "\t\t\t\tTest: " << intersection.geometryInOutside().global(quadPos) << " vs. " << geometryOut.local(quadPosGlobal) << std::endl;

				// The transposed inverse Jacobian of the map from the reference element
				// to the grid element
				const auto jacobian = geometryIntersection.jacobianInverseTransposed(quadPos);
				const auto jacobianIn = geometryIn.jacobianInverseTransposed(geometryIn.local(quadPosGlobal));
				const auto jacobianOut = geometryOut.jacobianInverseTransposed(geometryOut.local(quadPosGlobal));

				// The determinant term in the integral transformation formula
				const auto integrationElement = geometryIntersection.integrationElement(quadPos);

				//get local gradients
				std::vector<FieldMatrix<FltPrec,1,DomainDim>> referenceGradientsIn,referenceGradientsOut;
				localView.tree().finiteElement().localBasis().evaluateJacobian(geometryIn.local(quadPosGlobal), referenceGradientsIn);
				localViewOut.tree().finiteElement().localBasis().evaluateJacobian(geometryOut.local(quadPosGlobal), referenceGradientsOut);

				//get global gradients
				std::vector<FieldVector<FltPrec,DomainDim> > 	gradientsIn(referenceGradientsIn.size()), gradientsOut(referenceGradientsIn.size());
				for (size_t i=0; i<referenceGradientsIn.size(); i++) {
					jacobianIn.mv(referenceGradientsIn[i][0], gradientsIn[i]);
					jacobianOut.mv(referenceGradientsOut[i][0], gradientsOut[i]);
				}
				
				//save indices of global basis functions of both elements (i=intersection.inside() == localView, j = intersection.outside() = localViewOut) to
				//check later in the loop over each elements' dofs, whether the support is only in one element or both
				std::vector<int> globalIIndices(localView.size()),globalJIndices(localViewOut.size());
				for( int i = 0; i < localView.size(); i++ ) {
					globalIIndices[i] = localView.index(i);
					globalJIndices[i] = localViewOut.index(i);
				}
				
				//vector to save local indices of i (i=localView()=intersection.inner()) whose global functions do NOT have support in the other element
				//(other = j = localViewOut = intersection.outer())
				std::vector<int> extraContribution;
				for( int i = 0; i < localView.size(); i++ ) {
					const int globalI = localView.index(i);
					
					//for dofs not on the current intersection, the contribution is NOT computed in the inner loop, as the global basis function does
					//not occur in the opposite element.
					//on the otherhand, each intersection is visited twice (intersection.inside() & intersection.outside() the other way around),
					//thus basisfunctions on the very intersection are calculated twice.
					//therefore, the contribution for the first case needs to be calculated extra (loop below), and for the second case, the
					//contributions need to be halfed
					
					//save the local index in the other element (i.e.j=localViewOut=intersection.outer) of the current global basisfunction of the element (i.e. i=localView=intersection.inner)
					//needed to calculate the gradient jump if the currently considered basis function with local index i in the element has support also in the other element
					int localJIndexOfI = -1;
					const auto tmp2 = std::find(globalJIndices.begin(),globalJIndices.end(),globalI);
					if( tmp2 != globalJIndices.end() ) {
						localJIndexOfI = std::distance(globalJIndices.begin(), tmp2);
					}
					else {						
						//extra case 1
						//note: when intersection.inside() & intersection.outside() are inverted, for the 2nd element the
						//dofs w/ support just in the outside() element are computed
						extraContribution.push_back(i);
					}
					
					for( int j = 0; j < localViewOut.size(); j++ ) {
						const int globalJ = localViewOut.index(j);
					
						int localIIndexOfJ = -1;
						const auto tmp  = std::find(globalIIndices.begin(),globalIIndices.end(),globalJ);
						if( tmp != globalIIndices.end() ) {
							 localIIndexOfJ = std::distance(globalIIndices.begin(), tmp);
						}
						
						//compute the gradient jumps
						FieldVector<FltPrec,DomainDim> diffIn, diffOut;
						diffIn = gradientsIn[i];
						if(localJIndexOfI != -1) {
							diffIn -= gradientsOut[localJIndexOfI];
						}
						diffOut = -gradientsOut[j];
						if(localIIndexOfJ != -1) {
							diffOut += gradientsIn[localIIndexOfJ];
						}
						
						auto contribution = evaluateIntegrand(h_F,diffIn,diffOut) * integrationElement * quadPoint.weight();
						if( localIIndexOfJ != -1 && localJIndexOfI != -1 ) {
							//extra case 2
							//symmetry: if global basis functions are on the intersection, the contribution would be doubled when the inverse case is considered
							//(same intersection, inside & outside the other way around)
							contribution /= 2;
						}
						//use opaque action function to be able to use the same function for both
						//norm calculation and stiffness matrix creation
						action(globalI,globalJ,contribution);
					}
				}
				
				//contributions for dofs not on the intersection within intersection.inside()
				//when inside and outside are interchanged, the corresponding contributions
				//are calculated, therefore those terms are missing
				for( int i=0; i<extraContribution.size(); i++ ) {
					for( int j=0; j<extraContribution.size(); j++ ) {
						const auto localI1 = extraContribution[i];
						const auto localI2 = extraContribution[j];
						const auto globalI1 = localView.index(localI1);
						const auto globalI2 = localView.index(localI2);
						
						const auto diffIn = gradientsIn[localI1];
						const auto diffOut = gradientsIn[localI2];
						
						action(globalI1,globalI2, evaluateIntegrand(h_F,diffIn,diffOut) * integrationElement * quadPoint.weight() );
					}
				}
			}
		}
	}
}

// Compute the stiffness matrix for a single element
template<class LocalView, class Matrix, class Precision = typename Matrix::block_type>
void assembleElementStiffnessMatrix(const LocalView& localView,
                                    Matrix& elementMatrix,
                                    const std::function<
																						const FieldMatrix<Precision, LocalView::Element::dimension, LocalView::Element::dimension>(
																							const FieldVector<Precision, LocalView::Element::dimension>
																						)
																					> diffusion,
																		const Precision diffusionInfinityNorm,
                                    const std::function<
																						FieldVector<Precision, LocalView::Element::dimension>(
																							FieldVector<Precision, LocalView::Element::dimension>
																						)
																					> beta,
																		const Precision betaInfinityNorm,
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
    const FieldMatrix<Precision,dim,dim> localD = diffusion(element.geometry().global(quadPos));
    for (size_t i=0; i<gradients.size(); i++) {
      jacobian.mv(referenceGradients[i][0], gradients[i]);
      localD.mv(gradients[i], Dgradients[i]);
		}
    
    //shape function values for the derivative-free term
    std::vector<FieldVector<Precision, 1>> shapeFunctionValues;  
    localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);
    
    const auto localBeta = beta(element.geometry().global(quadPos));
    
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
        //In the convection term, p and q seem to be inverted, but rightful. Reason: p is the loop over the Rows, q over the columns. When turning the weak formulation into a
        //linear system, it starts with a(u,p_j) = a(\sum_i u_i p_i,p_j) = \sum_i u_i a(p_i,p_j), i.e. the columns are in the first argument, the rows in the second. This 
        //convention needs to be adapted.
        elementMatrix[localRow][localCol] += (Dgradients[p] * gradients[q] + (localBeta * gradients[q]) * shapeFunctionValues[p] + mu *  shapeFunctionValues[p] * shapeFunctionValues[q])
                                    * quadPoint.weight() * integrationElement;
      }
    }
  }
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
  auto localViewOut = basis.localView();

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
    
    for (const auto& intersection : intersections(gridView,element) ) {
			if( intersection.boundary() ) continue;
		
			localViewOut.bind(intersection.outside());
			for( size_t i =0; i < localView.size(); i++ ) {
				const auto row = localView.index(i);
				for( size_t j = 0; j < localViewOut.size(); j++ ) {
					const auto col = localViewOut.index(j);
					nb.add(row,col);
				}
			}
		}
  }
}

template<typename Basis, typename Element>
void processElement(const Basis& basis,
														Element& element,
                            BCRSMatrix<typename Basis::GridView::ctype>& matrix,
                            BlockVector<typename Basis::GridView::ctype>& b,
                            const std::function<
                                typename Basis::GridView::ctype(FieldVector<typename Basis::GridView::ctype,
                                                   Basis::GridView::dimension>)
                                               > volumeTerm, 
                            const std::function<
																		const FieldMatrix<typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension>(
																			const FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
																		)
																	> diffusion,
														const typename Basis::GridView::ctype diffusionInfinityNorm,
                            const std::function<
																		FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>(
																			FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
																		)
																> beta,
														const typename Basis::GridView::ctype betaInfinityNorm,
                            const typename Basis::GridView::ctype mu)
{
	auto localView = basis.localView();
	localView.bind(element);

	Matrix<typename Basis::GridView::ctype> elementMatrix;
	assembleElementStiffnessMatrix(localView, elementMatrix, diffusion, diffusionInfinityNorm, beta, betaInfinityNorm, mu);

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
	BlockVector<typename Basis::GridView::ctype> localB;
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
                            BCRSMatrix<typename Basis::GridView::ctype>& matrix,
                            BlockVector<typename Basis::GridView::ctype>& b,
                            const std::function<
                                const typename Basis::GridView::ctype(const FieldVector<typename Basis::GridView::ctype,
                                                   Basis::GridView::dimension>)
                                               > volumeTerm, 
                            const std::function<
																		const FieldMatrix<typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension>(
																			const FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
																		)
																	> diffusion,
														const typename Basis::GridView::ctype diffusionInfinityNorm,
                            const std::function<
																		const FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>(
																			const FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
																		)
																	> beta,
                            const typename Basis::GridView::ctype betaInfinityNorm,
                            const typename Basis::GridView::ctype mu)
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
	
  //OpenMP requires random access iterators for a parallel for loop,
  //which cannot be provided by dune. therefore use a master-slave
  //approach: the master thread generates a task for each partition
  //element, that is then computed by each slave thread.
  //the access to the global stiffness matrix / load vector is handled
  //using atomics, therefore no race conditions and write-updated-element
  //issues can arise.
	#pragma omp parallel if(basis.dimension() > 1e4)
	#pragma omp single
	{
		for(auto element = gridView.template begin<0>(); element != LastElement; element++)
		
			#pragma omp task default(none) firstprivate(element) shared(diffusion,diffusionInfinityNorm,betaInfinityNorm,mu,matrix,b,basis,beta,volumeTerm)
				processElement( basis, *element,matrix,b,volumeTerm,diffusion,diffusionInfinityNorm,beta, betaInfinityNorm,mu );

		#pragma omp taskwait
	}
}

class Triangle {
public:
	struct Bisection {};
	using CrissCross = Bisection;
	struct RedGreen {};
	struct Standard {};
	struct NonDelaunay {};
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

template <int dim>
class GridGenerator<Triangle::NonDelaunay, dim> {
public:
	using GridType = Dune::UGGrid<dim>;
	
	template < typename U = double >
	static
	auto generate(const std::array<U,dim> lowerLeftCorner, const std::array<U,dim> upperRightCorner, const uint edgeNumber) {
		assert((edgeNumber-1)%4 == 0);
		
		GridFactory<GridType> factory;
		
		const int PointsPerRow = edgeNumber;
		const U AdvancePerPointX = (upperRightCorner[0]-lowerLeftCorner[0]) / (edgeNumber-1);
		const U AdvancePerPointY = (upperRightCorner[1]-lowerLeftCorner[1]) / (edgeNumber-1);
		
		std::cerr << "\t" << "Starting /w points" << std::endl;
		
		for( int j=0; j<edgeNumber; j++ ) {
			for( int i=0; i<edgeNumber; i++ ) {
				if( j%2 == 1) {//odd rows
					if( i%4 != 0 and i%2 == 0) {
						factory.insertVertex({lowerLeftCorner[0] + i*AdvancePerPointX + 0.375*AdvancePerPointX, lowerLeftCorner[1] + j*AdvancePerPointY});
					}
					else {
						factory.insertVertex({lowerLeftCorner[0] + i*AdvancePerPointX, lowerLeftCorner[1] + j*AdvancePerPointY});
					}
				} else {//even rows
					if( i%4 == 0) {
						factory.insertVertex({lowerLeftCorner[0] + i*AdvancePerPointX, lowerLeftCorner[1] + j*AdvancePerPointY});
					} else {
						if( j%4 == 0) {
							factory.insertVertex({lowerLeftCorner[0] + i*AdvancePerPointX - 0.25*AdvancePerPointX, lowerLeftCorner[1] + j*AdvancePerPointY});
						} else {
							if( i%2 == 0 ) {
								factory.insertVertex({lowerLeftCorner[0] + i*AdvancePerPointX, lowerLeftCorner[1] + j*AdvancePerPointY});
							} else {
								factory.insertVertex({lowerLeftCorner[0] + i*AdvancePerPointX - 0.25*AdvancePerPointX, lowerLeftCorner[1] + j*AdvancePerPointY});
							}
						}
					}
				}
			}
		}
		
		std::cerr << "\t" << "Starting w/ elements" << std::endl;
		
		for( uint i=0; i < edgeNumber-1; i+=2 ) {
			for( uint j=0; j < edgeNumber-1; j+=2 ) {
				const uint Advance = i+PointsPerRow*j;
				factory.insertElement(GeometryTypes::triangle, {			0					+Advance,			 1				+Advance, PointsPerRow		+Advance});
				factory.insertElement(GeometryTypes::triangle, {PointsPerRow+1	+Advance, PointsPerRow	+Advance,			 1					+Advance});
				factory.insertElement(GeometryTypes::triangle, {PointsPerRow+1	+Advance,			 1				+Advance,PointsPerRow+2		+Advance});
				factory.insertElement(GeometryTypes::triangle, {			2					+Advance,PointsPerRow+2	+Advance,			 1					+Advance});
				
				factory.insertElement(GeometryTypes::triangle, {  2*PointsPerRow	+Advance,  PointsPerRow		+Advance,2*PointsPerRow+1	+Advance});
				factory.insertElement(GeometryTypes::triangle, {   PointsPerRow		+Advance, PointsPerRow+1	+Advance,2*PointsPerRow+1	+Advance});
				factory.insertElement(GeometryTypes::triangle, {  PointsPerRow+1	+Advance, PointsPerRow+2	+Advance,2*PointsPerRow+1	+Advance});
				factory.insertElement(GeometryTypes::triangle, {2*PointsPerRow+2	+Advance,2*PointsPerRow+1	+Advance, PointsPerRow+2	+Advance});
			}
		}
		
		std::cerr << "\t" <<  "Done inserting." << std::endl;
		
		return factory.createGrid();
	}
};

int main(int argc, char *argv[])
{
  // Set up MPI, if available
  MPIHelper::instance(argc, argv);
  
  const double PI = StandardMathematicalConstants<double>::pi();
  constexpr int Dim = 2;//\Omega\subset\mathbb R^{dim}, \Omega=(0,1)^2 (see below, other sets may not work)

  //~ const double mu = 1;
  const double mu = 0;
  const double eps = 1e-5;
  //~ const double eps = std::atof(argv[5]);
  //~ const double eps = 0.6;
  const double diffusionInfinityNorm = eps;
  const auto diffusion = [eps](const FieldVector<double,Dim>& x) -> const FieldMatrix<double,Dim,Dim> { FieldMatrix<double,Dim,Dim> tmp = {{100,std::cos(x[0])},{std::cos(x[0]),1}}; tmp *= eps; return tmp; };
  //~ const FieldMatrix<double, Dim, Dim> D = {{eps,0},{0,eps}};
  //~ const FieldMatrix<double, Dim, Dim> D = {{0,0},{0,0}};
  //~ const FieldMatrix<double, 2, 2> D = {{eps*2,eps*1},{eps*1,eps*3}};
  const double betaInfinityNorm = 1;
  //ex. 5.2 (opaper_followup)
  //~ const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {-x[1],x[0]}; };
  //ex. 5.3 (opaper_followup)
  //~ const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {std::cos(PI/3),std::sin(PI/3)}; };
  const auto beta = [=](const FieldVector<double,Dim>& x) -> const FieldVector<double,Dim> { return {2,1}; };
  //~ const FieldVector<double,Dim> beta = {2,1};
  //~ const FieldVector<double,Dim> beta = {std::atof(argv[5]),0.5*std::atof(argv[5])};
  //~ const FieldVector<double,Dim> beta = {0,0};
  //~ const FieldVector<double,Dim> beta = {1,0};
  
  constexpr int LagrangeOrder = 1;
  using GridMethod = Square::Standard;
  
  //~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x){return (2.0*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]);};
  //~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x){return (5.0*PI*PI*eps + mu) * sin(2*PI*x[0]) * sin(PI*x[1]);};
  //~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x){return eps*5*PI*PI*sin(PI*x[0])*sin(PI*x[1])-eps*2*PI*PI*cos(PI*x[0])*cos(PI*x[1])+mu*sin(PI*x[0])*sin(PI*x[1]);};
  //~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x){return (2.0*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]) + 2*PI*cos(PI*x[0])*sin(PI*x[1])+PI*sin(PI*x[0])*cos(PI*x[1]);};
  #warning "Usage of beta in the sourceTerm only works for constant convection!"
  //~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x) -> const double {return (2.0*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]) + beta({0,0})[0]*PI*cos(PI*x[0])*sin(PI*x[1])+beta({0,0})[1]*PI*sin(PI*x[0])*cos(PI*x[1]);};
  auto const sourceTerm = [=](const FieldVector<double,Dim>& x) -> const double {return 100*(2*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]) + 100*beta({0,0})[0]*PI*cos(PI*x[0])*sin(PI*x[1])+100*beta({0,0})[1]*PI*sin(PI*x[0])*cos(PI*x[1]);};
  //~ auto const sourceTerm = [=](const FieldVector<double,Dim>& x){return 0;};
  
	//~ auto const f = [=] (const auto& coords) { return exp(-std::pow(coords[0]-0.5,2)/0.2-3*std::pow(coords[1]-0.5,2)/0.2); };
	//~ auto const f = [=] (const auto& coords) { return std::sin(PI*coords[0])*std::sin(PI*coords[1]); };
	//~ auto const f = [=] (const auto& coords) { return std::sin(2*PI*coords[0])*std::sin(PI*coords[1]); };
	auto const f = [=] (const auto& coords) { return 100*sin(PI*coords[0])*sin(PI*coords[1]); };
	//~ auto const Df = [=](const auto& coords) -> FieldVector<double,Dim> { return f(coords) * FieldVector<double,Dim>{-2*(coords[0]-0.5)/0.2,-6*(coords[1]-0.5)/0.2 }; };
	//~ auto const Df = [=](const auto& coords) -> FieldVector<double,Dim> { return { PI*std::cos(PI*coords[0])*std::sin(PI*coords[1]), PI*std::sin(PI*coords[0])*std::cos(PI*coords[1]) }; };
	//~ auto const Df = [=](const auto& coords) -> FieldVector<double,Dim> { return { 2*PI*std::cos(2*PI*coords[0])*std::sin(PI*coords[1]), PI*std::sin(2*PI*coords[0])*std::cos(PI*coords[1]) }; };
	auto const Df = [=](const auto& x) -> FieldVector<double,Dim> { return { 100*PI*std::cos(PI*x[0])*std::sin(PI*x[1]), 100*PI*std::sin(PI*x[0])*std::cos(PI*x[1]) }; };
  
  //~ const auto kappaU = [=] (const FieldVector<double,2>& coords) { return std::pow(coords[0]-0.5,2)+std::pow(coords[1]-0.5,2); };
  //~ const auto kappaL = [=] (const FieldVector<double,2>& coords) { return -std::pow(coords[0]-0.5,2)-std::pow(coords[1]-0.5,2); };
  const auto kappaU = [=] (const FieldVector<double,2>& coords) { return 100; };
  const auto kappaL = [=] (const FieldVector<double,2>& coords) { return 0; };
  
  if( argc < 4 ) {
		std::cerr << argv[0] << " <Edges> <omega> <CIP-gamma>" << std::endl;
		return -1;
	}
  const unsigned int edges = std::atoi(argv[1]);
  const double omega = std::atof(argv[2]);
  const double gamma = std::atof(argv[3]);
  
  std::cerr << "Configuration: " << std::endl
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
  
  auto grid =	 GridGenerator<GridMethod, Dim>::generate( {0,0}, {1,1}, edges );
  using Grid = GridGenerator<GridMethod, Dim>::GridType;

  using GridView = Grid::LeafGridView;
  GridView gridView = grid->leafGridView();
  
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
  
  std::cout << "Testing getSVector:" << std::endl;
  const auto sVectorStart = std::chrono::high_resolution_clock::now();
  const auto sVector = getSVector( basis, diffusionInfinityNorm, betaInfinityNorm, mu );
  const auto sVectorEnd = std::chrono::high_resolution_clock::now();
  const auto uniformityCheck = [](const BlockVector<double>& vec) { bool result=true; for(int i=1;i<vec.size();i++) { if(std::abs(vec[i]-vec[i-1])>=1e-5) { result=false;break;}} return result;};
  std::cout << "\tTest 1: max element vs. Diameter: " << *std::max_element(sVector.begin(),sVector.end()) << " vs. " << (Diameter*Diameter*mu+Diameter*betaInfinityNorm+diffusionInfinityNorm) << std::endl;
  std::cout << "\tTest 2: min element: " << *std::min_element(sVector.begin(),sVector.end()) << std::endl;
  std::cout << "\tTest 3: uniformity: " << (uniformityCheck(sVector) ? "yes" : "no" ) << std::endl;
  std::cout << "Time for generation: " << std::chrono::duration<float,std::milli>(sVectorEnd-sVectorStart).count() << " ms." << std::endl;

  std::cerr << "Assemble Problem" << std::endl;
  assembleProblem(basis, stiffnessMatrix, b, sourceTerm, diffusion, diffusionInfinityNorm, beta, betaInfinityNorm, mu);
  std::cerr << "Assemble Problem End" << std::endl;
  
  const auto stiffnessMatrixBeforeCIP = std::get<0>(transcribeDuneToEigen( stiffnessMatrix, b, 85 ));
  addCIP(basis,stiffnessMatrix,betaInfinityNorm,gamma);
	const auto stiffnessMatrixAfterCIP = std::get<0>(transcribeDuneToEigen( stiffnessMatrix, b, 85 ));

  // Determine Dirichlet dofs by marking all degrees of freedom whose Lagrange nodes
  // comply with a given predicate.
  auto predicate = [](const auto x)
  {
		const bool ret = 1e-5 > x[0] || x[0] > 0.99999 || 1e-5 > x[1] || x[1] > 0.99999; //everywhere
		//ex. 5.2
		//~ const bool ret = x[0] > 0.99999 || 1e-5 > x[1];
    return ret;
  };

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

  // Set Dirichlet values
  auto dirichletValues = [](const auto x) -> const double
  {
    return 0;
    //ex. 5.2 (opaper_followup)
    //~ if( x[0] < 0.333333 and x[1] < 1e-5) {
			//~ return 0;
		//~ }
		//~ else if ( 0.333333 < x[0] and x[0] < 0.6666667 and x[1] < 1e-5) {
			//~ return 0.5;
		//~ }
		//~ else {
			//~ return 1;
		//~ }
		//ex. 5.3 (opaper_followup)
		//~ if( x[0] < 1e-5 or x[1] > 0.9999) {
			//~ return 1;
		//~ }
		//~ else {
			//~ return 0;
		//~ }
  };
  Functions::interpolate(basis,b,dirichletValues, dirichletNodes);
    
  const Vector Rhs(b);
  std::cerr << "Modify Dirichlet Rows End" << std::endl;
  
  //---
  //tweak symmetry accordingly if needed
  //~ storeMatrixMarket(stiffnessMatrix, "stiffness.mtx");
  //~ storeMatrixMarket(b, "stiffness-rhs.mtx");
  
  //prepare vectors for non-const bounds
  Eigen::Array<double,Eigen::Dynamic,1> uKappaU(b.size());
  Eigen::Array<double,Eigen::Dynamic,1> uKappaL(b.size());
  Functions::interpolate(basis,uKappaU,kappaU);
  Functions::interpolate(basis,uKappaL,kappaL);
  
  outputVector<double>(basis,uKappaU);

  ///////////////////////////
  //   Compute solution
  ///////////////////////////

  // Choose an initial iterate that fulfills the Dirichlet conditions
  Vector x(basis.size());
  x = b;

	{
		std::cerr << "Solving" << std::endl;
		const auto solverStart = std::chrono::high_resolution_clock::now();
		// Turn the matrix into a linear operator
		MatrixAdapter<Matrix,Vector,Vector> linearOperator(stiffnessMatrix);

		// Sequential incomplete LU decomposition as the preconditioner
		SeqILU<Matrix,Vector,Vector> preconditioner(stiffnessMatrix,
																								1.0);  // Relaxation factor

		// Preconditioned conjugate gradient solver
		BiCGSTABSolver<Vector> cg(linearOperator,
		//~ CGSolver<Vector> cg(linearOperator,
												preconditioner,
												1e-5, // Desired residual reduction factor
												50,   // Maximum number of iterations
												2);   // Verbosity of the solver
		Dune::UMFPack<Matrix> solver(stiffnessMatrix, 0);

		// Object storing some statistics about the solving process
		InverseOperatorResult statistics;

		// Solve!
		//~ cg.apply(x, b, statistics);
		solver.apply(x, b, statistics);
		const auto solverEnd = std::chrono::high_resolution_clock::now();
		std::cerr << "\tTook: " << std::chrono::duration<float,std::milli>(solverEnd-solverStart).count() << " ms." << std::endl;
		std::cerr << "Solving End" << std::endl;
	}
	
	std::cerr << "(Dune) ||u_h^0-f||: h = " << H << ", " << L2Norm<double>( gridView, basis.localView(), x, f) << std::endl;
	outputVector<double>(basis,x,std::ios::trunc, "test_output");
	
	//-----------------------------------------------
	//Eigen-Version of solving	
	
	std::cerr << "Transcribe to Eigen" << std::endl;
	auto [stiffnessEigen, RhsEigen] = transcribeDuneToEigen( stiffnessMatrix, Rhs, 85 ); //85 for occupation should be enough for all grids and lagrange elements up to 3
	//~ Eigen::saveMarket(stiffnessEigen, "stiffness.mtx");
	std::cerr << "Transcribe to Eigen End" << std::endl;
	
	const Eigen::Vector<double,Eigen::Dynamic> u0 = transcribeDuneToEigen(x);
	const Eigen::Vector<double,Eigen::Dynamic> eigenSVector = transcribeDuneToEigen(sVector);
	
	//~ Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,Eigen::Lower|Eigen::Upper> solver;
	//~ Eigen::BiCGSTAB<Eigen::SparseMatrix<double,Eigen::RowMajor> > solver;
	//~ solver.setTolerance(1e-9);
	//~ solver.compute(stiffnessEigen);
	//~ Eigen::SparseLU<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int> > solver;
	Eigen::UmfPackLU<Eigen::SparseMatrix<double>> solver;
	solver.analyzePattern(stiffnessEigen);
	solver.factorize(stiffnessEigen);
	
	const Eigen::Vector<double,Eigen::Dynamic> uEigen = solver.solve(RhsEigen);
	std::cerr << "(Eigen) ||u_h^0-f||: h = " << H << ", " << L2Norm<double>( gridView, basis.localView(), uEigen, f) << std::endl;
	outputVector<double>(basis,uEigen,std::ios::trunc, "test_output_eigen");
	
	if( 1.0 / H <= 20 && LagrangeOrder == 1) {
		std::cout << stiffnessEigen << std::endl;
		std::cout << "Before CIP:" << std::endl;
		std::cout << stiffnessMatrixBeforeCIP << std::endl;
		std::cout << "After CIP/Before Dirichlet:" << std::endl;
		std::cout << stiffnessMatrixAfterCIP << std::endl;
		std::cout << "CIP-Modifications:" << std::endl;
		std::cout << stiffnessMatrixAfterCIP-stiffnessMatrixBeforeCIP << std::endl;
		std::cout << "CIP-Eigenvals: " << std::endl << Eigen::MatrixXd(stiffnessMatrixAfterCIP-stiffnessMatrixBeforeCIP).eigenvalues() << std::endl;
		std::cout << "total stiffnessmatrix Eigenvals: " << std::endl << Eigen::MatrixXd(stiffnessEigen).eigenvalues() << std::endl;
		std::cout << "total stiffnessmatrix det: " << std::endl << Eigen::MatrixXd(stiffnessEigen).determinant() << std::endl;
		std::cout << "CIP-Symmetry check: " << (Eigen::MatrixXd(stiffnessMatrixAfterCIP-stiffnessMatrixBeforeCIP) - Eigen::MatrixXd(stiffnessMatrixAfterCIP-stiffnessMatrixBeforeCIP).transpose()).squaredNorm() << std::endl;
		std::cout << "stiffness Symmetry check: " << (Eigen::MatrixXd(stiffnessEigen) - Eigen::MatrixXd(stiffnessEigen).transpose()).squaredNorm() << std::endl;
		std::cout << "Rhs:" << std::endl << RhsEigen << std::endl;
	}
	
	//for fem-only testing
	//~ return 0;
	
	//--------------------------------------
	const auto L2NormBind = [&gridView,&basis](const auto& u) { return L2Norm<double>( gridView, basis.localView(), u  ); };
	const auto OutputMethodBind = [&basis](const auto& u, const std::ios::openmode mode, const std::string filename) { return outputVector<double>( basis, u, mode, filename ); };
	
	//Newton-Method
	const auto newtonStart = std::chrono::high_resolution_clock::now();
	Eigen::Vector<double,Eigen::Dynamic> eigenU
		= newtonMethod<double>( basis, stiffnessEigen, RhsEigen, u0, eigenSVector,Diameter, uKappaU, uKappaL, L2NormBind );
	const auto newtonEnd = std::chrono::high_resolution_clock::now();
	std::cerr << "\tTook " << std::chrono::duration<float,std::milli>(newtonEnd-newtonStart).count() << " ms." << std::endl;
	std::cerr << "H = " << H << std::endl;
	std::cerr << "(Newton|Eigen) ||u^+-f||_L2: " << L2Norm<double>( gridView, basis.localView(), eigenU.array().min(uKappaU).max(uKappaL).matrix(), f) << std::endl;
	std::cerr << "(Newton|Eigen) ||u^+-f||_A = " << ANorm( gridView, basis.localView(), eigenU.array().min(uKappaU).max(uKappaL).matrix(), diffusion, diffusionInfinityNorm, mu, f, Df ) << std::endl;
	std::cerr << "(Newton|Eigen) ||u^+-f||_CIP = " << cipNorm( basis, eigenU.array().min(uKappaU).max(uKappaL).matrix(), diffusion, diffusionInfinityNorm, betaInfinityNorm, mu, gamma, f, Df ) << std::endl;
	std::cerr << "(Newton|Eigen) ||u^-||_s = " << sNorm(eigenU - eigenU.array().min(uKappaU).max(uKappaL).matrix(), diffusionInfinityNorm,betaInfinityNorm,mu,Diameter) << std::endl;
	
	//~ //for newton-only testing
	//~ return 0;
	
	//-----------
	Eigen::Vector<double,Eigen::Dynamic> eigenX = fixedpointMethod( u0, stiffnessEigen, RhsEigen, omega, eigenSVector, Diameter, uKappaU, uKappaL, L2NormBind, OutputMethodBind );
	
	std::cerr << "(Richard|Eigen) ||eigenX-f||: h = " << H << ", " << L2Norm<double>( gridView, basis.localView(), eigenX, f) << std::endl;
	std::cerr << "(Richard|Eigen) ||eigen u^+-f||_L2: " << L2Norm<double>( gridView, basis.localView(), eigenX.array().min(uKappaU).max(uKappaL).matrix(), f) << std::endl;
	std::cerr << "(Richard|Eigen) ||eigen u^+-f||_A = " << ANorm( gridView, basis.localView(), eigenX.array().min(uKappaU).max(uKappaL).matrix(), diffusion, diffusionInfinityNorm, mu, f, Df ) << std::endl;
	std::cerr << "(Richard|Eigen) ||eigen u^+-f||_CIP = " << cipNorm( basis, eigenX.array().min(uKappaU).max(uKappaL).matrix(), diffusion,diffusionInfinityNorm, betaInfinityNorm, mu, gamma, f, Df ) << std::endl;
	std::cerr << "(Richard|Eigen) ||eigen u^-||_s = " << sNorm(eigenX - eigenX.array().min(uKappaU).max(uKappaL).matrix(), diffusionInfinityNorm,betaInfinityNorm,mu,Diameter) << std::endl;
	
	//-----------
	const auto normalsolution(x);
	
	x = fixedpointMethod( x, stiffnessMatrix, Rhs, omega, sVector, Diameter, uKappaU, uKappaL, L2NormBind, OutputMethodBind );
	Vector uplus(x.size()), uminus(x.size());
	
	for( int i = 0; i < x.size(); i++ ) {
		uplus[i] = std::clamp(x[i],uKappaL[i],uKappaU[i]);
		uminus[i] = x[i] - uplus[i];
	}
	
	//--------------
	//some norms
	std::cerr << "||u_0-f||_2: h = " << H << ", " << L2Norm<double>( gridView, basis.localView(), normalsolution, f) << std::endl;

	std::cerr << "||u_N-f||_2 = " << L2Norm<double>( gridView, basis.localView(), x, f) << std::endl;
	std::cerr << "(Richard|Dune) ||u^+-f||_L2 = " << L2Norm<double>( gridView, basis.localView(), uplus, f) << std::endl;
	
	auto const ANormMaybeWorking = [&](BCRSMatrix<double>& A, BlockVector<double> u) { BlockVector<double> _tmp(u.size()); A.mv(u,_tmp); return std::sqrt(u*_tmp); };
	BlockVector<double> tmp;
	Functions::interpolate(basis,tmp,f);
	tmp -= uplus;
	
	std::cerr << "(Richard|Dune) ||u^+-u_0||_A = " << ANormMaybeWorking( stiffnessMatrix, tmp) << std::endl;
	std::cerr << "(Richard|Dune) ||u^+-f||_A = " << ANorm( gridView, basis.localView(), uplus, diffusion,diffusionInfinityNorm, mu, f, Df ) << std::endl;
	std::cerr << "(Richard|Dune) ||u^+-f||_CIP = " << cipNorm( basis, uplus, diffusion,diffusionInfinityNorm, betaInfinityNorm, mu, gamma, f, Df ) << std::endl;
	std::cerr << "(Richard|Dune) ||u^-||_s = " << sNorm(uminus, diffusionInfinityNorm,betaInfinityNorm,mu,Diameter) << std::endl;
	
	BlockVector<double> tmp2(Rhs), tmp3(Rhs.size());
	stiffnessMatrix.mv(uplus, tmp3);
	tmp2 -= tmp3;
	for( int i = 0; i < tmp2.size(); i++ ) {
		tmp2[i] -= sVector[i] * uminus[i];
	}
	std::cerr << "||Rhs - Auplus - s(uminus)||_\\infty = " << std::get<0>(inftyNorm(tmp2)) << std::endl;
	
	Eigen::Vector<double,Eigen::Dynamic> duneXtranscribed = ([](const auto& u){ Eigen::Vector<double,Eigen::Dynamic> tmp(u.size()); for( int i=0; i<u.size();i++ ) tmp[i]=u[i]; return tmp; })(x);
	std::cerr << "||u^+_eigen - u^+_dune||_L2 = " << L2Norm<double>( gridView, basis.localView(), eigenU.array().min(uKappaU).max(uKappaL).matrix()-duneXtranscribed.array().min(uKappaU).max(uKappaL).matrix() ) << std::endl;
	
	return 0;
}
