#include <config.h>

#include <iostream>
#include <iomanip>
#include <vector>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/yaspgrid.hh>
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


using namespace Dune;

// Compute the stiffness matrix for a single element
template<class LocalView, class Matrix, class Precision = typename Matrix::block_type>
void assembleElementStiffnessMatrix(const LocalView& localView,
                                    Matrix& elementMatrix,
                                    const FieldMatrix<Precision, 2, 2> D,
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
  int order = 2 * (localFiniteElement.localBasis().order()-1);
  std::cout << "Order = " << order << std::endl;
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
        elementMatrix[localRow][localCol] += (-Dgradients[p] * gradients[q] + mu *  shapeFunctionValues[p] * shapeFunctionValues[q])
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
	std::cout << "Matrix:" << std::endl;
	for( int i = 0; i < elementMatrix.N(); i++ ) {
		for( int j = 0; j < elementMatrix.M(); j++ ) {
			std::cout << std::setw(12) << (elementMatrix.exists(i,j) ? elementMatrix[i][j] : 0);
		}
		std::cout << std::endl;
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
  int order = dim;
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


template<class Basis>
void assembleProblem(const Basis& basis,
                            BCRSMatrix<double>& matrix,
                            BlockVector<double>& b,
                            const std::function<
                                double(FieldVector<double,
                                                   Basis::GridView::dimension>)
                                               > volumeTerm, 
                            const FieldMatrix<double, 2, 2> D, //Vektorwertigkeit unschön
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

  for (const auto& element : elements(gridView))
  {
    // Now let's get the element stiffness matrix
    // A dense matrix is used for the element stiffness matrix
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
      b[row] += localB[p];
    }
  }
}


int main(int argc, char *argv[])
{
  // Set up MPI, if available
  MPIHelper::instance(argc, argv);

  const double PI = StandardMathematicalConstants<double>::pi();

  const double mu = 1;
  const double eps = 1e-5;
  const FieldMatrix<double, 2, 2> D = {{eps,0},{0,eps}};
  const double UpperBound = 0.8;
  const int edges = 1;

  //////////////////////////////////
  //   Generate the grid
  //////////////////////////////////

  constexpr int dim = 2;
  using Grid = YaspGrid<dim>;
  auto grid = std::make_shared<Grid>( Dune::FieldVector<double,2>{1, 1}, std::array{edges, edges} );

  //~ grid->globalRefine(2);

  using GridView = Grid::LeafGridView;
  GridView gridView = grid->leafGridView();

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

  Functions::LagrangeBasis<GridView,2> basis(gridView);

  auto sourceTerm = [=](const FieldVector<double,dim>& x){return (2*PI*PI*eps + mu) * sin(PI*x[0]) * sin(PI*x[1]);};
  assembleProblem(basis, stiffnessMatrix, b, sourceTerm, D, mu);
  
  //@Debug

  // Determine Dirichlet dofs by marking all degrees of freedom whose Lagrange nodes
  // comply with a given predicate.
  auto predicate = [](auto x)
  {
		//@Debug
		bool ret = 1e-5 > x[0] || x[0] > 0.999 || 1e-5 > x[1] || x[1] > 0.999;
		std::cout << "x = " << x << " : " << (ret ? 1 : 0) << std::endl;
    //~ return x[0] < 1e-5
        //~ || x[1] < 1e-5
        //~ || x[0] > 0.999
        //~ || x[1] > 0.999;
    //~ return !(1e-5 < x[0] && x[0] < 0.999 & 1e-5 < x[1] && x[1] < 0.999)
    return ret;
  };

  // Evaluating the predicate will mark all Dirichlet degrees of freedom
  std::vector<bool> dirichletNodes;
  Functions::interpolate(basis, dirichletNodes, predicate);
  //@Debug
  std::cout << "Dirichlet-Nodes: ";
  for( auto const& val :  dirichletNodes) std::cout << (val ? 1 : 0) << " ";
  std::cout << std::endl;
  std::vector<FieldVector<double, 1>> testShapeFunctionValues;  
	//~ basis.evaluateFunction(FieldVector<double,2>{0.5,0.5}, testShapeFunctionValues);
	//~ std::cout << testShapeFunctionValues << std::endl;

  ///////////////////////////////////////////
  //   Modify Dirichlet rows
  ///////////////////////////////////////////
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

  /////////////////////////////////////////////////////////////////////////////
  // Write matrix and load vector to files, to be used in later examples
  /////////////////////////////////////////////////////////////////////////////
  std::string baseName = "getting-started-poisson-fem-"
                       + std::to_string(grid->maxLevel()) + "-refinements";
  storeMatrixMarket(stiffnessMatrix, baseName + "-matrix.mtx");
  storeMatrixMarket(b, baseName + "-rhs.mtx");

  ///////////////////////////
  //   Compute solution
  ///////////////////////////

  // Choose an initial iterate that fulfills the Dirichlet conditions
  Vector x(basis.size());
  x = b;
  
  	//~ std::cout << "Matrix:" << std::endl;
	//~ for( int i = 0; i < stiffnessMatrix.N(); i++ ) {
		//~ for( int j = 0; j < stiffnessMatrix.M(); j++ ) {
			//~ std::cout << std::setw(5) << (stiffnessMatrix.exists(i,j) ? stiffnessMatrix[i][j] : 0);
		//~ }
		//~ std::cout << std::endl;
	//~ }

	{
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
	}

  //~ // Output result
  //~ VTKWriter<GridView> vtkWriter(gridView);
  //~ vtkWriter.addVertexData(x, "solution");
  //~ vtkWriter.write("getting-started-poisson-fem-result");
  
	/////////////////////////////////////////////////////////////////////
  // Some tests for further assessment for new features
  auto localView = basis.localView();
  
  for( const auto& element : elements(basis.gridView()) ) {
		std::cout << "Binding element" << std::endl;
		localView.bind(element);
		std::cout << "Size: " << localView.size() << std::endl;
		auto geo = element.geometry();
		std::cout << geo.corners() << " corners" << std::endl;
		for( int i = 0; i < geo.corners(); i++ ) {
			std::cout << "Corner " << i << "(" << geo.corner(i) << ")" << '\t' << x[localView.index(i)] << " (global index " << localView.index(i) << ")" << std::endl;
		}
			//~ for( const auto& vertex : vertices(localView.tree().gridView()) ) {
				//~ std::cout << vertex.geometry().corner(0) << std::endl;
		localView.unbind();
  }
  
  MultipleCodimMultipleGeomTypeMapper<GridView> mapper( gridView, mcmgVertexLayout() );
  std::cout << x.size() << " vs. " << mapper.size() << std::endl;
  
  int numVertices = 0;
  for( auto const& vertex : vertices(gridView) ) {
		numVertices++;
		std::cout << "Vertex has index " << gridView.indexSet().index(vertex) << ", u[index] = " << x[gridView.indexSet().index(vertex)] << std::endl;
	}
	std::cout << numVertices << std::endl;
  
	//only works for codim-0-elements
	//~ auto stVertex = gridView.begin<2>();
	//~ for ( const auto& intersects : intersections( gridView, *stVertex ) ) {
		//~ std::cout << intersects.type() << std::endl;
	//~ }
	
	//-----------
	const auto normalSolution(x);
	//~ VTKWriter<GridView> vtkWriter(gridView);
  //~ vtkWriter.addVertexData(normalSolution, "solution");
  //~ vtkWriter.write("getting-started-poisson-fem-result");
	
	//-----------------
	Vector uplus(basis.size());
	Vector uminus(x);
	Vector sVector(basis.size());
	Vector newB(basis.size());
	Vector y(basis.size());
	
	std::cout << "Matrix:" << std::endl;
	for( int i = 0; i < stiffnessMatrix.N(); i++ ) {
		for( int j = 0; j < stiffnessMatrix.M(); j++ ) {
			std::cout << std::setw(12) << (stiffnessMatrix.exists(i,j) ? stiffnessMatrix[i][j] : 0);
		}
		std::cout << std::endl;
	}
	std::cout << "matrix(end,1) = " << (stiffnessMatrix.exists(stiffnessMatrix.N()-1,0) ? stiffnessMatrix[stiffnessMatrix.N()-1][0] : 0) << std::endl;
	
	int n = 300;
	do {
		//@Debug
		//~ std::cerr << x[12] << std::endl;
		for( int i = 0; i < x.size(); i++ ) {
			uplus[i] = std::clamp(x[i],0.0,UpperBound);
		}
		uminus = x;
		uminus -= uplus;
		
		sVector = uminus;
		sVector *= std::sqrt(2)*eps + 2.0/ std::pow(static_cast<double>(edges), 2.0);
		//~ std::cout << "s-Multiplikator = " << std::sqrt(2)*eps + 2.0/ std::pow(static_cast<double>(edges), 2.0) << std::endl;
		//~ for( auto const& vertex : vertices(gridView) ) {
			//~ const int idx = gridView.indexSet().index(vertex);
			//~ sVector[idx] = (std::sqrt(2)*eps + 2.0/ std::pow(static_cast<double>(edges), 2.0) ) * uminus[idx];
		//~ }
		Vector aUplus(basis.size()), Au(basis.size());
		stiffnessMatrix.mv( x, Au );
		stiffnessMatrix.mv( uplus, aUplus );
		
		for( int i = 0; i < newB.size(); i++ ) {
			//~ newB[i] = x[i] + 1*(Rhs[i] - uplus[i] - sVector[i]);
			//~ newB[i] = x[i] + 1*(Rhs[i] - aUplus[i] - sVector[i]);
			newB[i] = Au[i] + 0.01*(Rhs[i] - aUplus[i] - sVector[i]);
		}
		
		Vector diffRhsNewB(Rhs), diffXuPlus(x), diffAUAUplus(Au);
		diffRhsNewB -= newB;
		diffXuPlus -= uplus;
		diffAUAUplus -= aUplus;
		std::cout	<< std::setw(15) << "x"
							<< std::setw(15) << "Au"
							<< std::setw(15) << "aUplus"
							<< std::setw(15) << "diff Au/Au+"
							<< std::setw(15) << "uminus"
							<< std::setw(15) << "uplus"
							<< std::setw(15) << "diff x/u+"
							<< std::setw(15) << "sVector"
							<< std::setw(15) << "Rhs"
							<< std::setw(15) << "newB"
							<< std::setw(15) << "diff"
							<< std::endl;
		for( int i = 0; i < newB.size(); i++ ) {
			std::cout << std::setw(15) << x[i]
								<< std::setw(15) << Au[i]
								<< std::setw(15) << aUplus[i]
								<< std::setw(15) << diffAUAUplus[i]
								<< std::setw(15) << uminus[i]
								<< std::setw(15) << uplus[i]
								<< std::setw(15) << diffXuPlus[i]
								<< std::setw(15) << sVector[i]
								<< std::setw(15) << Rhs[i]
								<< std::setw(15) << newB[i]
								<< std::setw(15) << diffRhsNewB[i]
								<< std::endl;
		}
		
		y = x;
		
		MatrixAdapter<Matrix,Vector,Vector> linearOperator(stiffnessMatrix);

		// Sequential incomplete LU decomposition as the preconditioner
		SeqILU<Matrix,Vector,Vector> preconditioner(stiffnessMatrix,
                                              1.0);  // Relaxation factor
		CGSolver<Vector> cg(linearOperator,
                      preconditioner,
                      1e-5, // Desired residual reduction factor
                      20000,   // Maximum number of iterations
                      2);   // Verbosity of the solver

  // Object storing some statistics about the solving process
		InverseOperatorResult statistics;

		// Solve!
		cg.apply(y, newB, statistics);
		
		auto tmp(y);
		tmp -= x;
		if( tmp.two_norm() < 1e-12 or (--n <= 0) ) break;
		x = y;
	}
	while( true );
	std::cout << "Stopped with " << n << " remaining iterations." << std::endl;
	
	//~ vtkWriter.addVertexData(x, "boundedSolution");
	//~ vtkWriter.write("getting-started-poisson-fem-result");	
}
