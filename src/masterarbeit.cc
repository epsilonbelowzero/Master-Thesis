#include <config.h>

#include <vector>

#include <dune/geometry/quadraturerules.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/matrixmarket.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>


using namespace Dune;

// Compute the stiffness matrix for a single element
template<class LocalView, class Matrix>
void assembleElementStiffnessMatrix(const LocalView& localView,
                                    Matrix& elementMatrix)
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
  const auto& quadRule = QuadratureRules<double, dim>::rule(element.type(),
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
    std::vector<FieldMatrix<double,1,dim> > referenceGradients;
    localFiniteElement.localBasis().evaluateJacobian(quadPos,
                                                     referenceGradients);

    // Compute the shape function gradients on the grid element
    std::vector<FieldVector<double,dim> > gradients(referenceGradients.size());
    for (size_t i=0; i<gradients.size(); i++)
      jacobian.mv(referenceGradients[i][0], gradients[i]);

    // Compute the actual matrix entries
    for (size_t p=0; p<elementMatrix.N(); p++)
    {
      auto localRow = localView.tree().localIndex(p);
      for (size_t q=0; q<elementMatrix.M(); q++)
      {
        auto localCol = localView.tree().localIndex(q);
        elementMatrix[localRow][localCol] += (gradients[p] * gradients[q])
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
void assemblePoissonProblem(const Basis& basis,
                            BCRSMatrix<double>& matrix,
                            BlockVector<double>& b,
                            const std::function<
                                double(FieldVector<double,
                                                   Basis::GridView::dimension>)
                                               > volumeTerm)
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
    assembleElementStiffnessMatrix(localView, elementMatrix);

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

  //////////////////////////////////
  //   Generate the grid
  //////////////////////////////////

  constexpr int dim = 2;
  using Grid = UGGrid<dim>;
  std::shared_ptr<Grid> grid = GmshReader<Grid>::read("l-shape.msh");

  grid->globalRefine(2);

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

  Functions::LagrangeBasis<GridView,1> basis(gridView);

  auto sourceTerm = [](const FieldVector<double,dim>& x){return -5.0;};
  assemblePoissonProblem(basis, stiffnessMatrix, b, sourceTerm);

  // Determine Dirichlet dofs by marking all degrees of freedom whose Lagrange nodes
  // comply with a given predicate.
  auto predicate = [](auto x)
  {
    return x[0] < 1e-8
        || x[1] < 1e-8
        || (x[0] > 0.4999 && x[1] > 0.4999);
  };

  // Evaluating the predicate will mark all Dirichlet degrees of freedom
  std::vector<bool> dirichletNodes;
  Functions::interpolate(basis, dirichletNodes, predicate);

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
    return (x[0]< 1e-8 || x[1] < 1e-8) ? 0 : 0.5;
  };
  Functions::interpolate(basis,b,dirichletValues, dirichletNodes);

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

  // Turn the matrix into a linear operator
  MatrixAdapter<Matrix,Vector,Vector> linearOperator(stiffnessMatrix);

  // Sequential incomplete LU decomposition as the preconditioner
  SeqILU<Matrix,Vector,Vector> preconditioner(stiffnessMatrix,
                                              1.0);  // Relaxation factor

  // Preconditioned conjugate gradient solver
  CGSolver<Vector> cg(linearOperator,
                      preconditioner,
                      1e-5, // Desired residual reduction factor
                      50,   // Maximum number of iterations
                      2);   // Verbosity of the solver

  // Object storing some statistics about the solving process
  InverseOperatorResult statistics;

  // Solve!
  cg.apply(x, b, statistics);

  // Output result
  VTKWriter<GridView> vtkWriter(gridView);
  vtkWriter.addVertexData(x, "solution");
  vtkWriter.write("getting-started-poisson-fem-result");
}
