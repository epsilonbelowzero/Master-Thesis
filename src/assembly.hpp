#pragma once

#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>

#include "stabilisation_term.hpp"

//Contains assembly routings for stiffness matrix, load vector and S matrix

//the following assembling routines are adapted from the example for the
//Laplace equation in the DUNE book by Prof. Sander
//There are some additions: the CIP term, and assembleProblem is split
//into two functions: assembleProblem and processElement. This is necessary
//to use OpenMP to speed up assembly.

// Get the occupation pattern of the stiffness matrix
template<class Basis>
void getOccupationPattern(const Basis& basis, Dune::MatrixIndexSet& nb)
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
    
    //add occupation for the J(.,.) term (stabilisation)
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

//calculates the J(.,.)-term eq. (2.22) and adds it onto the stiffness matrix
template < class Basis >
void addCIP(const Basis& basis,
			Dune::BCRSMatrix<typename Basis::GridView::ctype>& stiffnessMatrix,
			const std::function<Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>)> beta,
			const typename Basis::GridView::ctype gamma)
{
	using FltPrec = typename Basis::GridView::ctype;
	
	const auto action = [&stiffnessMatrix](const int globalI,const int globalJ, const FltPrec contrib) {
		stiffnessMatrix[globalI][globalJ] += contrib;
	};
	addCIPImpl(basis,action,beta,gamma);
}

// Compute the stiffness matrix for a single element
template<class LocalView, class Matrix, class Precision = typename Matrix::block_type>
void assembleElementStiffnessMatrix(const LocalView& localView,
                                    Matrix& elementMatrix,
                                    const std::function<const Dune::FieldMatrix<Precision, LocalView::Element::dimension, LocalView::Element::dimension>(
															const Dune::FieldVector<Precision, LocalView::Element::dimension>
									)> diffusion,
                                    const std::function<Dune::FieldVector<Precision, LocalView::Element::dimension>(
															Dune::FieldVector<Precision, LocalView::Element::dimension>
									)> beta,
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
  
  const auto& quadRule = Dune::QuadratureRules<Precision, dim>::rule(element.type(),
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
    std::vector<Dune::FieldMatrix<Precision,1,dim> > referenceGradients;
    localFiniteElement.localBasis().evaluateJacobian(quadPos,
                                                     referenceGradients);

    // Compute the shape function gradients on the grid element
    // Dgradients: multiply the gradient w/ matrix D
    std::vector<Dune::FieldVector<Precision,dim> > gradients(referenceGradients.size());
    std::vector<Dune::FieldVector<Precision,dim> > Dgradients(referenceGradients.size());
    const Dune::FieldMatrix<Precision,dim,dim> localD = diffusion(element.geometry().global(quadPos));
    for (size_t i=0; i<gradients.size(); i++) {
      jacobian.mv(referenceGradients[i][0], gradients[i]);
      localD.mv(gradients[i], Dgradients[i]);
	}
    
    //shape function values for the derivative-free term
    std::vector<Dune::FieldVector<Precision, 1>> shapeFunctionValues;  
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
template<class LocalView, class FltPrec>
void assembleElementVolumeTerm(
        const LocalView& localView,
        Dune::BlockVector<FltPrec>& localB,
        const std::function<FltPrec(Dune::FieldVector<FltPrec,
                                   LocalView::Element::dimension>
        )> volumeTerm)
{
  using Element = typename LocalView::Element;
  auto element = localView.element();
  constexpr int dim = Element::dimension;

  // Set of shape functions for a single element
  const auto& localFiniteElement = localView.tree().finiteElement();

  // Set all entries to zero
  localB.resize(localFiniteElement.size());
  localB = 0;

  // A quadrature rule. order is too  high but doesn't harm
  int order = dim*20;
  const auto& quadRule = Dune::QuadratureRules<FltPrec, dim>::rule(element.type(), order);

  // Loop over all quadrature points
  for (const auto& quadPoint : quadRule)
  {
    // Position of the current quadrature point in the reference element
    const Dune::FieldVector<FltPrec,dim>& quadPos = quadPoint.position();

    // The multiplicative factor in the integral transformation formula
    const FltPrec integrationElement = element.geometry().integrationElement(quadPos);

    FltPrec functionValue = volumeTerm(element.geometry().global(quadPos));

    // Evaluate all shape function values at this point
    std::vector<Dune::FieldVector<FltPrec,1> > shapeFunctionValues;
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

//additional function not present in the example by Prof. Sander
//background is the use of OpenMP to speed up assembly
//the loop over the grid elements is parallelised, hence a thread executes
//this function with one simplex from the grid
template<typename Basis, typename Element>
void processElement(const Basis& basis,
					Element& element,
					Dune::BCRSMatrix<typename Basis::GridView::ctype>& matrix,
					Dune::BlockVector<typename Basis::GridView::ctype>& b,
					const std::function<typename Basis::GridView::ctype(Dune::FieldVector<typename Basis::GridView::ctype,
																		Basis::GridView::dimension>
					)> volumeTerm, 
					const std::function<const Dune::FieldMatrix<typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension>(
																			const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
					)> diffusion,
                    const std::function<Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>(
						Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
					)> beta,
					const typename Basis::GridView::ctype mu)
{
	auto localView = basis.localView();
	localView.bind(element);

	Dune::Matrix<typename Basis::GridView::ctype> elementMatrix;
	assembleElementStiffnessMatrix(localView, elementMatrix, diffusion, beta, mu);

	for(size_t p=0; p<elementMatrix.N(); p++)
	{
		// The global index of the p-th degree of freedom of the element
		auto row = localView.index(p);

		for (size_t q=0; q<elementMatrix.M(); q++ )
		{
			// The global index of the q-th degree of freedom of the element
			auto col = localView.index(q);
			//if OpenMP is used, many threads may access the same element
			//due to the simplicity of the operation (update), this is
			//sufficient
			#pragma omp atomic
			matrix[row][col] += elementMatrix[p][q];
		}
	}

	// Now get the local contribution to the right-hand side vector
	Dune::BlockVector<typename Basis::GridView::ctype> localB;
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
					 Dune::BCRSMatrix<typename Basis::GridView::ctype>& matrix,
					 Dune::BlockVector<typename Basis::GridView::ctype>& b,
					 const std::function<const typename Basis::GridView::ctype(const Dune::FieldVector<typename Basis::GridView::ctype,
																			   Basis::GridView::dimension>
					 )> volumeTerm, 
                     const std::function<const Dune::FieldMatrix<typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension>(
						const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
					 )> diffusion,
                     const std::function<const Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>(
																			const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
					 )> beta,
					 const typename Basis::GridView::ctype mu)
{
  auto gridView = basis.gridView();

  // MatrixIndexSets store the occupation pattern of a sparse matrix.
  // They are not particularly efficient, but simple to use.
  Dune::MatrixIndexSet occupationPattern;
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

  std::cerr << "basis.dimension() = " << basis.dimension() << std::endl;

  const auto LastElement = gridView.template end<0>();
	
  //OpenMP requires random access iterators for a parallel for loop,
  //which cannot be provided by dune. therefore use a master-slave
  //approach: the master thread generates a task for each partition
  //element (=simplex), that is then computed by each slave thread.
  //the access to the global stiffness matrix / load vector is handled
  //using atomics, therefore no race conditions and write-updated-element
  //issues can arise.
	#pragma omp parallel if(basis.dimension() > 1e4)
	#pragma omp single
	{
		for(auto element = gridView.template begin<0>(); element != LastElement; element++)
		
			#pragma omp task default(none) firstprivate(element) shared(diffusion,matrix,b,basis,beta,volumeTerm,mu)
				processElement( basis, *element,matrix,b,volumeTerm,diffusion,beta,mu);

		#pragma omp taskwait
	}
}

//calculates the S-Matrix. Because it is a diagonal matrix, we use a 
//vector instead of a matrix datatype (hence the name)
//needs to be adjusted (-> step 2) to distinguish local / global data
template < class Basis >
Dune::BlockVector<typename Basis::GridView::ctype> getSVector(
	const Basis& basis, 
	const std::function<Dune::FieldMatrix<typename Basis::GridView::ctype,Basis::GridView::dimension,Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>)> diffusionTensor,
	const std::function<Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>)> convectiveFlow,
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
	//ONLY works for 2D set \Omega
	
	using FltPrec = typename Basis::GridView::ctype;
	Dune::BlockVector<FltPrec> xVals, yVals; //save x/y coordinate for lagrange node i
	Dune::Functions::interpolate(basis,xVals,[](auto x) { return x[0]; });
	Dune::Functions::interpolate(basis,yVals,[](auto x) { return x[1]; });
	//store infinity norm of the diffusion tensor D and the flow beta over the set omega_i
	//omega_i = all simplices where x_i is a corner of
	Dune::BlockVector<FltPrec> D_omega_i(basis.dimension()), beta_omega_i(basis.dimension());
	
	Dune::BlockVector<FltPrec> result(basis.dimension());
	std::vector<int> noParticipatingElems(basis.dimension());//no=Number; needed for average calculation
	auto localView = basis.localView();
	
	//detect a mesh node (=corner of an element) by comparing local coordinates
	const auto isCorner = [](const FltPrec x, const FltPrec y) -> bool {
		//std::numeric_limits<FltPrec>::epsilon() doesn't work - some corners are then not detected with the approach below
		constexpr FltPrec Limit = 1e-15;
		return	(std::abs(	x) < Limit && std::abs(	 y) < Limit) ||	//bottom left corner
				(std::abs(1-x) < Limit && std::abs(	 y) < Limit) ||	//bottom right corner
				(std::abs(	x) < Limit && std::abs(1-y) < Limit) ||	//top left corner
				(std::abs(1-x) < Limit && std::abs(1-y) < Limit);	//quads only, top right corner
	};
	
	//step 1
	for( const auto& elem : elements(basis.gridView()) ) {
		localView.bind(elem);
		
		const FltPrec infinity_D_omega_i = getApproxMaximumNorm(elem,diffusionTensor);
		const FltPrec infinity_beta_omega_i = getApproxMaximumNorm(elem,convectiveFlow);
		
		const FltPrec diam = diameter(elem.geometry());
		int noCornersDetected = 0;
		for( int i=0; i < localView.size(); i++ ) {
			const int globalID = localView.index(i);
			
			//update infinity norm of D / beta if there is a larger one
			D_omega_i[globalID] = std::max(D_omega_i[globalID],infinity_D_omega_i);
			beta_omega_i[globalID] = std::max(beta_omega_i[globalID],infinity_beta_omega_i);
			
			//if its a mesh node, then add the current element's diameter
			//and update the number of elements this is a corner of
			//needed to calculate \mathfrak h in step 2, which is an
			//average of simplex diameters
			const auto localCoord = elem.geometry().local({xVals[globalID], yVals[globalID]});
			if( isCorner(localCoord[0],localCoord[1]) ) {
				result[globalID] += diam;
				noParticipatingElems[globalID]++;
				
				noCornersDetected++;
			}
		}
		assert(noCornersDetected == elem.geometry().corners());
	}
	
	//step 2
	for( int i=0;i<result.size(); i++) {
		if(noParticipatingElems[i] == 0) continue; //skip Lagrange nodes that are no mesh nodes
		
		const FltPrec h_i = result[i] / noParticipatingElems[i];
		//global data
		//~ result[i] = diffusionInfinityNorm + h_i * betaInfinityNorm + h_i * h_i * mu;
		//local data
		result[i] = D_omega_i[i] + h_i * beta_omega_i[i] + h_i * h_i * mu;
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
