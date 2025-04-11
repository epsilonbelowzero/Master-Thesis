#pragma once

#include <functional>
#include <vector>

#include <dune/geometry/quadraturerules.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include "util.hpp"

//Contains only the following function

//Implementation of the calculation of the J(.,.) term eq. (2.22)
//J(v_h,w_h) = \gamma sum_{inner facets F} \int_F ||\beta||_{L^\infty(F)} h_F^2 [[ \nabla v_h ]] [[ \nabla w_h ]] ds
//[[ \nabla v_h ]] is the jump of the gradient of v_h over the facet F, h_F is the diameter of the facet
template < class Basis >
void addCIPImpl(const Basis& basis,
				const std::function<void(const int,const int,const typename Basis::GridView::ctype)> action,
				const std::function<Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>)> beta,
				const typename Basis::GridView::ctype gamma)
{
	using FltPrec = typename Basis::GridView::ctype;
	constexpr int DomainDim = Basis::GridView::dimension;
	
	//integral term in (2.22)
	//the remainder is needed for the arguments: get the faces F, and
	//calculate the jumps of the gradients
	const auto evaluateIntegrand = [gamma,beta](const FltPrec h_F, const Dune::FieldVector<FltPrec,DomainDim> diffIn, const Dune::FieldVector<FltPrec,DomainDim> diffOut, const auto intersection) -> FltPrec {
		const FltPrec betaInfNormF = getApproxMaximumNorm(intersection,beta);
		return gamma *betaInfNormF * (h_F*h_F) * (diffIn*diffOut);
	};
	
	//loop over all intersections of two elements = faces F
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
			const auto& quadRule = Dune::QuadratureRules<FltPrec,intersectionDim>::rule(intersection.type(),quadOrder);
			const auto& localFiniteElement = localView.tree().finiteElement();
			
			auto const geometryIn = intersection.inside().geometry();
			auto const geometryIntersection = intersection.geometry();
			auto const geometryOut = intersection.outside().geometry();
			
			for( const auto& quadPoint : quadRule ) {
				const auto quadPos = quadPoint.position();
				const auto quadPosGlobal = geometryIntersection.global(quadPos);

				// The transposed inverse Jacobian of the map from the reference element
				// to the grid element
				const auto jacobian = geometryIntersection.jacobianInverseTransposed(quadPos);
				const auto jacobianIn = geometryIn.jacobianInverseTransposed(geometryIn.local(quadPosGlobal));
				const auto jacobianOut = geometryOut.jacobianInverseTransposed(geometryOut.local(quadPosGlobal));

				// The determinant term in the integral transformation formula
				const auto integrationElement = geometryIntersection.integrationElement(quadPos);

				//get local gradients
				std::vector<Dune::FieldMatrix<FltPrec,1,DomainDim>> referenceGradientsIn,referenceGradientsOut;
				localView.tree().finiteElement().localBasis().evaluateJacobian(geometryIn.local(quadPosGlobal), referenceGradientsIn);
				localViewOut.tree().finiteElement().localBasis().evaluateJacobian(geometryOut.local(quadPosGlobal), referenceGradientsOut);

				//get global gradients
				std::vector<Dune::FieldVector<FltPrec,DomainDim> > 	gradientsIn(referenceGradientsIn.size()), gradientsOut(referenceGradientsIn.size());
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
						Dune::FieldVector<FltPrec,DomainDim> diffIn, diffOut;
						diffIn = gradientsIn[i];
						if(localJIndexOfI != -1) {
							diffIn -= gradientsOut[localJIndexOfI];
						}
						diffOut = -gradientsOut[j];
						if(localIIndexOfJ != -1) {
							diffOut += gradientsIn[localIIndexOfJ];
						}
						
						auto contribution = evaluateIntegrand(h_F,diffIn,diffOut,intersection) * integrationElement * quadPoint.weight();
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
				
				//contributions for dofs not on the intersection within intersection.inside(),
				//i.e., no support in intersection.outside().
				//when inside and outside are interchanged, the corresponding contributions for intersection.outside()
				//are calculated, therefore those terms are missing
				for( int i=0; i<extraContribution.size(); i++ ) {
					for( int j=0; j<extraContribution.size(); j++ ) {
						const auto localI1 = extraContribution[i];
						const auto localI2 = extraContribution[j];
						const auto globalI1 = localView.index(localI1);
						const auto globalI2 = localView.index(localI2);
						
						const auto diffIn = gradientsIn[localI1];
						const auto diffOut = gradientsIn[localI2];
						
						action(globalI1,globalI2, evaluateIntegrand(h_F,diffIn,diffOut,intersection) * integrationElement * quadPoint.weight() );
					}
				}
			}
		}
	}
}
