#pragma once

#include <functional>
#include <type_traits>
#include <cmath>

#include <dune/istl/bvector.hh>
#include <dune/istl/matrix.hh>

#include <dune/geometry/quadraturerules.hh>

#include <eigen3/Eigen/Dense>

#include "util.hpp"
#include "stabilisation_term.hpp"

//Contains norm calculations: L_2 norm, energy norm, stabilised norm, S norm

//calculate the ||.||_a norm eq. (2.4). as proved there, the convective
//term doesn't contribute due to dirichlet boundary conditions
//||v||_a^2 = (D \nabla v, \nabla v)_\Omega + (mu v,v)_\Omega
//and (.,.)_\Omega = L^2(\Omega) inner product 
//f / Df are - if available - the correct solution function and its derivative
template < typename LocalView, typename GridView, typename VectorImpl > requires IsEnumeratable<VectorImpl>
typename GridView::ctype ANorm( const GridView& gridView, LocalView localView,
	const VectorImpl& u,
	 const std::function<Dune::FieldMatrix<typename GridView::ctype, GridView::dimension, GridView::dimension>(
						const Dune::FieldVector<typename GridView::ctype, GridView::dimension>
					)
				> diffusion,
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
			
			const FltPrec integrationElement = elem.geometry().integrationElement(quadPos);
			
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

//calculates the ||.||_h norm eq. (2.23)
template < class Basis, typename VectorImpl > requires IsEnumeratable<VectorImpl>
typename Basis::GridView::ctype cipNorm(	const Basis& basis,
								const VectorImpl& u,
								const std::function<
												typename Dune::FieldMatrix<typename Basis::GridView::ctype, Basis::GridView::dimension, Basis::GridView::dimension>(
													typename Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>
												)
											> diffusion,
								const std::function<Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype, Basis::GridView::dimension>)> beta,
								const typename Basis::GridView::ctype mu,
								const typename Basis::GridView::ctype gamma,
								const std::function<typename Basis::GridView::ctype(Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>)> f,
								const std::function<Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>(const Dune::FieldVector<typename Basis::GridView::ctype,Basis::GridView::dimension>)> Df)
{
	//WORKS ONLY for functions of at least C^1!
	//i.e. [\grad u] = 0
	using FltPrec = typename Basis::GridView::ctype;
	
	//reuse the A-norm (2.4) which lacks the J(.,.)-term
	FltPrec result = std::pow(ANorm(basis.gridView(),basis.localView(),u,diffusion,mu,f,Df),2);
	
	//add the J(.,.) on top
	const auto action = [&u,&result](const int globalI,const int globalJ, const FltPrec contrib) {
		result += u[globalI] * u[globalJ] * contrib;
	};
	addCIPImpl(basis,action,beta,gamma);

	//there were issues with small negative values, hence solve the issue
	if( result < 0 and std::abs(result) < std::numeric_limits<float>::epsilon() ) {
		result = std::abs(result);
	}
	
	return std::sqrt(result);
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

//L_2 norm of a function. f is the correct solution function if available
template < typename VectorImpl, typename LocalView, typename GridView > requires IsEnumeratable<VectorImpl>
typename GridView::ctype L2Norm( GridView gridView, LocalView localView, const VectorImpl u,
								const std::function<typename GridView::ctype(Dune::FieldVector<typename GridView::ctype,LocalView::Element::dimension>)> f = zeroFunction
) {
	constexpr int dim = LocalView::Element::dimension;
	using FltPrec = typename GridView::ctype;
	
	FltPrec l2err = 0;
	
	for( const auto& elem : elements(gridView) ) {
		localView.bind(elem);
		
		const auto& localFiniteElement = localView.tree().finiteElement();
		const auto& order = 3*localFiniteElement.localBasis().order();
		const auto& quadRule = Dune::QuadratureRules<FltPrec, dim>::rule(elem.type(), order);
		
		for( const auto& quadPoint : quadRule ) {
			const Dune::FieldVector<FltPrec,dim>& quadPos = quadPoint.position();
			
			const FltPrec integrationElement = elem.geometry().integrationElement(quadPos);
			std::vector<Dune::FieldVector<FltPrec,1> > shapeFunctionValues;
			localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues );
			
			const FltPrec functionValue = f(elem.geometry().global(quadPos));
			
			FltPrec localU = 0;
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

//calculates the s-norm eq. (2.30) with \alpha=1

//Eigen uses expression templates and therefore the appropriate call uses this version; const Eigen::Vector<FltPrec,Eigen::Dynamic> u doesn't work for this reason
//other possible solution: call .eval() on the corresponding vector before calling this function and use Eigen::Vector as type instead.
//due to more specialisation, dune-vectors use the function overload below
template < typename VectorImpl, typename FltPrec > requires IsEnumeratable<VectorImpl>
FltPrec sNorm(
	const VectorImpl& u,
	const Dune::BlockVector<FltPrec>& sVector,
	const FltPrec diffusionInfinityNorm,
	const FltPrec betaInfinityNorm,
	const FltPrec mu,
	const FltPrec H)
{
	FltPrec result{0};
	
	for( int i=0; i < u.size(); i++ ) {
		result += sVector[i] * u[i] * u[i];
	}
	
	return std::sqrt(result);
	//for global data
	//~ return std::sqrt((diffusionInfinityNorm + betaInfinityNorm * H + mu * H * H) * u.dot(u));
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
