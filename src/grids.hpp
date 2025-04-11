#pragma once

#include <config.h>

#include <memory>

#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/albertagrid.hh>

#include "util.hpp"

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
		auto grid = Dune::StructuredGridFactory<GridType>::createSimplexGrid( {lowerLeftCorner[0],lowerLeftCorner[1]}, {upperRightCorner[0],upperRightCorner[1]}, {1,1} );
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
		auto grid = Dune::StructuredGridFactory<GridType >::createSimplexGrid( {lowerLeftCorner[0],lowerLeftCorner[1]}, {upperRightCorner[0],upperRightCorner[1]}, {edgeNumber,edgeNumber} );
		
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
		Dune::GridFactory<GridType> factory;
		factory.insertVertex({lowerLeftCorner[0]													,lowerLeftCorner[1]														});
		factory.insertVertex({0.5*(lowerLeftCorner[0]+upperRightCorner[0]),lowerLeftCorner[1]														});
		factory.insertVertex({upperRightCorner[0]													,lowerLeftCorner[1]														});
		factory.insertVertex({lowerLeftCorner[0]													, 0.5*(lowerLeftCorner[1]+upperRightCorner[1])});
		factory.insertVertex({0.5*(lowerLeftCorner[0]+upperRightCorner[0]), 0.5*(lowerLeftCorner[1]+upperRightCorner[1])});
		factory.insertVertex({upperRightCorner[0]													, 0.5*(lowerLeftCorner[1]+upperRightCorner[1])});
		factory.insertVertex({lowerLeftCorner[0]													, upperRightCorner[1]													});
		factory.insertVertex({0.5*(lowerLeftCorner[0]+upperRightCorner[0]), upperRightCorner[1]													});
		factory.insertVertex({upperRightCorner[0]													, upperRightCorner[1]													});
		
		factory.insertElement(Dune::GeometryTypes::simplex(2), {3,0,4});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {0,1,4});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {1,2,4});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {2,5,4});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {4,7,6});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {3,4,6});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {4,5,8});
		factory.insertElement(Dune::GeometryTypes::simplex(2), {4,8,7});
		
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
		
		Dune::GridFactory<GridType> factory;
		
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
				factory.insertElement(Dune::GeometryTypes::triangle, {			0					+Advance,			 1				+Advance, PointsPerRow		+Advance});
				factory.insertElement(Dune::GeometryTypes::triangle, {PointsPerRow+1	+Advance, PointsPerRow	+Advance,			 1					+Advance});
				factory.insertElement(Dune::GeometryTypes::triangle, {PointsPerRow+1	+Advance,			 1				+Advance,PointsPerRow+2		+Advance});
				factory.insertElement(Dune::GeometryTypes::triangle, {			2					+Advance,PointsPerRow+2	+Advance,			 1					+Advance});
				
				factory.insertElement(Dune::GeometryTypes::triangle, {  2*PointsPerRow	+Advance,  PointsPerRow		+Advance,2*PointsPerRow+1	+Advance});
				factory.insertElement(Dune::GeometryTypes::triangle, {   PointsPerRow		+Advance, PointsPerRow+1	+Advance,2*PointsPerRow+1	+Advance});
				factory.insertElement(Dune::GeometryTypes::triangle, {  PointsPerRow+1	+Advance, PointsPerRow+2	+Advance,2*PointsPerRow+1	+Advance});
				factory.insertElement(Dune::GeometryTypes::triangle, {2*PointsPerRow+2	+Advance,2*PointsPerRow+1	+Advance, PointsPerRow+2	+Advance});
			}
		}
		
		std::cerr << "\t" <<  "Done inserting." << std::endl;
		
		return factory.createGrid();
	}
};
