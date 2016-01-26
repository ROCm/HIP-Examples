/*********************

Hotspot Expand
by Sam Kauffman - Univeristy of Virginia
Generate larger input files for Hotspot by expanding smaller versions

*/


#include "64_128.h"
//#include "64_256.h"
//#include "1024_2048.h"
//#include "1024_4096.h"
//#include "1024_8192.h"
//#include "1024_16384.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#define OUT_SIZE IN_SIZE*MULTIPLIER

using namespace std;

void expand( char * infName, char * outfName )
{
	const int x = MULTIPLIER;
	double val;
	fstream fs;
	double ** outMatr;

	// allocate 2d array of doubles
	outMatr = (double **) malloc( OUT_SIZE * sizeof( double * ) );
	for ( int i = 0; i < OUT_SIZE; i++ )
		outMatr[i] = (double *) malloc(OUT_SIZE * sizeof( double ) );

	// copy values into larger array
	fs.open( infName, ios::in );
	if ( !fs )
		cerr << "Failed to open input file.\n";
	for ( int row = 0; row < IN_SIZE; row++ )
		for ( int col = 0; col < IN_SIZE; col++ )
		{
			fs >> val;
			for ( int rowOff = 0; rowOff < x; rowOff++ )
				for ( int colOff = 0; colOff < x; colOff++ )
					outMatr[x * row + rowOff][x * col + colOff] = val;
		}
	fs.close();

	fs.open( outfName, ios::out );
	if ( !fs )
		cerr << "Failed to open output file.\n";
	fs.precision( 6 );
	fs.setf( ios::fixed );
	for ( int row = 0; row < OUT_SIZE; row++ )
		for ( int col = 0; col < OUT_SIZE; col++ )
			fs << outMatr[row][col] << "\n";
	fs.close();

	for ( int i = 0; i < OUT_SIZE; i++ )
		free( outMatr[i] );
	free( outMatr );
}

int main( int argc, char* argv[] )
{
	expand( TEMP_IN, TEMP_OUT );
	expand( POWER_IN, POWER_OUT );

	cout << "Data written to files " << TEMP_OUT << " and " << POWER_OUT << ".\n";
}
