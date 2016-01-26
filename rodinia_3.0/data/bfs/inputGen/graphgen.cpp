/*
 * graphgen.cpp
 * by Sam Kauffman - Univeristy of Virginia
 *
 * This program generates graphs of the format described in GraphFormat.txt
 * and SampleGraph.jpg for use with BFS (breadth-first search) in Rodinia.
 *
 * The graph is not guaranteed to be connected, are there may be multiple edges
 * and loops.
 *
 * Usage:
 * graphgen <num> [<filename_bit>]
 * num = number of nodes
 * Output filename is "graph<filename_bit>.txt". filename_bit defaults to num.
 *
 * This program uses the TR1 header <random>.
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <climits>

// These names may vary by implementation
//#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential_engine
#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential
//#define UNIFORM_INT_DISTRIBUTION uniform_int_distribution
#define UNIFORM_INT_DISTRIBUTION uniform_int

using namespace std;

#define MIN_NODES 20
#define MAX_NODES ULONG_MAX
#define MIN_EDGES 2
#define MAX_INIT_EDGES 4 // Nodes will have, on average, 2*MAX_INIT_EDGES edges
#define MIN_WEIGHT 1
#define MAX_WEIGHT 10

typedef unsigned int uint;
typedef unsigned long ulong;

struct edge; // forward declaration
typedef vector<edge> node;
struct edge {
	ulong dest;
	uint weight;
};

int main( int argc, char ** argv )
{
	// Parse command lined
	ulong numNodes;
	string s;
	if ( argc < 2 )
	{
		cerr << "Error: enter a number of nodes.\n";
		exit( 1 );
	}
	numNodes = strtoul( argv[1], NULL, 10 );
	if ( numNodes < MIN_NODES || numNodes > MAX_NODES || argv[1][0] == '-' )
	{
		cerr << "Error: Invalid argument: " << argv[1] << "\n";
		exit( 1 );
	}
	s = argc > 2 ? argv[2] : argv[1]; // filename bit
	string filename = "graph" + s + ".txt";

	cout << "Generating graph with " << numNodes << " nodes...\n";
	node * graph;
	graph = new node[numNodes];

	// Initialize random number generators
	// C RNG for numbers of edges and weights
	srand( time( NULL ) );
	// TR1 RNG for choosing edge destinations
	LINEAR_CONGRUENTIAL_ENGINE<ulong, 48271, 0, ULONG_MAX> gen( time( NULL ) );
	UNIFORM_INT_DISTRIBUTION<ulong> randNode( 0, numNodes - 1 );

	// Generate graph
	uint numEdges;
	ulong nodeID;
	uint weight;
	ulong i;
	uint j;
	for ( i = 0; i < numNodes; i++ )
	{
		numEdges = rand() % ( MAX_INIT_EDGES - MIN_EDGES + 1 ) + MIN_EDGES;
		for ( j = 0; j < numEdges; j++ )
		{
			nodeID = randNode( gen );
			weight = rand() % ( MAX_WEIGHT - MIN_WEIGHT + 1 ) + MIN_WEIGHT;
			graph[i].push_back( edge() );
			graph[i].back().dest = nodeID;
			graph[i].back().weight = weight;
			graph[nodeID].push_back( edge() );
			graph[nodeID].back().dest = i;
			graph[nodeID].back().weight = weight;
		}
	}

	// Output
	cout << "Writing to file \"" << filename << "\"...\n";
	ofstream outf( filename );
	outf << numNodes << "\n";
	ulong totalEdges = 0;
	for ( uint i = 0; i < numNodes; i++ )
	{
		numEdges = graph[i].size();
		outf << totalEdges << " " << numEdges << "\n";
		totalEdges += numEdges;
	}
	outf << "\n" << randNode( gen ) << "\n\n";
	outf << totalEdges << "\n";
	for ( ulong i = 0; i < numNodes; i++ )
		for ( uint j = 0; j < graph[i].size(); j++ )
			outf << graph[i][j].dest << " " << graph[i][j].weight << "\n";
	outf.close();

	delete[] graph;
}
