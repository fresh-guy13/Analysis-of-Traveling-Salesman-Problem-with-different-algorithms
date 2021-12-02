/** 
 * Interface to branch and bound solver
 */
#ifndef BNB_H
#define BNB_H

#include <cmath>
#include <limits>
#include <stdint.h>
#include <memory>

namespace BnB {

/**
 * Index of vertex
 */
using Index=uint32_t;

/**
 * Trace item used for logging solution
 */ 
struct TraceItem {
    uint32_t distance;  // distance of tour
    double time;        // Time solution found
};

/**
 * Solution struct returned by solver
 */ 
struct Solution
{
    std::vector<Index> tour;      // Best TSP tour found
    uint32_t distance;            // Best distance found
    std::vector<TraceItem> trace; // Trace of solutions found
};

/**
 * Coordinate type
 */
struct Coord
{
    double x;
    double y;

    Coord operator-(const Coord& other) const {
	Coord res;
	res.x = x - other.x;
	res.y = y - other.y;
	return res;
    }
};

/**
 * Calculates euclidean distance between two coordinates
 */
inline uint32_t euc_2d(Coord a, Coord b)
{
    auto square = [](double x) {return x*x;};
    return round(sqrt(square(a.x - b.x) + square(a.y - b.y)));
}
    
/**
 * Adjacency matrix for TSP graph
 */
class DistanceMatrix
{
public:
    DistanceMatrix(const std::vector<Coord>& coords);

    const uint32_t& at(int i, int j) const {
	return m_dist_mat[i*m_size + j];
    }
    
private:
    std::vector<uint32_t> m_dist_mat;
    size_t m_size; // dimension of square matrix
};

    

/**
 * Determines if edges (a,b) and (c,d) intersect, allowing for colinearity
 *
 * Reference: CLRS 33.1 on line-segments
 */
inline bool intersect(const Coord& a, const Coord& b, const Coord& c, const Coord& d)
{
    auto cross_product = [](Coord&& a, Coord&& b) -> double
    {
        return a.x * b.y - b.x * a.y;
    };

    auto direction = [&](const Coord& a, const Coord& b, const Coord& c) -> double
    {
        return cross_product(c-a, b-a);
    };
		
    double d1 = direction(c, d, a);
    double d2 = direction(c, d, b);
    double d3 = direction(a, b, c);
    double d4 = direction(a, b, d);
    if (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
	and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))
	return true;
    else
	return false;
}

/**
 * Node representing a particular state in the branch and bound solver
 */
struct TspNode
    : std::enable_shared_from_this<TspNode>
{
    /**
     * ctor
     */
    TspNode(Index index, std::vector<bool>&& subproblem, Index level=0, std::shared_ptr<TspNode> parent=nullptr, uint32_t distance=0);

    /**
     * Less than operator for use with priority queue
     */
    bool operator<(const TspNode& other) const {
	return priority() < other.priority();
    }

    /**
     * Priority of node (greater is better).
     * Give higher weight for leaf nodes.
     */
    double priority() const {
	double num = level;
	return (distance == 0) ? 0 : num / lowerbound();
    }

    /**
     * Lowerbound of TSP distance
     */
    uint32_t lowerbound() const { return distance + lower_est; }

    /**
     * Return a node for each subproblem
     */
    std::vector<std::shared_ptr<TspNode>> expand(const DistanceMatrix& dist_mat);

    /**
     * Updates estimate on cost of visiting remaining vertices
     */
    void update_lower_est(const DistanceMatrix& dist_mat);

    /**
     * Returns tour up until node
     */
    std::vector<Index> path() const;
    
    Index index;
    Index level;
    std::vector<bool> subproblem;
    std::shared_ptr<TspNode> parent;
    uint32_t distance;
    uint32_t lower_est;
    uint32_t cost;
};

/**
 * Custom comparitor for shared pointers holding a TspNode
 */
class NodeCmp {
public:
    bool operator()(const std::shared_ptr<TspNode>& a,
		    const std::shared_ptr<TspNode>& b)
    {
        return (*a < *b);
    }
};

/**
 * Branch and bound solver for TSP problem on hamiltonian graph
 *
 * @param coords Coordinates of each vertex in graph
 * @param max_time Maximum time before solver stops
 * @param depth_first Whether or not to use depth first search (default is best-first)
 * @param debug Whether or not to print debug statements
 * @return Branch and bound solution type
 */
Solution solve(const std::vector<Coord>& coords,
	       float max_time=std::numeric_limits<float>::infinity(), bool depth_first=false,
	       bool debug=false);

} // namespace
 
#endif
