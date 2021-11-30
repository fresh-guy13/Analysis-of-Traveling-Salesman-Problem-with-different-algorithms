#ifndef BNB_H
#define BNB_H

#include <stdint.h>
#include <memory>

namespace BnB {

using Index=uint32_t;

struct TraceItem {
    uint32_t distance;
    double time;
};

struct Solution
{
    std::vector<Index> tour;
    uint32_t distance;
    std::vector<TraceItem> trace;
};

struct Coord
{
    double x;
    double y;

    Coord operator-(Coord& other) {
	Coord res;
	res.x = x - other.x;
	res.y = y - other.y;
	return res;
    }
};

class DistanceMatrix
{
public:
    DistanceMatrix(const std::vector<Coord>& coords);

    uint32_t& at(int i, int j) {
	return m_dist_mat[i*m_size + j];
    }
    
private:
    std::vector<uint32_t> m_dist_mat;
    size_t m_size;
};

struct TspNode
    : std::enable_shared_from_this<TspNode>
{
    TspNode(Index index, std::vector<bool>&& subproblem, Index level=0, std::shared_ptr<TspNode> parent=nullptr, uint32_t distance=0);

    bool operator<(const TspNode& other) const {
	return priority() < other.priority();
    }

    double priority() const {
	return (distance == 0) ? 0 : (double)level / lowerbound();
    }
    
    uint32_t lowerbound() const { return distance + lower_est; }

    std::vector<std::shared_ptr<TspNode>> expand(DistanceMatrix dist_mat);

    void update_lower_est(DistanceMatrix dist_mat);

    std::vector<Index> path() const;
    
    Index index;
    Index level;
    std::vector<bool> subproblem;
    std::shared_ptr<TspNode> parent;
    uint32_t distance;
    uint32_t lower_est;
    uint32_t cost;
};

using qnode = std::pair<uint32_t, uint32_t>;

class mycomparison
{
 public:
  
  mycomparison() = default;
  bool operator() (const qnode& lhs, const qnode&rhs) const
  {
    return lhs.first > rhs.first;
  }
};

class NodeCmp {
public:
    bool operator()(const std::shared_ptr<TspNode>& a,
		    const std::shared_ptr<TspNode>& b)
    {
        return (*a < *b);
    }
};
    
Solution solve(const std::vector<Coord>& coords);

} // namespace
 
#endif
