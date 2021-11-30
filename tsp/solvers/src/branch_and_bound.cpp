#include <algorithm>
#include <cmath>
#include <climits>
#include <iostream>
#include <queue>
#include <set>
#include <vector>
#include "branch_and_bound.h"

namespace BnB {
	  
uint32_t euc_2d(Coord a, Coord b)
{
    auto square = [](double x) {return x*x;};
    return round(sqrt(square(a.x - b.x) + square(a.y - b.y)));
}

DistanceMatrix::DistanceMatrix(const std::vector<Coord>& coords) :
    m_size(coords.size())
{
    m_dist_mat.reserve(m_size);
    
    for (size_t i = 0; i < coords.size(); ++i) {
	for (size_t j = 0; j < coords.size(); ++j) {
	    uint32_t dist = euc_2d(coords[i], coords[j]);
	    m_dist_mat.push_back(dist);
	}
    }
}

    TspNode::TspNode(Index index, std::vector<bool>&& subproblem, Index level, std::shared_ptr<TspNode> parent, uint32_t distance) :
	index(index), level(level), subproblem(std::move(subproblem)), parent(parent), distance(distance), lower_est(0), cost(UINT_MAX)
{
}

std::vector<std::shared_ptr<TspNode>>
TspNode::expand(DistanceMatrix dist_mat)
{
    std::vector<std::shared_ptr<TspNode>> subnodes;
    uint32_t new_level = level + 1;
    
    for (uint32_t i = 0; i < subproblem.size(); ++i) {
	if (subproblem[i]) {
	    std::vector<bool> new_subproblem = subproblem;
	    new_subproblem[i] = false;
	    uint32_t new_distance = distance + dist_mat.at(index, i);
	    	    
	    std::shared_ptr<TspNode> new_node(new TspNode{i, std::move(new_subproblem), new_level, shared_from_this(), new_distance});
	    	    
	    subnodes.push_back(new_node);
	}
    }
    return subnodes;
}
    
void
TspNode::update_lower_est(DistanceMatrix dist_mat)
{
    lower_est = 0;
    
    // Create vector of remaining nodes
    std::vector<Index> remaining_nodes;
    
    for (uint32_t i = 0; i < subproblem.size(); ++i) {
	if (subproblem[i]) {
	    remaining_nodes.push_back(i);

	}
    }

    // Shortest edges from start and current node
    uint32_t from_start = UINT_MAX;
    uint32_t from_curr = UINT_MAX;
    for (const auto& i : remaining_nodes) {
	from_start = std::min(from_start, dist_mat.at(0, i));
	from_curr = std::min(from_curr, dist_mat.at(index, i));
    }
    if (!remaining_nodes.empty())
	lower_est += from_start + from_curr;

    // Sum of minimum spanning tree of remaining nodes
    std::set<uint32_t> tree;
    
    typedef std::priority_queue<qnode, std::vector<qnode>, mycomparison> tree_queue_type;
    tree_queue_type tree_queue;
    tree_queue.push({0, 0});

    while (!tree_queue.empty()) {
	auto [weight, idx] = tree_queue.top();
	tree_queue.pop();

	if (tree.count(idx) == 0) {
	    tree.insert(idx);
	    lower_est += weight;
	    for (size_t i = 0; i < remaining_nodes.size(); ++i) {
		if (tree.count(i) == 0) {
		    auto neighbor = remaining_nodes[i];
		    tree_queue.push({dist_mat.at(remaining_nodes[idx], neighbor), i});
		}
	    }
	}
    }
}

std::vector<Index>
TspNode::path() const
{
    std::vector<Index> _path;
    _path.push_back(index);
    
    auto node_ptr = parent;
    while (node_ptr != nullptr) {
	_path.push_back(node_ptr->index);
	node_ptr = node_ptr->parent;
    }
    std::reverse(_path.begin(), _path.end());
    return _path;
}
    
inline double cross_product(Coord a, Coord b)
{
    return a.x * b.y - b.x * a.y;
}

inline double direction(Coord a, Coord b, Coord c)
{
    return cross_product(c-a, b-a);
}

bool intersect(Coord a, Coord b, Coord c, Coord d)
{
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
    
Solution solve(const std::vector<Coord>& coords)
{
    DistanceMatrix dist_mat(coords);

    for (const auto& coord : coords) {
	std::cout << "x: " << coord.x << ", y: " << coord.y << '\n';
    }

    // Initialize unvistited nodes
    std::vector<bool> all_candidates(coords.size());
    all_candidates.flip();
    all_candidates[0] = 0; // Except first node

    uint32_t max_level = coords.size() - 1;

    std::priority_queue<std::shared_ptr<TspNode>,
			std::vector<std::shared_ptr<TspNode>>,
			NodeCmp> node_queue;
    node_queue.push(std::shared_ptr<TspNode>(new TspNode(0, std::move(all_candidates))));

    // TODO construct initial solution
    auto node_ptr = node_queue.top();
    for (size_t l = 0; l < max_level; ++l) {
	auto best_subnode = node_ptr;
	uint32_t best_dist = UINT_MAX;
	for (const auto& subnode : node_ptr->expand(dist_mat)) {
	    if (subnode->distance < best_dist) {
		best_subnode = subnode;
		best_dist = subnode->distance;
	    }
	}
	node_ptr = best_subnode;
    }
    std::shared_ptr<TspNode> best_solution = node_ptr;
    best_solution->cost = best_solution->distance + dist_mat.at(best_solution->index, 0);
    std::cout << best_solution->level << '\n';

    std::vector<TraceItem> trace;

    auto no_intersections = [&](const std::shared_ptr<TspNode>& node) {

	if (node->parent == nullptr or node->parent->parent == nullptr) {
	    return true;
	}

	Coord cur_b = coords[node->index];
	Coord cur_a = coords[node->parent->index];

	auto node_ptr = node->parent;
	Coord b = coords[node_ptr->index];
	while (node_ptr->parent != nullptr) {
	    Coord a = coords[node_ptr->parent->index];
	    if (intersect(a, b, cur_a, cur_b)) {
		return false;
	    }
	    
	    node_ptr = node_ptr->parent;
	    b = a;
	}

	return true;
    };

    auto get_time = [=]() -> double {return 0;};

    size_t idx = 0;
    
    while (!node_queue.empty()) {
	std::shared_ptr<TspNode> node = node_queue.top();
	node_queue.pop();

	for (auto& subnode : node->expand(dist_mat)) {
	    if (subnode->level == max_level) {
		subnode->cost = subnode->distance + dist_mat.at(subnode->index, 0);
		if (subnode->cost < best_solution->cost) {
		    best_solution = subnode;
		    trace.push_back({best_solution->cost, get_time()});
		}
	    }
	    else {
		if (!no_intersections(subnode)) {
		    continue;
		}

		subnode->update_lower_est(dist_mat);

		if (subnode->lowerbound() < best_solution->cost) {
		    std::cout << subnode->level << ", " << subnode->lowerbound() << "," << best_solution->cost << ", " << idx++ << '\n';
		    node_queue.push(subnode);
		}
	    }
	}
    }
    std::cout << "Finished\n";
    std::cout << "  " << best_solution->cost << '\n';
    return {best_solution->path(), best_solution->cost, trace};
}

} // namespace
