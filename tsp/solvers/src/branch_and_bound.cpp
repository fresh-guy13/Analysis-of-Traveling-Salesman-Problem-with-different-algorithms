/**
 * Branch and bound solver
 */
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <set>
#include <vector>
#include "branch_and_bound.h"

#include <pybind11/pybind11.h>

namespace BnB {
    
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
TspNode::expand(const DistanceMatrix& dist_mat)
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
TspNode::update_lower_est(const DistanceMatrix& dist_mat)
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
    uint32_t from_start = std::numeric_limits<uint32_t>::max();
    uint32_t from_curr = std::numeric_limits<uint32_t>::max();
    for (const auto& i : remaining_nodes) {
	from_start = std::min(from_start, dist_mat.at(0, i));
	from_curr = std::min(from_curr, dist_mat.at(index, i));
    }
    if (!remaining_nodes.empty())
	lower_est += from_start + from_curr;

    // Sum of minimum spanning tree of remaining nodes
    std::set<uint32_t> tree;  // mst
    
    // Use a priority queue
    using qnode = std::pair<uint32_t, uint32_t>;
    class qnode_cmp
    {
    public:
	bool operator() (const qnode& lhs, const qnode&rhs) const
	    {
		return lhs.first > rhs.first;
	    }
    };
    typedef std::priority_queue<qnode, std::vector<qnode>, qnode_cmp> tree_queue_type;
    
    tree_queue_type tree_queue;
    tree_queue.push({0, 0});

    while (!tree_queue.empty()) {
	auto [weight, idx] = tree_queue.top();
	tree_queue.pop();

	if (tree.count(idx) == 0) {
	    tree.insert(idx);
	    lower_est += weight;  // update mst weight
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
        
Solution solve(const std::vector<Coord>& coords, float max_time, bool depth_first, bool debug)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Helper function for determining if adding node causes intersections with
    // any edge in path
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

    // Helper function for getting current time since start
    auto get_time = [=]() -> double {
        using namespace std::chrono;	

	auto time_now = high_resolution_clock::now();
	duration<double> ts = time_now - start_time;
	return ts.count();
    };

    // Adjacency matrix
    DistanceMatrix dist_mat(coords);

    // Initialize unvistited nodes
    std::vector<bool> all_candidates(coords.size());
    all_candidates.flip();
    all_candidates[0] = 0; // Except first node

    // Level of terminal leaf nodes
    uint32_t max_level = coords.size() - 1;

    // Initialize priority queue holding nodes in branch and bound solver
    std::vector<std::shared_ptr<TspNode>> node_queue;
    node_queue.push_back(std::shared_ptr<TspNode>(new TspNode(0, std::move(all_candidates))));

    // Initialize solution with closest neighbor
    auto node_ptr = node_queue.front();
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
    
    std::vector<TraceItem> trace;
    trace.push_back({best_solution->cost, get_time()});

    // Main solver loop
    while (!node_queue.empty() && (get_time() < max_time)) {

	// Check for ctrl-C on python side
	if (PyErr_CheckSignals() != 0) {
	    if (debug) {
		std::cout << "-- Caught keyboard interrupt --\n";
	    }
	    break;
	}
	
	// Pop off top node
	auto node = node_queue.back();
	if (!depth_first) {
	    node = node_queue.front();
	    std::pop_heap(node_queue.begin(), node_queue.end(), NodeCmp());
	}
	node_queue.pop_back();

	// Expand subproblem
	for (auto& subnode : node->expand(dist_mat)) {

	    // Found a solution (at terminal leaf node)
	    if (subnode->level == max_level) {

		// Total distance includes traveling back to beginning
		subnode->cost = subnode->distance + dist_mat.at(subnode->index, 0);

		// Best solution found?
		if (subnode->cost < best_solution->cost) {
		    best_solution = subnode;
		    trace.push_back({best_solution->cost, get_time()});
		}
	    }
	    else {
		// Optimal solutions contain no intersections
		if (!no_intersections(subnode)) {
		    continue;
		}

		// Update lower estimation of remaining nodes (MST lower bound
		// functions)
		subnode->update_lower_est(dist_mat);

		// Not a dead end?
		if (subnode->lowerbound() < best_solution->cost) {
		    node_queue.push_back(subnode);
		    if (!depth_first)
			std::push_heap(node_queue.begin(), node_queue.end(), NodeCmp());

		    if (debug) {
			std::cout << subnode->level << ", "
				  << subnode->lowerbound() << ", "
				  << best_solution->cost << ", "
				  << get_time() << ", "
				  << node_queue.size() << '\n';
		    }
		}
	    }
	}
    }

    if (debug) {
	std::cout << "Finished\n";
	std::cout << "  " << best_solution->cost << '\n';
    }
    
    return {best_solution->path(), best_solution->cost, trace};
}

} // namespace
