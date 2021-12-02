/**
 * Python bindings to cpp solvers
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "branch_and_bound.h"

namespace py = pybind11;

BnB::Solution bnb(const py::array_t<double>& coords, float max_time, bool depth_first, bool debug)
{
    py::buffer_info buf = coords.request();

    if (buf.ndim != 2) {
	throw std::runtime_error("Number of dimensions must be two");
    }

    if (buf.shape[1] != 2) {
	throw std::runtime_error("Second dimension's size must be 2");
    }

    BnB::Coord *ptr = static_cast<BnB::Coord*>(buf.ptr);
    
    auto solution = BnB::solve({ptr, ptr+buf.size/2}, max_time, depth_first, debug);

    // To allow keyboard interrupts
    PyErr_Clear();
    
    return solution;
}

PYBIND11_MODULE(_solvers, m) {
   m.doc() = "c++ extensions for solvers";
   
   m.def("branch_and_bound", &bnb, "Branch and bound solver",
	 py::arg("coords"), py::arg("max_time"), py::arg("depth_first")=false, py::arg("debug")=false);

   py::class_<BnB::Solution>(m, "bnb_solution")
       .def_readwrite("tour", &BnB::Solution::tour)
       .def_readwrite("distance", &BnB::Solution::distance)
       .def_readwrite("trace", &BnB::Solution::trace);

   py::class_<BnB::TraceItem>(m, "bnb_trace")
       .def_readwrite("distance", &BnB::TraceItem::distance)
       .def_readwrite("time", &BnB::TraceItem::time)
       .def("__repr__",
	    [](const BnB::TraceItem &ti) {
		return "distance: " + std::to_string(ti.distance) + ", time: " + std::to_string(ti.time) + "s";
	    });
}

