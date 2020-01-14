
#include "AABB.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;


PYBIND11_MODULE(pyaabb, m)
{
	auto &aabb = py::class_<AABB::AABB>(m, "AABB")
	.def(py::init<>())
	.def("build", [](AABB::AABB &aabb, const std::vector<std::array<AABB::Vector3, 2>> &cornerlist) {
		aabb.init(cornerlist);
	})
	.def("point_find_bbox", [](const AABB::AABB &aabb, const AABB::Vector3 &p) {
		std::vector<unsigned int> list;
		aabb.point_find_bbox(p, list);
		return list;
	})
	.def("segment_find_bbox", [](const AABB::AABB &aabb, const AABB::Vector3 &seg0, const AABB::Vector3 &seg1) {
		std::vector<unsigned int> list;
		aabb.segment_find_bbox(seg0, seg1, list);
		return list;
	})
	.def("bbox_find_bbox", [](const AABB::AABB &aabb, const AABB::Vector3 &bbd0, const AABB::Vector3 &bbd1) {
		std::vector<unsigned int> list;
		aabb.bbox_find_bbox(bbd0, bbd1, list);
		return list;
	})
	.def("nearest_point", [](const AABB::AABB &aabb, const AABB::Vector3 &p, const std::function<std::pair<double, AABB::Vector3>(const AABB::Vector3 &, int)> &sq_distance) {
		int nearest_facet;
		AABB::Vector3 nearest_point;
		double sq_dist;
		nearest_facet = aabb.nearest_facet(p, sq_distance, nearest_point, sq_dist);
		return std::make_tuple(nearest_facet, nearest_point, sq_dist);
	});
	aabb.doc() = "AABB";
}