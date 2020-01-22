#pragma once

#include <Eigen/Core>

#include <vector>
#include <array>

namespace AABB
{
	typedef double Scalar;
	typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
	typedef Eigen::Matrix<Scalar, 2, 1> Vector2;

	class AABB
	{
	public:
		static const int ORI_POSITIVE = 1;
		static const int ORI_ZERO = 0;
		static const int ORI_NEGATIVE = -1;

	private:
		std::vector<std::array<Vector3, 2>> boxlist;
		size_t n_corners = -1;

		void init_envelope_boxes_recursive(
			const std::vector<std::array<Vector3, 2>> &cornerlist,
			int node_index,
			int b, int e);

		void triangle_search_bbd_recursive(
			const Vector3 &triangle0, const Vector3 &triangle1, const Vector3 &triangle2,
			std::vector<unsigned int> &list,
			int n, int b, int e) const;

		void point_search_bbd_recursive(
			const Vector3 &point,
			std::vector<unsigned int> &list,
			int n, int b, int e) const;

		void segment_search_bbd_recursive(
			const Vector3 &seg0, const Vector3 &seg1,
			std::vector<unsigned int> &list,
			int n, int b, int e) const;

		void bbd_searching_recursive(
			const Vector3 &bbd0, const Vector3 &bbd1,
			std::vector<unsigned int> &list,
			int n, int b, int e) const;

		void nearest_facet_recursive(
			const Vector3 &p, const std::function<std::pair<double, Vector3>(const Vector3 &, int)> &sq_distance,
			int &nearest_facet, Vector3 &nearest_point, double &sq_dist,
			int n, int b, int e) const;

		static int envelope_max_node_index(int node_index, int b, int e);

		bool is_triangle_cut_bounding_box(const Vector3 &tri0, const Vector3 &tri1, const Vector3 &tri2, int index) const;
		bool is_point_cut_bounding_box(const Vector3 &p, int index) const;
		bool is_segment_cut_bounding_box(const Vector3 &seg0, const Vector3 &seg1, int index) const;
		bool is_bbd_cut_bounding_box(const Vector3 &bbd0, const Vector3 &bbd1, int index) const;

	public:
		void init(const std::vector<std::array<Vector3, 2>> &cornerlist);


		inline void point_find_bbox(
			const Vector3 &p,
			std::vector<unsigned int> &list) const
		{
			point_search_bbd_recursive(
				p, list, 1, 0, n_corners);
		}
		inline void segment_find_bbox(
			const Vector3 &seg0, const Vector3 &seg1,
			std::vector<unsigned int> &list) const
		{
			segment_search_bbd_recursive(
				seg0, seg1, list, 1, 0, n_corners);
		}
		inline void bbox_find_bbox(
			const Vector3 &bbd0, const Vector3 &bbd1,
			std::vector<unsigned int> &list) const
		{
			list.clear();
			assert(n_corners >= 0);
			bbd_searching_recursive(bbd0, bbd1, list, 1, 0, n_corners);
		}

		void get_nearest_facet_hint(
			const Vector3 &p, const std::function<std::pair<double, Vector3>(const Vector3 &, int)> &sq_distance,
			int &nearest_facet, Vector3 &nearest_point, double &sq_dist) const;

		inline int nearest_facet(
			const Vector3 &p, const std::function<std::pair<double, Vector3>(const Vector3 &, int)> &sq_distance,
			Vector3 &nearest_point, double &sq_dist) const
		{
			assert(n_corners >= 0);

			int nearest_facet;
			get_nearest_facet_hint(p, sq_distance, nearest_facet, nearest_point, sq_dist);
			nearest_facet_recursive(
				p, sq_distance,
				nearest_facet, nearest_point, sq_dist,
				1, 0, n_corners);
			return nearest_facet;
		}
};
} // namespace fastEnvelope