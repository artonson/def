#include "AABB.h"

#include "Predicates_psm.h"

#include <cassert>

namespace AABB
{

namespace
{

double inner_point_box_squared_distance(
	const Vector3 &p,
	const std::array<Vector3, 2> &B)
{
	double result = pow(p[0] - B[0][0], 2);
	result = std::min(result, pow(p[0] - B[1][0], 2));
	for (int c = 1; c < 3; ++c)
	{
		result = std::min(result, pow(p[c] - B[0][c], 2));
		result = std::min(result, pow(p[c] - B[1][c], 2));
	}
	return result;
}

double point_box_signed_squared_distance(
	const Vector3 &p,
	const std::array<Vector3, 2> &B)
{
	bool inside = true;
	double result = 0.0;
	for (int c = 0; c < 3; c++)
	{
		if (p[c] < B[0][c])
		{
			inside = false;
			result += pow(p[c] - B[0][c], 2);
		}
		else if (p[c] > B[1][c])
		{
			inside = false;
			result += pow(p[c] - B[1][c], 2);
		}
	}
	if (inside)
	{
		result = -inner_point_box_squared_distance(p, B);
	}
	return result;
}

double point_box_center_squared_distance(
	const Vector3 &p, const std::array<Vector3, 2> &B)
{
	double result = 0.0;
	for (int c = 0; c < 3; ++c)
	{
		double d = p[c] - 0.5 * (B[0][c] + B[1][c]);
		result += pow(d, 2);
	}
	return result;
	}

	void get_tri_corners(const Vector3 &triangle0, const Vector3 &triangle1, const Vector3 &triangle2, Vector3 &mint, Vector3 &maxt)
	{
		mint[0] = std::min(std::min(triangle0[0], triangle1[0]), triangle2[0]);
		mint[1] = std::min(std::min(triangle0[1], triangle1[1]), triangle2[1]);
		mint[2] = std::min(std::min(triangle0[2], triangle1[2]), triangle2[2]);
		maxt[0] = std::max(std::max(triangle0[0], triangle1[0]), triangle2[0]);
		maxt[1] = std::max(std::max(triangle0[1], triangle1[1]), triangle2[1]);
		maxt[2] = std::max(std::max(triangle0[2], triangle1[2]), triangle2[2]);
	}
	bool box_box_intersection(const Vector3 &min1, const Vector3 &max1, const Vector3 &min2, const Vector3 &max2) //TDOO
	{
		if (max1[0] < min2[0] || max1[1] < min2[1] || max1[2] < min2[2])
			return 0;
		if (max2[0] < min1[0] || max2[1] < min1[1] || max2[2] < min1[2])
			return 0;
		return 1;
	}

	Vector2 to_2d(const Vector3 &p, int t)
	{
		return Vector2(p[(t + 1) % 3], p[(t + 2) % 3]);
	}


	int orient_2d(const Vector2 &p1, const Vector2 &p2, const Vector2 &p3)
	{
// #ifdef ENVELOPE_WITH_GEO
		const int result = -GEO::PCK::orient_2d(p1.data(), p2.data(), p3.data());
// #else
		// const Scalar result = orient2d(p1.data(), p2.data(), p3.data());
// #endif
		if (result > 0)
			return ::AABB::AABB::ORI_POSITIVE;
		else if (result < 0)
			return ::AABB::AABB::ORI_NEGATIVE;
		else
			return ::AABB::AABB::ORI_ZERO;
	}
} // namespace

void AABB::init_envelope_boxes_recursive(
	const std::vector<std::array<Vector3, 2>> &cornerlist,
	int node_index,
	int b, int e)
{
	assert(b != e);
	assert(node_index < boxlist.size());

	if (b + 1 == e)
	{
		boxlist[node_index] = cornerlist[b];
		return;
	}
	int m = b + (e - b) / 2;
	int childl = 2 * node_index;
	int childr = 2 * node_index + 1;

	assert(childl < boxlist.size());
	assert(childr < boxlist.size());

	init_envelope_boxes_recursive(cornerlist, childl, b, m);
	init_envelope_boxes_recursive(cornerlist, childr, m, e);

	assert(childl < boxlist.size());
	assert(childr < boxlist.size());
	for (int c = 0; c < 3; ++c)
	{
		boxlist[node_index][0][c] = std::min(boxlist[childl][0][c], boxlist[childr][0][c]);
		boxlist[node_index][1][c] = std::max(boxlist[childl][1][c], boxlist[childr][1][c]);
	}
}

void AABB::triangle_search_bbd_recursive(
	const Vector3 &triangle0, const Vector3 &triangle1, const Vector3 &triangle2,
	std::vector<unsigned int> &list,
	int n, int b, int e) const
{
	assert(e != b);

	assert(n < boxlist.size());
	bool cut = is_triangle_cut_bounding_box(triangle0, triangle1, triangle2, n);

	if (cut == false)
		return;

	// Leaf case
	if (e == b + 1)
	{
		list.emplace_back(b);
		return;
	}

	int m = b + (e - b) / 2;
	int childl = 2 * n;
	int childr = 2 * n + 1;

	//assert(childl < boxlist.size());
	//assert(childr < boxlist.size());

	// Traverse the "nearest" child first, so that it has more chances
	// to prune the traversal of the other child.
	triangle_search_bbd_recursive(
		triangle0, triangle1, triangle2, list,
		childl, b, m);
	triangle_search_bbd_recursive(
		triangle0, triangle1, triangle2, list,
		childr, m, e);
}

void AABB::point_search_bbd_recursive(
	const Vector3 &point,
	std::vector<unsigned int> &list,
	int n, int b, int e) const
{
	assert(e != b);

	assert(n < boxlist.size());
	bool cut = is_point_cut_bounding_box(point, n);

	if (cut == false)
		return;

	// Leaf case
	if (e == b + 1)
	{
		list.emplace_back(b);
		return;
	}

	int m = b + (e - b) / 2;
	int childl = 2 * n;
	int childr = 2 * n + 1;

	//assert(childl < boxlist.size());
	//assert(childr < boxlist.size());

	// Traverse the "nearest" child first, so that it has more chances
	// to prune the traversal of the other child.
	point_search_bbd_recursive(
		point, list,
		childl, b, m);
	point_search_bbd_recursive(
		point, list,
		childr, m, e);
}

void AABB::segment_search_bbd_recursive(
	const Vector3 &seg0, const Vector3 &seg1,
	std::vector<unsigned int> &list,
	int n, int b, int e) const
{
	assert(e != b);

	assert(n < boxlist.size());
	bool cut = is_segment_cut_bounding_box(seg0, seg1, n);

	if (cut == false)
		return;

	// Leaf case
	if (e == b + 1)
	{
		list.emplace_back(b);
		return;
	}

	int m = b + (e - b) / 2;
	int childl = 2 * n;
	int childr = 2 * n + 1;

	//assert(childl < boxlist.size());
	//assert(childr < boxlist.size());

	// Traverse the "nearest" child first, so that it has more chances
	// to prune the traversal of the other child.
	segment_search_bbd_recursive(
		seg0, seg1, list,
		childl, b, m);
	segment_search_bbd_recursive(
		seg0, seg1, list,
		childr, m, e);
}

void AABB::bbd_searching_recursive(
	const Vector3 &bbd0, const Vector3 &bbd1,
	std::vector<unsigned int> &list,
	int n, int b, int e) const
{
	assert(e != b);

	assert(n < boxlist.size());
	bool cut = is_bbd_cut_bounding_box(bbd0, bbd1, n);

	if (cut == false)
		return;

	// Leaf case
	if (e == b + 1)
	{
		list.emplace_back(b);
		return;
	}

	int m = b + (e - b) / 2;
	int childl = 2 * n;
	int childr = 2 * n + 1;

	//assert(childl < boxlist.size());
	//assert(childr < boxlist.size());

	// Traverse the "nearest" child first, so that it has more chances
	// to prune the traversal of the other child.
	bbd_searching_recursive(
		bbd0, bbd1, list,
		childl, b, m);
	bbd_searching_recursive(
		bbd0, bbd1, list,
		childr, m, e);
}
int AABB::envelope_max_node_index(int node_index, int b, int e)
{
	assert(e > b);
	if (b + 1 == e)
	{
		return node_index;
	}
	int m = b + (e - b) / 2;
	int childl = 2 * node_index;
	int childr = 2 * node_index + 1;
	return std::max(
		envelope_max_node_index(childl, b, m),
		envelope_max_node_index(childr, m, e));
}

void AABB::init(const std::vector<std::array<Vector3, 2>> &cornerlist)
{
	n_corners = cornerlist.size();

	boxlist.resize(
		envelope_max_node_index(
			1, 0, n_corners) +
		1 // <-- this is because size == max_index + 1 !!!
	);

	init_envelope_boxes_recursive(cornerlist, 1, 0, n_corners);
}

bool AABB::is_triangle_cut_bounding_box(
	const Vector3 &tri0, const Vector3 &tri1, const Vector3 &tri2, int index) const
{
	const auto &bmin = boxlist[index][0];
	const auto &bmax = boxlist[index][1];
	Vector3 tmin, tmax;

	get_tri_corners(tri0, tri1, tri2, tmin, tmax);
	bool cut = box_box_intersection(tmin, tmax, bmin, bmax);
	if (cut == false)
		return false;

	if (cut)
	{

		std::array<Vector2, 3> tri;
		std::array<Vector2, 4> mp;
		int o0, o1, o2, o3, ori;
		for (int i = 0; i < 3; i++)
		{
			tri[0] = to_2d(tri0, i);
			tri[1] = to_2d(tri1, i);
			tri[2] = to_2d(tri2, i);

			mp[0] = to_2d(bmin, i);
			mp[1] = to_2d(bmax, i);
			mp[2][0] = mp[0][0];
			mp[2][1] = mp[1][1];
			mp[3][0] = mp[1][0];
			mp[3][1] = mp[0][1];

			for (int j = 0; j < 3; j++)
			{
				o0 = orient_2d(mp[0], tri[j % 3], tri[(j + 1) % 3]);
				o1 = orient_2d(mp[1], tri[j % 3], tri[(j + 1) % 3]);
				o2 = orient_2d(mp[2], tri[j % 3], tri[(j + 1) % 3]);
				o3 = orient_2d(mp[3], tri[j % 3], tri[(j + 1) % 3]);
				ori = orient_2d(tri[(j + 2) % 3], tri[j % 3], tri[(j + 1) % 3]);
				if (ori == 0)
					continue;
				if (ori * o0 <= 0 && ori * o1 <= 0 && ori * o2 <= 0 && ori * o3 <= 0)
					return false;
			}
		}
	}

	return cut;
}

bool AABB::is_point_cut_bounding_box(
	const Vector3 &p, int index) const
{
	const auto &bmin = boxlist[index][0];
	const auto &bmax = boxlist[index][1];
	if (p[0] < bmin[0] || p[1] < bmin[1] || p[2] < bmin[2])
		return false;
	if (p[0] > bmax[0] || p[1] > bmax[1] || p[2] > bmax[2])
		return false;
	return true;
}

bool AABB::is_segment_cut_bounding_box(const Vector3 &seg0, const Vector3 &seg1, int index) const
{
	const auto &bmin = boxlist[index][0];
	const auto &bmax = boxlist[index][1];
	Scalar min[3], max[3];
	min[0] = std::min(seg0[0], seg1[0]);
	min[1] = std::min(seg0[1], seg1[1]);
	min[2] = std::min(seg0[2], seg1[2]);
	max[0] = std::max(seg0[0], seg1[0]);
	max[1] = std::max(seg0[1], seg1[1]);
	max[2] = std::max(seg0[2], seg1[2]);
	if (max[0] < bmin[0] || max[1] < bmin[1] || max[2] < bmin[2])
		return false;
	if (min[0] > bmax[0] || min[1] > bmax[1] || min[2] > bmax[2])
		return false;
	return true;
}
bool AABB::is_bbd_cut_bounding_box(
	const Vector3 &bbd0, const Vector3 &bbd1, int index) const
{
	const auto &bmin = boxlist[index][0];
	const auto &bmax = boxlist[index][1];

	return box_box_intersection(bbd0, bbd1, bmin, bmax);
}

void AABB::get_nearest_facet_hint(
	const Vector3 &p, const std::function<std::pair<double, Vector3>(const Vector3 &, int)> &sq_distance,
	int &nearest_f, Vector3 &nearest_point, double &sq_dist) const
{
	int b = 0;
	int e = n_corners;
	int n = 1;
	while (e != b + 1)
	{
		int m = b + (e - b) / 2;
		int childl = 2 * n;
		int childr = 2 * n + 1;
		if (
			point_box_center_squared_distance(p, boxlist[childl]) <
			point_box_center_squared_distance(p, boxlist[childr]))
		{
			e = m;
			n = childl;
		}
		else
		{
			b = m;
			n = childr;
		}
	}
	nearest_f = b;

	const auto res = sq_distance(p, nearest_f);
	sq_dist = res.first;
	nearest_point = res.second;
}

void AABB::nearest_facet_recursive(
	const Vector3 &p, const std::function<std::pair<double, Vector3>(const Vector3 &, int)> &sq_distance,
	int &nearest_f, Vector3 &nearest_point, double &sq_dist,
	int n, int b, int e) const
{
	assert(e > b);

	// If node is a leaf: compute point-facet distance
	// and replace current if nearer
	if (b + 1 == e)
	{
		const auto res = sq_distance(p, b);
		double cur_sq_dist = res.first;
		Vector3 cur_nearest_point = res.second;
		if (cur_sq_dist < sq_dist)
		{
			nearest_f = b;
			nearest_point = cur_nearest_point;
			sq_dist = cur_sq_dist;
		}
		return;
	}
	int m = b + (e - b) / 2;
	int childl = 2 * n;
	int childr = 2 * n + 1;

	double dl = point_box_signed_squared_distance(p, boxlist[childl]);
	double dr = point_box_signed_squared_distance(p, boxlist[childr]);

	// Traverse the "nearest" child first, so that it has more chances
	// to prune the traversal of the other child.
	if (dl > dr)
	{
		if (dl < sq_dist)
		{
			nearest_facet_recursive(
				p, sq_distance,
				nearest_f, nearest_point, sq_dist,
				childl, b, m);
		}
		if (dr < sq_dist)
		{
			nearest_facet_recursive(
				p, sq_distance,
				nearest_f, nearest_point, sq_dist,
				childr, m, e);
		}
	}
	else
	{
		if (dr < sq_dist)
		{
			nearest_facet_recursive(
				p, sq_distance,
				nearest_f, nearest_point, sq_dist,
				childr, m, e);
		}
		if (dl < sq_dist)
		{
			nearest_facet_recursive(
				p, sq_distance,
				nearest_f, nearest_point, sq_dist,
				childl, b, m);
		}
	}
}

} // namespace AABB
