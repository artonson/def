#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/pca_estimate_normals.h>
#include <vector>
#include <fstream>

#include "edge_aware_upsample_point_set.h"

// types
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
// Point with normal vector stored in a std::pair.
typedef std::pair<Point, Vector> PointVectorPair;
// Concurrency
#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif
int main(int argc, char* argv[])
{
  // Reads a .xyz point set file in points[], *with normals*.
  std::vector<PointVectorPair> points;
  const char* output_directory;
  const char* filename;
  for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if ((arg == "-f") || (arg == "-filename")){
            filename = argv[i+1];
        }    
        if ((arg == "-o") || (arg == "-output")){
            output_directory = argv[i+1];
        }
  }
  //const char* input_filename = (argc>1)?argv[1]:filename"CGAL-4.14/examples/Point_set_processing_3/data/before_upsample.xyz";
  const char* output_filename = (argc>2)?argv[2]:"before_upsample_UPSAMPLED.xyz";
  std::ifstream stream(filename);
 
  if (!stream ||
      !CGAL::read_xyz_points(stream,
                        std::back_inserter(points),
                        CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
                        normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())))
  {
    std::cerr << "Error: cannot read file " << filename << std::endl;
    return EXIT_FAILURE;
  }
  unsigned int k=6;
  CGAL::pca_estimate_normals<Concurrency_tag>(points, k, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
                        normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()));
  //Algorithm parameters
  const double sharpness_angle = 25;   // control sharpness of the result.
  const double edge_sensitivity = 0;    // higher values will sample more points near the edges          
  const double neighbor_radius = 0.25;  // initial size of neighborhood.
  const std::size_t number_of_output_points = points.size() * 4;
  //Run algorithm 
  CGAL::edge_aware_upsample_point_set<Concurrency_tag>(
    points,
    std::back_inserter(points),
    CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
    normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>()).
    sharpness_angle(sharpness_angle).
    edge_sensitivity(edge_sensitivity).
    neighbor_radius(neighbor_radius).
    number_of_output_points(number_of_output_points));
  // Saves point set.
  std::ofstream out(output_filename);  
  out.precision(17);
  if (!out ||
     !CGAL::write_xyz_points(
      out, points,
      CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>()).
      normal_map(CGAL::Second_of_pair_property_map<PointVectorPair>())))
  {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
