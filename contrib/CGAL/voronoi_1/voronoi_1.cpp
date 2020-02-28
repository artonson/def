#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/vcm_estimate_edges.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_off_points.h>
#include <CGAL/IO/read_xyz_points.h>
#include <utility> // defines std::pair
#include <vector>
#include <string>
#include <fstream>
#include <boost/foreach.hpp>
// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
// Point with normal vector stored in a std::pair.
typedef std::pair<Point, Vector> PointVectorPair;
typedef std::vector<PointVectorPair> PointList;
typedef CGAL::cpp11::array<double,6> Covariance;
int main (int argc, char** argv) {
    // Reads a .xyz point set file in points[].
    std::list<PointVectorPair> points;
    double threshold = 0.16;
    double R = 0.2,
           r = 0.1;
    std::string filename, name, path_to_file, output_directory;
    for (int i = 1; i < argc; ++i){
        std::string arg = argv[i];
        if ((arg == "-f") || (arg == "-filename")){
            filename = argv[i+1];
            int idx1 = filename.rfind("/");
            int idx2 = filename.find(".");
            path_to_file = filename.substr(0, idx1);
            name = filename.substr(idx1+1, idx2-idx1-1);
        }
        if (arg == "-R"){
            R = std::stod(argv[i+1]);
        }
        if (arg == "-r"){
            r = std::stod(argv[i+1]);
        }
        if ((arg == "-t") || (argv[i] == "-threshold")){
            threshold = std::stod(argv[i+1]);
        }
        if ((arg == "-o") || (arg == "-output")){
            output_directory = argv[i+1];
        }
        
    }
    std::ifstream stream(filename);
    
    if (!stream ||
        !CGAL::read_xyz_points(stream,
                               std::back_inserter(points),
                               CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointVectorPair>())))
    {
        std::cerr << "Error: cannot read file " + filename << std::endl;
        return EXIT_FAILURE;
    }
    // Estimates covariance matrices per points.
    std::vector<Covariance> cov;
    CGAL::First_of_pair_property_map<PointVectorPair> point_map;
    CGAL::compute_vcm(points, cov, R, r,
                      CGAL::parameters::point_map (point_map).geom_traits (Kernel()));
    // Find the points on the edges.
    // Note that this step is not expensive and can be done several time to get better results
    std::ofstream output(output_directory+"/points_on_edges" + name + ".xyz");
    std::ofstream classification(output_directory+"/points_classification_" + name + ".txt");
    int i = 0;
    std::cout << "Saving classification to " << output_directory + "/points_classification_" + name + ".txt" << "\n";
    BOOST_FOREACH(const PointVectorPair& p, points)
    {
      
      if (CGAL::vcm_is_on_feature_edge(cov[i], threshold)) {
          output << p.first << "\n";
          classification << "1 "; }
      else {
          classification << "0 "; }
      ++i;
    }
    output.close();
    classification.close();
    return 0;
}
