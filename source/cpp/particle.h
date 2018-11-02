// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#ifndef PARTICLE_H
#define PARTICLE_H

#endif // PARTICLE_H

#include <vector>
#include <dolfin/geometry/Point.h>


// Define the particle atom as a vector of dolfin points
namespace dolfin{
      typedef std::vector<Point> particle;
}
