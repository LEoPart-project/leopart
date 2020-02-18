// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef PARTICLE_H
#define PARTICLE_H

#endif // PARTICLE_H

#include <dolfin/geometry/Point.h>
#include <vector>

// Define the particle atom as a vector of dolfin points
namespace dolfin
{
typedef std::vector<Point> particle;
}
