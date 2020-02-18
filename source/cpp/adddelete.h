// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com
// Copyright: (c) 2018
// License: GNU Lesser GPL version 3 or any later version
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifndef ADDDELETE_H
#define ADDDELETE_H

#include <memory>
#include <vector>

namespace dolfin
{
// Forward declarations
class Function;
class Point;
class Cell;

class particles;

class AddDelete
{
public:
  AddDelete(particles& P, std::size_t np_min, std::size_t np_max,
            std::vector<std::shared_ptr<const Function>> FList);
  AddDelete(particles& P, std::size_t np_min, std::size_t np_max,
            std::vector<std::shared_ptr<const Function>> FList,
            std::vector<std::size_t> pbound, std::vector<double> bounds);
  ~AddDelete();

  // Sweep to be done before advection
  void do_sweep();
  // Failsafe sweep (after advection) to make sure that cell
  // contains minimum number
  void do_sweep_failsafe(const std::size_t np_min);

  // To be deprecated?
  void do_sweep_weighted();

private:
  // Private methods
  void insert_particles(const std::size_t Np_def, const Cell& dolfin_cell);
  void insert_particles_weighted(const std::size_t Np_def,
                                 const Cell& dolfin_cell);
  void delete_particles(const std::size_t Np_surp, const std::size_t Npc,
                        const std::size_t cidx);
  void initialize_random_position(Point& xp_new,
                                  const std::vector<double>& x_min_max,
                                  const Cell& dolfin_cell);

  // TODO: Method needs careful checking
  void check_bounded_update(Point& pfeval, const std::size_t idx_func);

  // Access to particles
  particles* _P;

  // Min/Max number of particles
  std::size_t _np_min, _np_max;

  // List of functions
  std::vector<std::shared_ptr<const Function>> _FList;

  //
  std::vector<std::size_t> _pbound;
  std::vector<double> _bounds;
};
} // namespace dolfin
#endif // ADDDELETE_H
