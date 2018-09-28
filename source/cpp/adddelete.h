#ifndef ADDDELETE_H
#define ADDDELETE_H

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/math/basic.h>
#include <dolfin/common/Array.h>

#include "particle.h"
#include "particles.h"
#include "utils.h"

namespace dolfin{
    class AddDelete
    {
    public:
        AddDelete(std::vector<std::shared_ptr<const Function>> FList);
        AddDelete(particles& P, std::size_t np_min, std::size_t np_max,
                  std::vector<std::shared_ptr<const Function>> FList);
        AddDelete(particles &P, std::size_t np_min, std::size_t np_max,
                    std::vector<std::shared_ptr<const Function> > FList,
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
        void insert_particles_weighted(const std::size_t Np_def, const Cell& dolfin_cell);
        void delete_particles(const std::size_t Np_surp, const std::size_t Npc, const std::size_t cidx);
        void initialize_random_position(Point& xp_new, const std::vector<double>& x_min_max,
                                        const Cell& dolfin_cell);

        //TODO: Method needs careful checking
        void check_bounded_update(Point& pfeval, const std::size_t idx_func);

        // Min/Max number of particles
        std::size_t _np_min, _np_max;

        // List of functions
        std::vector<std::shared_ptr<const Function>> _FList;

        //
        std::vector<std::size_t> _pbound;
        std::vector<double> _bounds;

        // Access to particles
        particles* _P;
    };
}
#endif // ADDDELETE_H
