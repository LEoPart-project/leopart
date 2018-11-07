// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#ifndef ADVECT_PARTICLES_H
#define ADVECT_PARTICLES_H

#include <ufc.h>
#include "particle.h"
#include "particles.h"
#include "utils.h"

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <Eigen/Dense>

namespace dolfin{

    typedef struct facet_info_t
    {
      Point midpoint;
      Point normal;
    } facet_info;

    class advect_particles
    {

    public:
        // Constructors
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const std::string type1,
                         const std::string update_particle = "none");

        // Document
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const std::string type1,
                         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                         const std::string update_particle = "none");

        // Document
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const BoundaryMesh& bmesh, const std::string type1,
                         Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                         const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                         const std::string update_particle = "none");

        // Document
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const BoundaryMesh& bmesh, const std::string type1, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                         const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                         const std::string update_particle = "none" );

        // Step forward in time dt
        void do_step(double dt);

        // Destructor
        ~advect_particles();

    protected:
        particles* _P;

        void set_facets_info();
        void set_bfacets(const std::string btype);
        void set_bfacets(const BoundaryMesh& bmesh, const std::string btype, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> bidcs);

        std::vector<std::size_t> boundary_facets();
        std::vector<std::size_t> boundary_facets(const BoundaryMesh& bmesh, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> bidcs);

        // Initialize open, closed and periodic facets
        std::vector<std::size_t> obc_facets, cbc_facets, pbc_facets;
        std::vector<std::vector<double>>  pbc_lims;     // Coordinates of limits
        bool pbc_active = false;

        // Update particle info?
        const std::string update_particle;

        // Timestepping scheme related
        std::vector<double> dti;
        std::vector<double> weights;

        std::size_t _space_dimension, _value_size_loc;

        std::vector<facet_info> facets_info;
        std::vector<std::vector<std::size_t>> cell2facet;

        Function* uh;
        std::shared_ptr<const FiniteElement> _element;

        // Must receive a point xp
        std::tuple<std::size_t, double> time2intersect(std::size_t cidx, double dt, const Point xp, const Point up);

        // Consider placing in particle class
        //void push_particle(const double dt, const Point& up, const std::size_t cidx, const std::size_t pidx);

        // Methods for applying bc's
        void apply_open_bc(std::size_t cidx, std::size_t pidx);
        void apply_closed_bc(double dt, Point& up, std::size_t cidx, std::size_t pidx, std::size_t fidx);
        void apply_periodic_bc(double dt, Point& up, std::size_t cidx,  std::size_t pidx, std::size_t fidx);

        void pbc_limits_violation(std::size_t cidx, std::size_t pidx);

        // TODO: Make pure virtual function for do_step?
        // Method for substepping in multistep schemes

        void do_substep(double dt, Point& up, const std::size_t cidx, std::size_t* pidx,
                       const std::size_t step, const std::size_t num_steps,
                       const std::size_t xp0_idx, const std::size_t up0_idx,
                       std::vector<std::size_t>& reloc_local_c, std::vector<particle>& reloc_local_p);

    };

    class advect_rk2 : protected advect_particles
    {
    public:
        // Constructors
        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
                   const std::string update_particle = "none" );
        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                   const std::string update_particle = "none" );
        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1, const std::string type2,
                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                   const std::string update_particle = "none" );

        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1, const std::string type2,
                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                   const std::string update_particle = "none" );

        // Destructor
         ~advect_rk2();

        void do_step(double dt);

        // Something on particle updaters

    private:
        std::size_t xp0_idx, up0_idx;

        void update_particle_template()
        {
          const std::size_t gdim = _P->mesh()->geometry().dim();
          xp0_idx = _P->expand_template(gdim);
          up0_idx = _P->expand_template(gdim);

          // Copy position to xp0 property
          for (unsigned int cidx = 0; cidx < _P->mesh()->num_cells(); ++cidx)
          {
            for (unsigned int pidx = 0; pidx < _P->num_cell_particles(cidx); ++pidx)
              _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));
          }
        }

        void init_weights()
        {
          dti = {1.0, 1.0};
          weights = {0.5, 0.5};
        }
    };

    class advect_rk3 : protected advect_particles
    {
    public:
        // Constructors
        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
                   const std::string update_particle = "none" );
        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const std::string type1,
                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                   const std::string update_particle = "none" );
        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                   const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                   const std::string update_particle = "none" );

        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices1,
                   const std::string type2, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> indices2,
                   Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>> pbc_limits,
                   const std::string update_particle = "none" );

        // Destructor
         ~advect_rk3();

        void do_step(double dt);

    private:
        std::size_t xp0_idx, up0_idx;

        void update_particle_template()
        {
          const std::size_t gdim = _P->mesh()->geometry().dim();
          xp0_idx = _P->expand_template(gdim);
          up0_idx = _P->expand_template(gdim);

          // Copy position to xp0 property
          for (unsigned int cidx = 0; cidx < _P->mesh()->num_cells(); ++cidx)
          {
            for (unsigned int pidx = 0; pidx < _P->num_cell_particles(cidx); ++pidx)
              _P->set_property(cidx, pidx, xp0_idx, _P->x(cidx, pidx));
          }
        }

        void init_weights()
        {
          dti = {0.5, 0.75, 1.0};
          weights = {2./9., 3./9., 4./9.};
        }
    };
}
#endif // ADVECT_PARTICLES_H
