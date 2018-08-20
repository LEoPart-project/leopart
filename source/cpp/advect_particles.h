// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#ifndef ADVECT_PARTICLES_H
#define ADVECT_PARTICLES_H

#include <ufc.h>
#include <particle.h>
#include <particles.h>
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

    typedef std::tuple<Facet, Point, Point, std::vector<std::size_t>, std::vector<bool> > facet_info;

    class advect_particles
    {
    //friend class RefillCells; // Refill cells makes use of static functions and maybe more importantly
                              // and, maybe more importantly, RefillCells must access (and change!) xp_o, dup_o, drhop_o
    //friend class PDEStaticCondensation;

    public:
        // Constructors
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const BoundaryMesh& bmesh, const std::string type1,
                         const std::string update_particle = "none");
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const BoundaryMesh& bmesh, const std::string type1,
                         const Array<double>& pbc_limits,
                         const std::string update_particle = "none");

        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const BoundaryMesh& bmesh, const std::string type1, const Array<std::size_t>& indices1,
                         const std::string type2, const Array<std::size_t>& indices2,
                         const std::string update_particle = "none");
        advect_particles(particles& P, FunctionSpace& U, Function& uhi,
                         const BoundaryMesh& bmesh, const std::string type1, const Array<std::size_t>& indices1,
                         const std::string type2, const Array<std::size_t>& indices2, const Array<double>& pbc_limits,
                         const std::string update_particle = "none" );

        void do_step(double dt);

        // Destructor
        ~advect_particles();

    protected:
        particles* _P;

        void set_facets_info();
        void set_bfacets(const BoundaryMesh& bmesh, const std::string btype);
        void set_bfacets(const BoundaryMesh& bmesh, const std::string btype, const Array<std::size_t>& bidcs);

        // TODO: void update_facet_info() (moving meshes)
        std::vector<double> subtract(std::vector<double>& u, std::vector<double>& v);

        std::vector<std::size_t> boundary_facets(const BoundaryMesh& bmesh);
        std::vector<std::size_t> boundary_facets(const BoundaryMesh& bmesh, const Array<std::size_t>& bidcs);
        std::vector<std::size_t> interior_facets();

        // Initialize interior, open, closed and periodic facets
        std::vector<std::size_t> int_facets, obc_facets,
                                 cbc_facets, pbc_facets;
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
                       std::vector<std::size_t>& reloc_local_c, std::vector<particle>& reloc_local_p,
                       std::vector<std::vector<particle>>& comm_snd);

    };

    class advect_rk2 : protected advect_particles
    {
    public:
        // Constructors
        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const std::string update_particle = "none" );
        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const Array<double>& pbc_limits,
                   const std::string update_particle = "none" );
        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const Array<std::size_t>& indices1, const std::string type2, const Array<std::size_t>& indices2,
                   const std::string update_particle = "none" );

        advect_rk2(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const Array<std::size_t>& indices1, const std::string type2, const Array<std::size_t>& indices2,
                   const Array<double>& pbc_limits,
                   const std::string update_particle = "none" );

        // Destructor
         ~advect_rk2();

        void do_step(double dt);

        // Something on particle updaters



    private:
        std::size_t xp0_idx, up0_idx;

        void update_particle_template(){
            xp0_idx = _P->_ptemplate.size();
            up0_idx = _P->_ptemplate.size()+1;

            // Modify particle template, by appending slots for old position/velocity
            std::vector<size_t> append = {_P->_Ndim, _P->_Ndim};
            _P->_ptemplate.insert(_P->_ptemplate.end(), append.begin(), append.end());
            _P->_plen += 2 * _P->_Ndim;

            // Make zero vector
            Point zero_point;
            // Loop over cells
            for( CellIterator ci(*(_P->_mesh)); !ci.end(); ++ci){
                for(int pidx = 0; pidx < _P->_cell2part[ci->index()].size() ; pidx++){
                    // Create 2 slots:
                    // At xp0 slot, push the position, at up0 slot put 0
                   _P->_cell2part[ci->index()][pidx].push_back(_P->_cell2part[ci->index()][pidx][0]);
                   _P->_cell2part[ci->index()][pidx].push_back(zero_point);
                }
            }
        }

        void init_weights(){
            dti = {1.0, 1.0};
            weights = {0.5, 0.5};
        }
    };

    class advect_rk3 : protected advect_particles
    {
    public:
        // Constructors
        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const std::string update_particle = "none" );
        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const Array<double>& pbc_limits,
                   const std::string update_particle = "none" );
        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const Array<std::size_t>& indices1, const std::string type2, const Array<std::size_t>& indices2,
                   const std::string update_particle = "none" );

        advect_rk3(particles& P, FunctionSpace& U, Function& uhi, const BoundaryMesh& bmesh, const std::string type1,
                   const Array<std::size_t>& indices1, const std::string type2, const Array<std::size_t>& indices2,
                   const Array<double>& pbc_limits,
                   const std::string update_particle = "none" );

        // Destructor
         ~advect_rk3();

        void do_step(double dt);
    private:
        std::size_t xp0_idx, up0_idx;

        void update_particle_template(){
            xp0_idx = _P->_ptemplate.size();
            up0_idx = _P->_ptemplate.size()+1;

            // Modify particle template, by appending slots for old position/velocity
            std::vector<size_t> append = {_P->_Ndim, _P->_Ndim};
            _P->_ptemplate.insert(_P->_ptemplate.end(), append.begin(), append.end());
            _P->_plen += 2 * _P->_Ndim;

            // Make zero vector
            Point zero_point;
            // Loop over cells
            for( CellIterator ci(*(_P->_mesh)); !ci.end(); ++ci){
                for(int pidx = 0; pidx < _P->_cell2part[ci->index()].size() ; pidx++){
                    // Create 2 slots:
                    // At xp0 slot, push the position, at up0 slot put 0
                   _P->_cell2part[ci->index()][pidx].push_back(_P->_cell2part[ci->index()][pidx][0]);
                   _P->_cell2part[ci->index()][pidx].push_back(zero_point);
                }
            }
        }

        void init_weights(){
            dti = {0.5,0.75,1.0};
            weights = {2./9., 3./9., 4./9.};
        }
    };
}
#endif // ADVECT_PARTICLES_H
