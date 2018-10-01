// Author: Jakob Maljaars
// Contact: j.m.maljaars _at_ tudelft.nl/jakobmaljaars _at_ gmail.com

#ifndef PARTICLES_H
#define PARTICLES_H

#include "utils.h"
#include "particle.h"
#include <iostream>
#include <vector>
#include <limits>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>

#include <dolfin/fem/FiniteElement.h>
#include <Eigen/Dense>

namespace dolfin{
    class particles
    {
    friend class l2projection;

    friend class advect_particles;
    friend class advect_rk2;
    friend class advect_rk3;
    friend class PDEStaticCondensation;
    friend class AddDelete;

    public:
        particles(const Array<double>& p_array, const Array<int>& p_template, const int p_num,
                  const Mesh& mesh);

        ~particles();

        // Interpolate function to particles
        void interpolate(const Function& phih, const std::size_t property_idx);

        std::vector<double> get_positions();
        std::vector<double> get_property(const std::size_t idx);

    private:
        // Push particle to new position
        void push_particle(const double dt, const Point& up, const std::size_t cidx, const std::size_t pidx);

        // Initialize bounding boxes
        void make_bounding_boxes();
        // Update bounding boxes (on moving mesh)
        void update_bounding_boxes();
        // Check if point in bounding box
        static bool in_bounding_box(const std::vector<double>& point,
                                    const std::vector<double>& bounding_box,
                                    const double tol);

        // Particle collector, required in parallel
        void particle_communicator_collect(std::vector<std::vector<particle>>& comm_snd,
                                           const std::size_t cidx, const std::size_t pidx);
        // Particle pusher, required in parallel
        void particle_communicator_push(std::vector<std::vector<particle>>& comm_snd);
        // Unpack particle, required in parallel
        std::vector<double> unpack_particle(const particle part);

        void get_particle_contributions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
                                        Eigen::Matrix<double, Eigen::Dynamic, 1>& f,
                                        const Cell& dolfin_cell, std::shared_ptr<const FiniteElement> element,
                                        const std::size_t space_dimension, const std::size_t value_size_loc,
                                        const std::size_t property_idx);

        // TODO: locate/relocate funcionality

        // Attributes
        const Mesh* _mesh;
        std::size_t _Ndim;
        const unsigned int _num_cells;
        std::vector<std::vector<particle> >  _cell2part;

        // Particle properties
        std::size_t _Np;
        std::vector<std::size_t> _ptemplate;
        std::size_t _plen;

        // Needed for parallel
        const MPI_Comm _mpi_comm;
        const std::size_t _num_processes;
        std::vector<std::vector<double>> _bounding_boxes;

    };
}

#endif // PARTICLES_H
