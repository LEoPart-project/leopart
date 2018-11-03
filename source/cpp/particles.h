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

      // TODO: get rid of friends!
      friend class advect_particles;
      friend class advect_rk2;
      friend class advect_rk3;
      friend class AddDelete;

    public:
    particles(Eigen::Ref<const Eigen::Array<double,
              Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_array,
              const std::vector<unsigned int>& p_template,
              const Mesh& mesh);

    ~particles();

    // Get the position of a particle in a cell
    Point x(int cell_index, int particle_index)
    {
      return _cell2part[cell_index][particle_index][0];
    }

    // Return property i of particle in cell
    const Point& property(int cell_index, int particle_index,
                          int i) const
    {
      return _cell2part[cell_index][particle_index][i];
    }

    // Pointer to the mesh
    const Mesh* mesh() const
    {
      return _mesh;
    }

    // Get size of property i
    unsigned int ptemplate(int i)
    {
      return _ptemplate[i];
    }

    // Number of properties
    unsigned int num_properties()
    {
      return _ptemplate.size();
    }

    // Number of particles in Cell c
    unsigned int num_cell_particles(int c)
    {
      return _cell2part[c].size();
    }

    // Interpolate function to particles
    void interpolate(const Function& phih, const std::size_t property_idx);

        // Increment
        void increment(const Function& phih_new, const Function& phih_old, const std::size_t property_idx);

        // Increment using theta --> Consider replacing property_idcs
        void increment(const Function& phih_new, const Function& phih_old,
                       Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> property_idcs,
                       const double theta, const std::size_t step);

        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          positions();
        std::vector<double> get_property(const std::size_t idx);

        void get_particle_contributions(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& q,
                                        Eigen::Matrix<double, Eigen::Dynamic, 1>& f,
                                        const Cell& dolfin_cell, std::shared_ptr<const FiniteElement> element,
                                        const std::size_t space_dimension, const std::size_t value_size_loc,
                                        const std::size_t property_idx);

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

        // TODO: locate/relocate funcionality

        // Attributes
        const Mesh* _mesh;
        std::size_t _Ndim;
        std::vector<std::vector<particle> >  _cell2part;

        // Particle properties
        std::vector<unsigned int> _ptemplate;
        std::size_t _plen;

        // Needed for parallel
        const MPI_Comm _mpi_comm;
        std::vector<std::vector<double>> _bounding_boxes;

        // TO REMOVE
        int dummy, dummy2, dummy3, dummy4, dummy5;

    };
}

#endif // PARTICLES_H
