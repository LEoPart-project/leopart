
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <dolfin.h>

#include "particles.h"
#include "advect_particles.h"
#include "l2projection.h"
#include "pdestaticcondensation.h"
#include "formutils.h"


PYBIND11_MODULE(particle_wrapper, m)
{
  m.doc() = "example";

  py::class_<dolfin::particles>(m, "particles")
    .def(py::init<Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
         Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>,
         const int, const dolfin::Mesh&>())
    .def("interpolate", &dolfin::particles::interpolate)
    .def("get_positions", &dolfin::particles::get_positions)
    .def("get_property", &dolfin::particles::get_property);

  py::class_<dolfin::advect_particles>(m, "advect_particles")
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&, const dolfin::BoundaryMesh&, const std::string, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&, const dolfin::BoundaryMesh&, const std::string,
         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&,
         const dolfin::BoundaryMesh&, std::string , Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&,
         const dolfin::BoundaryMesh&, const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
         const std::string>())
    .def("do_step", &dolfin::advect_particles::do_step);


  py::class_<dolfin::advect_rk2>(m, "advect_rk2")
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&, const dolfin::BoundaryMesh&, const std::string, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&, const dolfin::BoundaryMesh&, const std::string,
         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&,
         const dolfin::BoundaryMesh&, std::string , Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&,
         const dolfin::BoundaryMesh&, const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
         const std::string>())
    .def("do_step", &dolfin::advect_rk2::do_step);


  py::class_<dolfin::advect_rk3>(m, "advect_rk3")
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&, const dolfin::BoundaryMesh&, const std::string, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&, const dolfin::BoundaryMesh&, const std::string,
         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&,
         const dolfin::BoundaryMesh&, std::string , Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>, const std::string>())
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, dolfin::Function&,
         const dolfin::BoundaryMesh&, const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         const std::string, Eigen::Ref<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>>,
         Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>,
         const std::string>())
    .def("do_step", &dolfin::advect_rk3::do_step);


  py::class_<dolfin::l2projection>(m, "l2projection")
    .def(py::init<dolfin::particles&, dolfin::FunctionSpace&, const std::size_t>())
    .def("project", (void (dolfin::l2projection::*)(dolfin::Function&)) &dolfin::l2projection::project)
    .def("project", (void (dolfin::l2projection::*)(dolfin::Function&, const double, const double)) &dolfin::l2projection::project)
    .def("project_cg", &dolfin::l2projection::project_cg);

  py::class_<dolfin::PDEStaticCondensation>(m, "PDEStaticCondensation")
    .def(py::init<const dolfin::Mesh&, dolfin::particles&,
         const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&,
         const std::size_t>())
    .def(py::init<const dolfin::Mesh&, dolfin::particles&,
         const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&, const dolfin::Form&,
         std::vector<std::shared_ptr<const dolfin::DirichletBC>>, const std::size_t>())
    .def("assemble", &dolfin::PDEStaticCondensation::assemble)
    .def("assemble_state_rhs", &dolfin::PDEStaticCondensation::assemble_state_rhs)
    .def("solve_problem", (void (dolfin::PDEStaticCondensation::*)(dolfin::Function&, dolfin::Function&, const std::string, const std::string))
    &dolfin::PDEStaticCondensation::solve_problem)
    .def("solve_problem", (void (dolfin::PDEStaticCondensation::*)(dolfin::Function&, dolfin::Function&, dolfin::Function&, const std::string, const std::string))
    &dolfin::PDEStaticCondensation::solve_problem)
    .def("apply_boundary", &dolfin::PDEStaticCondensation::apply_boundary);


}
