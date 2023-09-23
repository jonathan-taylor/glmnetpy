#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <glmnetpp>
#include "driver.h"
#include "internal.h"

using namespace glmnetpp;
namespace py = pybind11;

void update_pb(py::object, int);

// Binomial/Multinomial for dense X.
py::dict lognet(
    double parm,
    int ni,
    int no,
    Eigen::Ref<Eigen::MatrixXd> x,          // TODO: map?
    Eigen::Ref<Eigen::MatrixXd> y,          // TODO: map?
    Eigen::Ref<Eigen::MatrixXd> g,          // TODO: map? 
    const Eigen::Ref<Eigen::VectorXi> jd,
    const Eigen::Ref<Eigen::VectorXd> vp,
    Eigen::MatrixXd cl,         // TODO: map?
    int ne,
    int nx,
    int nlam,
    double flmin,
    const Eigen::Ref<Eigen::VectorXd> ulam,
    double thr,
    int isd,
    int intr,
    int maxit,
    int kopt,
    py::object pb,
    int lmu,
    Eigen::Ref<Eigen::MatrixXd> a0,
    Eigen::Ref<Eigen::VectorXd> ca,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> nin,
    double nulldev,
    Eigen::Ref<Eigen::VectorXd> dev,
    Eigen::Ref<Eigen::VectorXd> alm,
    int nlp,
    int jerr
    )
{
    using elnet_driver_t = ElnetDriver<util::glm_type::binomial>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, x, y, g, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, intr, maxit, kopt,
                lmu, a0, ca, ia, nin, nulldev, dev, alm, nlp, jerr,
                [&](int v) {update_pb(pb, v);}, ::InternalParams());
    };
    run(f, jerr);

  py::dict result;

  result["a0"] = a0;
  result["nin"] = nin;
  result["alm"] = alm;
  result["ca"] = ca;
  result["ia"] = ia;
  result["lmu"] = lmu;
  result["nulldev"] = nulldev;
  result["dev"] = dev;
  result["nlp"] = nlp;
  result["jerr"] = jerr;

  return result;
}

// Lognet for sparse X.
py::dict splognet(
    double parm,
    int ni,
    int no,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_data_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> x_indices_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> x_indptr_array,
    Eigen::MatrixXd y,          // TODO: map?
    Eigen::MatrixXd g,          // TODO: map? 
    const Eigen::Ref<Eigen::VectorXi> jd,
    const Eigen::Ref<Eigen::VectorXd> vp,
    Eigen::MatrixXd cl,         // TODO: map?
    int ne,
    int nx,
    int nlam,
    double flmin,
    const Eigen::Ref<Eigen::VectorXd> ulam,
    double thr,
    int isd,
    int intr,
    int maxit,
    int kopt,
    py::object pb,
    int lmu,
    Eigen::Ref<Eigen::MatrixXd> a0,
    Eigen::Ref<Eigen::VectorXd> ca,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> nin,
    double nulldev,
    Eigen::Ref<Eigen::VectorXd> dev,
    Eigen::Ref<Eigen::VectorXd> alm,
    int nlp,
    int jerr
    )
{

    // Map the scipy csc_matrix x  to Eigen
    // This prevents copying. However, note the lack of 'const' use, but we take care not to change data
    Eigen::Map<Eigen::VectorXd> x_data_map(x_data_array.mutable_data(),
					   x_data_array.size());
    Eigen::Map<Eigen::VectorXi> x_indices_map(x_indices_array.mutable_data(),
					      x_indices_array.size());
    Eigen::Map<Eigen::VectorXi> x_indptr_map(x_indptr_array.mutable_data(),
					     x_indptr_array.size());
    // Create MappedSparseMatrix from the mapped arrays
    Eigen::MappedSparseMatrix<double, Eigen::ColMajor> eigen_x(no,
							       ni,
							       x_data_array.size(), 
							       x_indptr_map.data(),
							       x_indices_map.data(),
							       x_data_map.data());

    using elnet_driver_t = ElnetDriver<util::glm_type::binomial>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, eigen_x, y, g, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, intr, maxit, kopt,
                lmu, a0, ca, ia, nin, nulldev, dev, alm, nlp, jerr,
                [&](int v) {update_pb(pb, v);}, ::InternalParams());
    };
    run(f, jerr);

  py::dict result;

  result["a0"] = a0;
  result["nin"] = nin;
  result["alm"] = alm;
  result["ca"] = ca;
  result["ia"] = ia;
  result["lmu"] = lmu;
  result["nulldev"] = nulldev;
  result["dev"] = dev;
  result["nlp"] = nlp;
  result["jerr"] = jerr;

  return result;
}

PYBIND11_MODULE(_lognet, m) {
    m.def("lognet", &lognet,
	  py::arg("parm"),
	  py::arg("ni"),
	  py::arg("no"),
	  py::arg("x"),
	  py::arg("y"),
	  py::arg("g"),
	  py::arg("jd"),
	  py::arg("vp"),
	  py::arg("cl"),	  
	  py::arg("ne"),
	  py::arg("nx"),
	  py::arg("nlam"),
	  py::arg("flmin"),
	  py::arg("ulam"),
	  py::arg("thr"),
	  py::arg("isd"),
	  py::arg("intr"),
	  py::arg("maxit"),
	  py::arg("kopt"),
	  py::arg("pb"),
	  py::arg("lmu"),
	  py::arg("a0"),
	  py::arg("ca"),
	  py::arg("ia"),
	  py::arg("nin"),
	  py::arg("nulldev"),
	  py::arg("dev"),
	  py::arg("alm"),
	  py::arg("nlp"),
	  py::arg("jerr"));
    
    m.def("splognet", &splognet,
	  py::arg("parm"),
	  py::arg("ni"),
	  py::arg("no"),
	  py::arg("x_data_array"),
	  py::arg("x_indices_array"),
	  py::arg("x_indptr_array"),
	  py::arg("y"),
	  py::arg("g"),
	  py::arg("jd"),
	  py::arg("vp"),
	  py::arg("cl"),	  
	  py::arg("ne"),
	  py::arg("nx"),
	  py::arg("nlam"),
	  py::arg("flmin"),
	  py::arg("ulam"),
	  py::arg("thr"),
	  py::arg("isd"),
	  py::arg("intr"),
	  py::arg("maxit"),
	  py::arg("kopt"),
	  py::arg("pb"),
	  py::arg("lmu"),
	  py::arg("a0"),
	  py::arg("ca"),
	  py::arg("ia"),
	  py::arg("nin"),
	  py::arg("nulldev"),
	  py::arg("dev"),
	  py::arg("alm"),
	  py::arg("nlp"),
	  py::arg("jerr"));

}
