#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <glmnetpp>
#include "driver.h"
#include "internal.h"

using namespace glmnetpp;
namespace py = pybind11;

// Multi-response Gaussian for dense X.
py::dict multelnet_exp(
    double parm,
    Eigen::Ref<Eigen::MatrixXd> x,          // TODO: map?
    Eigen::Ref<Eigen::MatrixXd> y,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> w,          // TODO: map?
    const Eigen::Ref<Eigen::VectorXi> jd,
    const Eigen::Ref<Eigen::VectorXd> vp,
    const Eigen::Ref<Eigen::MatrixXd> cl,
    int ne,
    int nx,
    int nlam,
    double flmin,
    const Eigen::Ref<Eigen::VectorXd> ulam,
    double thr,
    int isd,
    int jsd,
    int intr,
    int maxit,
    py::object pb,
    int lmu,
    Eigen::Ref<Eigen::MatrixXd> a0,
    Eigen::Ref<Eigen::VectorXd> ca,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> nin,
    Eigen::Ref<Eigen::VectorXd> rsq,
    Eigen::Ref<Eigen::VectorXd> alm,
    int nlp,
    int jerr
    )
{
    using elnet_driver_t = ElnetDriver<util::glm_type::gaussian>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, x, y, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, jsd, intr, maxit,
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr,
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
    result["rsq"] = rsq;
    result["nlp"] = nlp;
    result["jerr"] = jerr;

    return result;
}

// Multi-response Gaussian for sparse X.
py::dict multspelnet_exp(
    double parm,
    py::array_t<double, py::array::c_style | py::array::forcecast> x_data_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> x_indices_array,
    py::array_t<int, py::array::c_style | py::array::forcecast> x_indptr_array,
    Eigen::Ref<Eigen::MatrixXd> y,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> w,          // TODO: map?
    const Eigen::Ref<Eigen::VectorXi> jd,
    const Eigen::Ref<Eigen::VectorXd> vp,
    const Eigen::Ref<Eigen::MatrixXd> cl,
    int ne,
    int nx,
    int nlam,
    double flmin,
    const Eigen::Ref<Eigen::VectorXd> ulam,
    double thr,
    int isd,
    int jsd,
    int intr,
    int maxit,
    py::object pb,
    int lmu,
    Eigen::Ref<Eigen::MatrixXd> a0,
    Eigen::Ref<Eigen::VectorXd> ca,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> nin,
    Eigen::Ref<Eigen::VectorXd> rsq,
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

    using elnet_driver_t = ElnetDriver<util::glm_type::gaussian>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, eiegn_x, y, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, jsd, intr, maxit,
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr,
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
    result["rsq"] = rsq;
    result["nlp"] = nlp;
    result["jerr"] = jerr;

    return result;
}

PYBIND11_MODULE(glmnetpp, m) {
    m.def("wls", &wls_exp,
	  py::arg("alm0"),
	  py::arg("almc"),
	  py::arg("alpha"),
	  py::arg("m"),
	  py::arg("no"),
	  py::arg("ni"),
	  py::arg("x"),	  
	  py::arg("r"),
	  py::arg("xv"),
	  py::arg("v"),
	  py::arg("intr"),
	  py::arg("ju"),
	  py::arg("vp"),
	  py::arg("cl"),
	  py::arg("nx"),
	  py::arg("thr"),
	  py::arg("maxit"),
	  py::arg("a"),
	  py::arg("aint"),
	  py::arg("g"),
	  py::arg("ia"),
	  py::arg("iy"),
	  py::arg("iz"),
	  py::arg("mm"),
	  py::arg("nino"),
	  py::arg("rsqc"),
	  py::arg("nlp"),
	  py::arg("jerr"));
    
    m.def("spwls", &spwls_exp,
	  py::arg("alm0"),
	  py::arg("almc"),
	  py::arg("alpha"),
	  py::arg("m"),
	  py::arg("no"),
	  py::arg("ni"),
	  py::arg("x_data_array"),
	  py::arg("x_indices_array"),
	  py::arg("x_indptr_array"),
	  py::arg("xm"),
	  py::arg("xs"),
	  py::arg("r"),
	  py::arg("xv"),
	  py::arg("v"),
	  py::arg("intr"),
	  py::arg("ju"),
	  py::arg("vp"),
	  py::arg("cl"),
	  py::arg("nx"),
	  py::arg("thr"),
	  py::arg("maxit"),
	  py::arg("a"),
	  py::arg("aint"),
	  py::arg("g"),
	  py::arg("ia"),
	  py::arg("iy"),
	  py::arg("iz"),
	  py::arg("mm"),
	  py::arg("nino"),
	  py::arg("rsqc"),
	  py::arg("nlp"),
	  py::arg("jerr"));

    m.def("update_pb", &update_pb);
}
