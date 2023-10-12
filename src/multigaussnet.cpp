#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <glmnetpp>
#include "driver.h"
#include "internal.h"
#include "update_pb.h"

using namespace glmnetpp;
namespace py = pybind11;

// Multi-response Gaussian for dense X.
py::dict multigauss_exp(
    double parm,
    int ni,
    int no,
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
    int jerr,
    double fdev,   // begin glmnet.control
    double eps,
    double big,
    int mnlam,
    double devmax,
    double pmin,
    double exmx,
    int itrace,
    double prec,
    int mxit,
    double epsnr,
    int mxitnr     //end glmnet.control
    )
{
  InternalParams params = ::InternalParams();

  params.sml = fdev;
  params.eps = eps;
  params.big = big;
  params.mnlam = mnlam;
  params.rsqmax = devmax; // change of name
  params.pmin = pmin;
  params.exmx = exmx;
  params.itrace = itrace;
  params.bnorm_thr = prec;
  params.bnorm_mxit = mxit;
  params.epsnr = epsnr;
  params.mxitnr = mxitnr;

    using elnet_driver_t = ElnetDriver<util::glm_type::gaussian>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, x, y, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, jsd, intr, maxit,
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr,
                [&](int v) {update_pb(pb, v);}, params);
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
py::dict spmultigauss_exp(
    double parm,
    int ni,
    int no,
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
    int jerr,
    double fdev,   // begin glmnet.control
    double eps,
    double big,
    int mnlam,
    double devmax,
    double pmin,
    double exmx,
    int itrace,
    double prec,
    int mxit,
    double epsnr,
    int mxitnr     //end glmnet.control
    )
{

  InternalParams params = ::InternalParams();

  params.sml = fdev;
  params.eps = eps;
  params.big = big;
  params.mnlam = mnlam;
  params.rsqmax = devmax; // change of name
  params.pmin = pmin;
  params.exmx = exmx;
  params.itrace = itrace;
  params.bnorm_thr = prec;
  params.bnorm_mxit = mxit;
  params.epsnr = epsnr;
  params.mxitnr = mxitnr;

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
                parm, eigen_x, y, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, jsd, intr, maxit,
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr,
                [&](int v) {update_pb(pb, v);}, params);
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

PYBIND11_MODULE(_multigaussnet, m) {

    m.def("multigaussnet", &multigauss_exp,
	  py::arg("parm"),
	  py::arg("ni"),
	  py::arg("no"),
	  py::arg("x"),
	  py::arg("y"),
	  py::arg("w"),
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
	  py::arg("jsd"),
	  py::arg("intr"),
	  py::arg("maxit"),
	  py::arg("pb"),
	  py::arg("lmu"),
	  py::arg("a0"),
	  py::arg("ca"),
	  py::arg("ia"),
	  py::arg("nin"),
	  py::arg("rsq"),
	  py::arg("alm"),
	  py::arg("nlp"),
	  py::arg("jerr"),
	  py::arg("fdev"),
	  py::arg("eps"),
	  py::arg("big"),
	  py::arg("mnlam"),
	  py::arg("devmax"),
	  py::arg("pmin"),
	  py::arg("exmx"),
	  py::arg("itrace"),
	  py::arg("prec"),
	  py::arg("mxit"),
	  py::arg("epsnr"),
	  py::arg("mxitnr"));
    
    m.def("spmultigaussnet", &spmultigauss_exp,
	  py::arg("parm"),
	  py::arg("ni"),
	  py::arg("no"),
	  py::arg("x_data_array"),
	  py::arg("x_indices_array"),
	  py::arg("x_indptr_array"),
	  py::arg("y"),
	  py::arg("w"),
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
	  py::arg("jsd"),
	  py::arg("intr"),
	  py::arg("maxit"),
	  py::arg("pb"),
	  py::arg("lmu"),
	  py::arg("a0"),
	  py::arg("ca"),
	  py::arg("ia"),
	  py::arg("nin"),
	  py::arg("rsq"),
	  py::arg("alm"),
	  py::arg("nlp"),
	  py::arg("jerr"),
	  py::arg("fdev"),
	  py::arg("eps"),
	  py::arg("big"),
	  py::arg("mnlam"),
	  py::arg("devmax"),
	  py::arg("pmin"),
	  py::arg("exmx"),
	  py::arg("itrace"),
	  py::arg("prec"),
	  py::arg("mxit"),
	  py::arg("epsnr"),
	  py::arg("mxitnr"));

}
