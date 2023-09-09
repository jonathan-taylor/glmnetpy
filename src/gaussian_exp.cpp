#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <glmnetpp>
#include "driver.h"
#include "internal.h"

using namespace glmnetpp;
namespace py = pybind11;

// Gaussian for dense X.
py::dict elnet_exp(
    int ka,
    double parm,
    Eigen::Ref<Eigen::MatrixXd> x,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> y,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> w,          // TODO: figure out if we should allow updating (safe choice is to copy)
    const Eigen::Ref<Eigen::VectorXi> jd,
    const Eigen::Ref<Eigen::VectorXd> vp,
    Eigen::Ref<Eigen::MatrixXd> cl,         // TODO: map?
    int ne,
    int nx,
    int nlam,
    double flmin,
    const Eigen::Ref<Eigen::VectorXd> ulam,
    double thr,
    int isd,
    int intr,
    int maxit,
    SEXP pb,
    int lmu,
    Eigen::Ref<Eigen::VectorXd> a0,
    Eigen::Ref<Eigen::MatrixXd> ca,
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
                ka == 2, parm, x, y, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd == 1, intr == 1, maxit, 
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr, 
                [&](int v) {setpb_cpp(pb, v);}, ::InternalParams());
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

// Gaussian for sparse X.
py::dict spelnet_exp(
    int ka,
    double parm,
    const Eigen::Ref<Eigen::SparseMatrix<double> > x,
    Eigen::Ref<Eigen::VectorXd> y, // TODO: map?
    Eigen::Ref<Eigen::VectorXd> w, // TODO: map?
    const Eigen::Ref<Eigen::VectorXi> jd,
    const Eigen::Ref<Eigen::VectorXd> vp,
    Eigen::Ref<Eigen::MatrixXd> cl, // TODO: map?
    int ne,
    int nx,
    int nlam,
    double flmin,
    const Eigen::Ref<Eigen::VectorXd> ulam,
    double thr,
    int isd,
    int intr,
    int maxit,
    SEXP pb,
    int lmu,
    Eigen::Ref<Eigen::VectorXd> a0,
    Eigen::Ref<Eigen::MatrixXd> ca,
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
                ka == 2, parm, x, y, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd == 1, intr == 1, maxit, 
                lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr, 
                [&](int v) {setpb_cpp(pb, v);}, ::InternalParams());
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
