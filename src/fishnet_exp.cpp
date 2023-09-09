#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <glmnetpp>
#include "driver.h"
#include "internal.h"

using namespace glmnetpp;
namespace py = pybind11;

// Poisson for dense X.
py::dict fishnet_exp(
    double parm,
    Eigen::Ref<Eigen::MatrixXd> x,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> y,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> g,          // TODO: map? 
    const Eigen::Ref<Eigen::VectorXd> w,
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
    SEXP pb,
    int lmu,
    Eigen::Ref<Eigen::VectorXd> a0,
    Eigen::Ref<Eigen::MatrixXd> ca,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> nin,
    double nulldev,
    Eigen::Ref<Eigen::VectorXd> dev,
    Eigen::Ref<Eigen::VectorXd> alm,
    int nlp,
    int jerr
    )
{
    using elnet_driver_t = ElnetDriver<util::glm_type::poisson>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, x, y, g, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, intr, maxit,
                lmu, a0, ca, ia, nin, nulldev, dev, alm, nlp, jerr,
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
  result["nulldev"] = nulldev;
  result["dev"] = dev;
  result["nlp"] = nlp;
  result["jerr"] = jerr;

  return result;
}

// Poisson for sparse X.
py::dict spfishnet_exp(
    double parm,
    const Eigen::Ref<Eigen::SparseMatrix<double>> x,
    Eigen::VectorXd y,          // TODO: map?
    Eigen::VectorXd g,          // TODO: map? 
    const Eigen::Ref<Eigen::VectorXd> w,
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
    SEXP pb,
    int lmu,
    Eigen::Ref<Eigen::VectorXd> a0,
    Eigen::Ref<Eigen::MatrixXd> ca,
    Eigen::Ref<Eigen::VectorXi> ia,
    Eigen::Ref<Eigen::VectorXi> nin,
    double nulldev,
    Eigen::Ref<Eigen::VectorXd> dev,
    Eigen::Ref<Eigen::VectorXd> alm,
    int nlp,
    int jerr
    )
{
    using elnet_driver_t = ElnetDriver<util::glm_type::poisson>;
    elnet_driver_t driver;
    auto f = [&]() {
        driver.fit(
                parm, x, y, g, w, jd, vp, cl, ne, nx, nlam, flmin,
                ulam, thr, isd, intr, maxit,
                lmu, a0, ca, ia, nin, nulldev, dev, alm, nlp, jerr,
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
  result["nulldev"] = nulldev;
  result["dev"] = dev;
  result["nlp"] = nlp;
  result["jerr"] = jerr;

  return result;
}
