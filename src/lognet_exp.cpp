#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <glmnetpp>
#include "driver.h"
#include "internal.h"

using namespace glmnetpp;
namespace py = pybind11;

// Binomial/Multinomial for dense X.
py::dict lognet_exp(
    double parm,
    Eigen::Ref<Eigen::MatrixXd> x,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> y,          // TODO: map?
    Eigen::Ref<Eigen::VectorXd> g,          // TODO: map? 
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
    SEXP pb,
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

// Lognet for sparse X.
py::dict splognet_exp(
    double parm,
    const Eigen::Ref<Eigen::SparseMatrix<double>> x,
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
    SEXP pb,
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
