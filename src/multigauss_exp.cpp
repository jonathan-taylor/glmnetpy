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
    SEXP pb,
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

// Multi-response Gaussian for sparse X.
py::dict multspelnet_exp(
    double parm,
    const Eigen::Ref<Eigen::SparseMatrix<double> > x,
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
    SEXP pb,
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
