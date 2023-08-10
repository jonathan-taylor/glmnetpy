#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
// #include <Eigen/Dense>
#include <Eigen.h>
#include <glmnetpp>
#include "driver.h"

using namespace glmnetpp;
namespace py = pybind11;

// WLS for dense X.
py::dict wls_exp(
		 double alm0,
		 double almc,
		 double alpha,
		 int m,
		 int no,
		 int ni,
		 const Eigen::Map<Eigen::MatrixXd> x,
		 Eigen::Map<Eigen::VectorXd> r,
		 Eigen::Map<Eigen::VectorXd> xv,
		 const Eigen::Map<Eigen::VectorXd> v,
		 int intr,
		 const Eigen::Map<Eigen::VectorXi> ju,
		 const Eigen::Map<Eigen::VectorXd> vp,
		 const Eigen::Map<Eigen::MatrixXd> cl,
		 int nx,
		 double thr,
		 int maxit,
		 Eigen::Map<Eigen::VectorXd> a,
		 double aint,
		 Eigen::Map<Eigen::VectorXd> g,
		 Eigen::Map<Eigen::VectorXi> ia,
		 Eigen::Map<Eigen::VectorXi> iy,
		 int iz,
		 Eigen::Map<Eigen::VectorXi> mm,
		 int nino,
		 double rsqc,
		 int nlp,
		 int jerr
		 ) {
  using internal_t = ElnetPointInternal<
    util::glm_type::gaussian,
    util::mode_type<util::glm_type::gaussian>::wls,
    double, int, int>;
  using elnet_point_t = ElnetPoint<
    util::glm_type::gaussian,
    util::mode_type<util::glm_type::gaussian>::wls,
    internal_t>;
  auto f = [&]() {
    elnet_point_t elnet_point(
			      alm0, almc, alpha, x, r, xv, v, intr, ju, vp,
			      cl, nx, thr, maxit, a, aint, g, 
			      ia, iy, iz, mm, nino, rsqc, nlp);
    elnet_point.fit(m, jerr);
  };
  run(f, jerr);

  py::dict result;
  
  result["almc"] = almc;
  result["r"] = r;
  result["xv"] = xv;
  result["ju"] = ju;
  result["vp"] = vp;
  result["cl"] = cl;
  result["nx"] = nx;
  result["a"] = a;
  result["aint"] = aint;
  result["g"] = g;
  result["ia"] = ia;
  result["iy"] = iy;
  result["iz"] = iz;
  result["mm"] = mm;
  result["nino"] = nino;
  result["rsqc"] = rsqc;
  result["nlp"] = nlp;
  result["jerr"] = jerr;

  return result;
}

// WLS for sparse X.
py::dict spwls_exp(
    double alm0,
    double almc,
    double alpha,
    int m,
    int no,
    int ni,
    const Eigen::Map<Eigen::SparseMatrix<double>> x,
    const Eigen::Map<Eigen::VectorXd> xm,
    const Eigen::Map<Eigen::VectorXd> xs,
    Eigen::Map<Eigen::VectorXd> r,
    Eigen::Map<Eigen::VectorXd> xv,
    const Eigen::Map<Eigen::VectorXd> v,
    int intr,
    const Eigen::Map<Eigen::VectorXi> ju,
    const Eigen::Map<Eigen::VectorXd> vp,
    const Eigen::Map<Eigen::MatrixXd> cl,
    int nx,
    double thr,
    int maxit,
    Eigen::Map<Eigen::VectorXd> a,
    double aint,
    Eigen::Map<Eigen::VectorXd> g,
    Eigen::Map<Eigen::VectorXi> ia,
    Eigen::Map<Eigen::VectorXi> iy,
    int iz,
    Eigen::Map<Eigen::VectorXi> mm,
    int nino,
    double rsqc,
    int nlp,
    int jerr
    ) {
    using internal_t = SpElnetPointInternal<
        util::glm_type::gaussian,
        util::mode_type<util::glm_type::gaussian>::wls,
        double, int, int>;
    using elnet_point_t = SpElnetPoint<
        util::glm_type::gaussian,
        util::mode_type<util::glm_type::gaussian>::wls,
        internal_t>;
    auto f = [&]() {
        elnet_point_t elnet_point(
                alm0, almc, alpha, x, r, xm, xs, xv, v, intr, ju, vp,
                cl, nx, thr, maxit, a, aint, g, 
                ia, iy, iz, mm, nino, rsqc, nlp);
        elnet_point.fit(m, jerr);
    };
    run(f, jerr);

    py::dict result;
    
    result["almc"] = almc;
    result["r"] = r;
    result["xv"] = xv;
    result["ju"] = ju;
    result["vp"] = vp;
    result["cl"] = cl;
    result["nx"] = nx;
    result["a"] = a;
    result["aint"] = aint;
    result["g"] = g;
    result["ia"] = ia;
    result["iy"] = iy;
    result["iz"] = iz;
    result["mm"] = mm;
    result["nino"] = nino;
    result["rsqc"] = rsqc;
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
	  py::arg("x"),
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
}
