#include "internal.h"
#include <pybind11/pybind11.h>
#include <cstddef>

double InternalParams::sml = 1e-5;
double InternalParams::eps = 1e-6;
double InternalParams::big = 9.9e35;
int InternalParams::mnlam = 5;
double InternalParams::rsqmax = 0.999;
double InternalParams::pmin = 1e-9;
double InternalParams::exmx = 250.0;
int InternalParams::itrace = 0;
double InternalParams::bnorm_thr = 1e-10;
int InternalParams::bnorm_mxit = 100;
double InternalParams::epsnr = 1e-6;
int InternalParams::mxitnr = 25;

namespace py = pybind11;

py::dict get_int_parms(double& fdev,
		       double& eps,
		       double& big,
		       int& mnlam,
		       double& devmax,
		       double& pmin,
		       double& exmx,
		       int& itrace)
{
    fdev = InternalParams::sml; 
    eps = InternalParams::eps; 
    big = InternalParams::big; 
    mnlam = InternalParams::mnlam; 
    devmax = InternalParams::rsqmax;
    pmin = InternalParams::pmin; 
    exmx = InternalParams::exmx; 
    itrace = InternalParams::itrace;

    py::dict result;

    result["fdev"] = fdev;
    result["eps"] = eps;
    result["big"] = big;
    result["mnlam"] = mnlam;
    result["devmax"] = devmax;
    result["pmin"] = pmin;
    result["exmx"] = exmx;
    result["itrace"] = itrace;

    return result;
}

py::dict get_int_parms2(double& epsnr, int& mxitnr)
{
    epsnr = InternalParams::epsnr;
    mxitnr = InternalParams::mxitnr;

    py::dict result;
    
    result["epsnr"] = epsnr;
    result["mxitnr"] = mxitnr;

    return result;

}

py::dict get_bnorm(double& prec, int& mxit) {
    prec = InternalParams::bnorm_thr; 
    mxit = InternalParams::bnorm_mxit;

  py::dict result;

  result["prec"] = prec;
  result["mixit"] = mxit;

  return result;
}

PYBIND11_MODULE(glmnetpp_lognet, m) {

    m.def("get_int_parms", &get_int_parms,
	  py::arg("fdev"),
	  py::arg("eps"),
	  py::arg("big"),
	  py::arg("mnlam"),
	  py::arg("devmax"),
	  py::arg("pmin"),
	  py::arg("exmx"),
	  py::arg("itrace"));

    m.def("get_int_parms2", &get_int_parms2,
	  py::arg("epsnr"),
	  py::arg("mixitnr"));

    m.def("get_bnorm", &get_bnorm,
	  py::arg("bnorm_thr"),
	  py::arg("bnorm_mixit"));

}

