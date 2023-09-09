#include "internal.h"
#include <cstddef>
//#include <Rcpp.h>
//#include <RcppEigen.h>
// ORDER MATTERS
//#include <R.h>
//#include <Rinternals.h>

extern "C" {

void F77_SUB(chg_fract_dev)(double*); 
void F77_SUB(chg_dev_max)(double*); 
void F77_SUB(chg_min_flmin)(double*); 
void F77_SUB(chg_big)(double*); 
void F77_SUB(chg_min_lambdas)(int*); 
void F77_SUB(chg_min_null_prob)(double*); 
void F77_SUB(chg_max_exp)(double*); 
void F77_SUB(chg_itrace)(int*); 
void F77_SUB(chg_bnorm)(double*, int*); 
void F77_SUB(chg_epsnr)(double*); 
void F77_SUB(chg_mxitnr)(int*); 

} // end extern "C"

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

void chg_fract_dev(double arg) { /*TODO*/ F77_SUB(chg_fract_dev)(&arg); InternalParams::sml = arg; }

void chg_dev_max(double arg) { /*TODO*/ F77_SUB(chg_dev_max)(&arg); InternalParams::rsqmax = arg; }

void chg_min_flmin(double arg) { /*TODO*/ F77_SUB(chg_min_flmin)(&arg); InternalParams::eps = arg; }

void chg_big(double arg) { /*TODO*/ F77_SUB(chg_big)(&arg); InternalParams::big = arg; }

void chg_min_lambdas(int irg) { /*TODO*/ F77_SUB(chg_min_lambdas)(&irg); InternalParams::mnlam = irg; }

void chg_min_null_prob(double arg) { /*TODO*/ F77_SUB(chg_min_null_prob)(&arg); InternalParams::pmin = arg; }

void chg_max_exp(double arg) { /*TODO*/ F77_SUB(chg_max_exp)(&arg); InternalParams::exmx = arg; }

void chg_itrace(int irg) { /*TODO*/ F77_SUB(chg_itrace)(&irg); InternalParams::itrace = irg; }

void chg_bnorm(double arg, int irg) { 
    /*TODO*/
    F77_SUB(chg_bnorm)(&arg, &irg);
    InternalParams::bnorm_thr = arg; 
    InternalParams::bnorm_mxit = irg; 
}

py::dict get_bnorm(double& prec, int& mxit) {
    prec = InternalParams::bnorm_thr; 
    mxit = InternalParams::bnorm_mxit;

  py::dict result;

  result["prec"] = prec;
  result["mixit"] = mxit;

  return result;
}

void chg_epsnr(double arg) { /*TODO*/ F77_SUB(chg_epsnr)(&arg); InternalParams::epsnr = arg; }

void chg_mxitnr(int irg) { /*TODO*/ F77_SUB(chg_mxitnr)(&irg); InternalParams::mxitnr = irg; }

