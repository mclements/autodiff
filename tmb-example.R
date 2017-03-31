## Proportional hazards GSM with random effects
require(TMB)
require(foreign)
require(rstpm2)
src <- "#include <TMB.hpp>  
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(event); // double
  DATA_MATRIX(X);
  DATA_MATRIX(XD);
  DATA_MATRIX(Z);
  DATA_SCALAR(eps);   // boundary value for values that are too small or negative
  DATA_SCALAR(kappa); // scale for the quadratic penalty
  PARAMETER_VECTOR(beta);
  PARAMETER_VECTOR(u);
  PARAMETER(log_sigma);
  Type sigma = exp(log_sigma);
  vector<Type> eta = X*beta + Z*u;
  vector<Type> etaD = XD*beta;
  vector<Type> H = exp(eta);
  vector<Type> h = etaD*H;
  vector<Type> logl(event.size());
  Type pen = 0.0;
  for(int i=0; i<event.size(); ++i) {
    if (h(i)<eps) {
      logl(i) = event(i)*log(eps)-H(i);
      pen += h(i)*h(i)*kappa;
    } else {
      logl(i) = event(i)*log(h(i))-H(i);
    }
  }
  Type f = -sum(logl) + pen;
  for(int i=0; i<u.size(); ++i) {
    f -= dnorm(u(i), Type(0), sigma, true);
  }
  ADREPORT(sigma);
  return f;
}"
write(src, file="phlink.cpp")
compile("phlink.cpp") # slow compilation
dyn.load(dynlib("phlink"))
##
stmixed <- read.dta("http://fmwww.bc.edu/repec/bocode/s/stmixed_example2.dta")
system.time(fit <- {
    init <- stpm2(Surv(stime,event)~treat+factor(trial),data=stmixed,df=3) # initial values
    args <- init@args
    Z <- model.matrix(~factor(trial):treat-1,stmixed)
    f <- MakeADFun(data=list(X=args$X,XD=args$XD,Z=Z,event=as.double(stmixed$event),eps=1e-6,kappa=1.0),
                  parameters=list(beta=coef(init),u=rep(0,ncol(Z)),log_sigma=-4.0),
                  method="nlminb",
                  random="u",                
                  DLL="phlink", silent=TRUE)
    nlminb(f$par,f$fn,f$gr)
})# FAST!
summary(sdreport(f,fit$par))
