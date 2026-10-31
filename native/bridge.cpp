#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <cstring>
#include <cmath>
#include <memory>
#ifdef MISA_PROXQP
#include <proxsuite/proxqp/dense/dense.hpp>
#endif
#ifdef MISA_EIQUADPROG
#include <eiquadprog/eiquadprog-fast.hpp>
#endif
#ifdef MISA_QPOASES
#include <qpOASES.hpp>
#endif
#ifdef MISA_OSQP
#include <osqp.h>
#undef MAX_ITER
#endif
#ifdef MISA_HPIPM
extern "C" {
#include <hpipm_d_dense_qp_dim.h>
#include <hpipm_d_dense_qp.h>
#include <hpipm_d_dense_qp_sol.h>
#include <hpipm_d_dense_qp_ipm.h>
}
#endif
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Clock = std::chrono::steady_clock;
static uint64_t ns(Clock::time_point t) {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();}
struct NativeContext {
 int backend=-1,n=0,me=0,mi=0;
#ifdef MISA_PROXQP
 std::unique_ptr<proxsuite::proxqp::dense::QP<double>> prox;
#endif
#ifdef MISA_QPOASES
 std::unique_ptr<qpOASES::SQProblem> qpo;
 Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> old_h,old_a;
#endif
#ifdef MISA_OSQP
 OSQPWorkspace* osqp=nullptr;
 ~NativeContext(){if(osqp)osqp_cleanup(osqp);}
#endif
};
extern "C" void misa_native_free(uintptr_t context) noexcept {delete reinterpret_cast<NativeContext*>(context);}
// All matrices cross the ABI in column-major layout. Multipliers use
// H*x+g+Aeq^T*y+Aiq^T*z=0, z>=0 for Aiq*x<=biq.
extern "C" int misa_native_qp(int backend,int n,int me,int mi,const double* hp,const double* gp,const double* aep,const double* bep,const double* aip,const double* bip,int max_iter,double ftol,double otol,int warm,double* xp,double* yp,double* zp,int* iterations,int* raw,uint64_t* times,uintptr_t* handle) noexcept {
  try {
    NativeContext local;NativeContext* ctx=&local;
    if(handle){ctx=reinterpret_cast<NativeContext*>(*handle);if(ctx && (ctx->backend!=backend || ctx->n!=n || ctx->me!=me || ctx->mi!=mi)){delete ctx;ctx=nullptr;*handle=0;}if(!ctx){ctx=new NativeContext;ctx->backend=backend;ctx->n=n;ctx->me=me;ctx->mi=mi;*handle=reinterpret_cast<uintptr_t>(ctx);}}
    auto t=Clock::now();
    Mat H=Eigen::Map<const Mat>(hp,n,n), A=Eigen::Map<const Mat>(aep,me,n), C=Eigen::Map<const Mat>(aip,mi,n);
    Vec g=Eigen::Map<const Vec>(gp,n), b=Eigen::Map<const Vec>(bep,me), u=Eigen::Map<const Vec>(bip,mi);
    Eigen::Map<Vec> x(xp,n), y(yp,me), z(zp,mi);
    times[0]=ns(t); t=Clock::now();
#ifdef MISA_PROXQP
    if (backend==0) {
      bool reused=bool(ctx->prox);
      if(!reused)ctx->prox=std::make_unique<proxsuite::proxqp::dense::QP<double>>(n,me,mi);
      auto &qp=*ctx->prox;times[4]=(reused||warm)?1:0;
      qp.settings.eps_abs=std::min(ftol,otol); qp.settings.eps_rel=0; qp.settings.check_duality_gap=true;qp.settings.eps_duality_gap_abs=otol;qp.settings.eps_duality_gap_rel=0; qp.settings.max_iter=max_iter; qp.settings.verbose=false;
      Vec lower=Vec::Constant(mi,-std::numeric_limits<double>::infinity());
      if(reused){qp.settings.initial_guess=proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;qp.update(H,g,A,b,C,lower,u,true);times[2]=ns(t);}else{qp.init(H,g,A,b,C,lower,u);times[1]=ns(t);}
      t=Clock::now();
      if(warm && !reused) qp.solve(Vec(x),Vec(y),Vec(z)); else qp.solve();
      times[3]=ns(t); x=qp.results.x; y=qp.results.y; z=qp.results.z;
      *iterations=qp.results.info.iter; *raw=int(qp.results.info.status);
      using namespace proxsuite::proxqp;
      return qp.results.info.status==QPSolverOutput::PROXQP_SOLVED?0:qp.results.info.status==QPSolverOutput::PROXQP_MAX_ITER_REACHED?1:qp.results.info.status==QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE?2:3;
    }
#endif
#ifdef MISA_EIQUADPROG
    if(backend==1) {
      eiquadprog::solvers::EiquadprogFast qp; qp.reset(n,me,mi); qp.setMaxIter(max_iter);
      Mat ce=-A, ci=-C; Vec result(n); times[1]=ns(t); t=Clock::now();
      auto status=qp.solve_quadprog(H,g,ce,b,ci,u,result); times[3]=ns(t);
      x=result; y.setZero(); z.setZero(); *raw=int(status); *iterations=qp.getIteratios();
      auto active=qp.getActiveSet(); auto dual=qp.getLagrangeMultipliers();
      for(size_t k=0;k<qp.getActiveSetSize();++k) { int j=active[k]; if(j<0 && -j-1<me) y[-j-1]=dual[k]; else if(j>=0 && j<mi) z[j]=dual[k]; }
      return *raw==0?0:*raw==3?1:*raw==1?2:3;
    }
#endif
#ifdef MISA_QPOASES
    if(backend==2) {
      using namespace qpOASES;
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> hr=H, ar(me+mi,n); ar.topRows(me)=A; ar.bottomRows(mi)=C;
      Vec lo(me+mi),up(me+mi); lo.head(me)=b;lo.tail(mi).setConstant(-INFTY); up.head(me)=b;up.tail(mi)=u;
      bool reused=bool(ctx->qpo);if(!reused)ctx->qpo=std::make_unique<SQProblem>(n,me+mi);auto &qp=*ctx->qpo;times[4]=reused?1:0; Options opt;opt.setToReliable();opt.printLevel=PL_NONE;opt.terminationTolerance=std::min(ftol,otol);opt.boundTolerance=ftol;qp.setOptions(opt);
      int_t nw=max_iter;times[1]=ns(t);t=Clock::now();
      auto status=reused?qp.hotstart(hr.data(),g.data(),ar.data(),nullptr,nullptr,lo.data(),up.data(),nw):qp.init(hr.data(),g.data(),ar.data(),nullptr,nullptr,lo.data(),up.data(),nw); times[3]=ns(t);
      ctx->old_h=std::move(hr);ctx->old_a=std::move(ar);
      qp.getPrimalSolution(xp); Vec dual(n+me+mi);qp.getDualSolution(dual.data()); y=-dual.segment(n,me);z=-dual.tail(mi);
      *raw=int(status);*iterations=nw;
      return status==SUCCESSFUL_RETURN?0:status==RET_MAX_NWSR_REACHED?1:qp.isInfeasible()?2:3;
    }
#endif
#ifdef MISA_OSQP
    if(backend==3) {
      int m=me+mi; std::vector<c_int> pp(n+1),pi,ap(n+1),ai; std::vector<c_float> pv,av;
      for(int j=0;j<n;++j) { pp[j]=pi.size();for(int i=0;i<=j;++i) {pi.push_back(i);pv.push_back(H(i,j));} ap[j]=ai.size();for(int i=0;i<m;++i) {double v=i<me?A(i,j):C(i-me,j);{ai.push_back(i);av.push_back(v);}} }pp[n]=pi.size();ap[n]=ai.size();
      csc P={c_int(pv.size()),n,n,pp.data(),pi.data(),pv.data(),-1}, AA={c_int(av.size()),m,n,ap.data(),ai.data(),av.data(),-1};
      Vec lo(m),up(m);lo.head(me)=b;lo.tail(mi).setConstant(-OSQP_INFTY);up.head(me)=b;up.tail(mi)=u;
      OSQPData data;data.n=n;data.m=m;data.P=&P;data.A=&AA;data.q=g.data();data.l=lo.data();data.u=up.data();OSQPSettings settings;osqp_set_default_settings(&settings);settings.verbose=0;settings.eps_abs=std::min(ftol,otol);settings.eps_rel=0;settings.max_iter=max_iter;settings.polish=1;
      bool reused=ctx->osqp!=nullptr;times[4]=(reused||warm)?1:0;OSQPWorkspace* qp=ctx->osqp;
      if(reused){*raw=osqp_update_P_A(qp,pv.data(),nullptr,pv.size(),av.data(),nullptr,av.size());*raw|=osqp_update_lin_cost(qp,g.data());*raw|=osqp_update_bounds(qp,lo.data(),up.data());osqp_update_eps_abs(qp,std::min(ftol,otol));osqp_update_max_iter(qp,max_iter);times[2]=ns(t);}
      else{*raw=osqp_setup(&qp,&data,&settings);ctx->osqp=qp;times[1]=ns(t);}
      if(*raw)return 3;
      if(warm && !reused){Vec dual(m);dual.head(me)=y;dual.tail(mi)=z;osqp_warm_start(qp,xp,dual.data());}
      t=Clock::now();osqp_solve(qp);times[3]=ns(t);
      x=Eigen::Map<Vec>(qp->solution->x,n);y=Eigen::Map<Vec>(qp->solution->y,me);z=Eigen::Map<Vec>(qp->solution->y+me,mi);*raw=qp->info->status_val;*iterations=qp->info->iter;
      return *raw==OSQP_SOLVED?0:*raw==OSQP_MAX_ITER_REACHED?1:(*raw==OSQP_PRIMAL_INFEASIBLE || *raw==OSQP_DUAL_INFEASIBLE)?2:3;
    }
#endif
#ifdef MISA_HPIPM
    if(backend==4) {
      d_dense_qp_dim dim; std::vector<char> dm(d_dense_qp_dim_memsize());d_dense_qp_dim_create(&dim,dm.data());d_dense_qp_dim_set_all(n,me,0,mi,0,&dim);
      d_dense_qp qp;std::vector<char> qm(d_dense_qp_memsize(&dim));d_dense_qp_create(&dim,&qp,qm.data());
      d_dense_qp_set_H(H.data(),&qp);d_dense_qp_set_g(g.data(),&qp);d_dense_qp_set_A(A.data(),&qp);d_dense_qp_set_b(b.data(),&qp);d_dense_qp_set_C(C.data(),&qp);Vec lower=Vec::Constant(mi,-1e30),mask=Vec::Zero(mi);d_dense_qp_set_lg(lower.data(),&qp);d_dense_qp_set_ug(u.data(),&qp);d_dense_qp_set_lg_mask(mask.data(),&qp);
      d_dense_qp_sol sol;std::vector<char> sm(d_dense_qp_sol_memsize(&dim));d_dense_qp_sol_create(&dim,&sol,sm.data());
      d_dense_qp_ipm_arg arg;std::vector<char> am(d_dense_qp_ipm_arg_memsize(&dim));d_dense_qp_ipm_arg_create(&dim,&arg,am.data());d_dense_qp_ipm_arg_set_default(ROBUST,&arg);d_dense_qp_ipm_arg_set_iter_max(&max_iter,&arg);d_dense_qp_ipm_arg_set_tol_stat(&otol,&arg);d_dense_qp_ipm_arg_set_tol_eq(&ftol,&arg);d_dense_qp_ipm_arg_set_tol_ineq(&ftol,&arg);d_dense_qp_ipm_arg_set_tol_comp(&otol,&arg);
      d_dense_qp_ipm_ws ws;std::vector<char> wm(d_dense_qp_ipm_ws_memsize(&dim,&arg));d_dense_qp_ipm_ws_create(&dim,&arg,&ws,wm.data());times[1]=ns(t);t=Clock::now();d_dense_qp_ipm_solve(&qp,&sol,&arg,&ws);times[3]=ns(t);
      d_dense_qp_sol_get_v(&sol,xp);d_dense_qp_sol_get_pi(&sol,yp);y=-y;d_dense_qp_sol_get_lam_ug(&sol,zp);d_dense_qp_ipm_get_status(&ws,raw);d_dense_qp_ipm_get_iter(&ws,iterations);return *raw==0?0:*raw==1?1:3;
    }
#endif
    *raw=-999;return 3;
  } catch(...) { *raw=-998;return 3; }
}
#ifdef MISA_HPIPM
extern "C" {
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>
}
struct MisaOcpStage {const double *a,*b,*bias,*qmat,*rmat,*smat,*q,*r,*c,*d,*lower,*upper;int ng;};
extern "C" int misa_native_ocp(int N,int nx,int nu,const double* initial,const MisaOcpStage* stages,int max_iter,double ftol,double otol,double* controls,double* multipliers,int* iterations,int* raw,uint64_t* times) noexcept {
 try {
  auto t=Clock::now();
  d_ocp_qp_dim dim;std::vector<char> dm(d_ocp_qp_dim_memsize(N));d_ocp_qp_dim_create(N,&dim,dm.data());
  for(int k=0;k<=N;++k){d_ocp_qp_dim_set_nx(k,nx,&dim);d_ocp_qp_dim_set_nu(k,k<N?nu:0,&dim);d_ocp_qp_dim_set_ng(k,k<N?stages[k].ng:0,&dim);d_ocp_qp_dim_set_nbx(k,k==0?nx:0,&dim);}
  d_ocp_qp qp;std::vector<char> qm(d_ocp_qp_memsize(&dim));d_ocp_qp_create(&dim,&qp,qm.data());d_ocp_qp_set_all_zero(&qp);
  std::vector<int> idx(nx);for(int j=0;j<nx;++j)idx[j]=j;
  d_ocp_qp_set_idxbx(0,idx.data(),&qp);d_ocp_qp_set_lbx(0,const_cast<double*>(initial),&qp);d_ocp_qp_set_ubx(0,const_cast<double*>(initial),&qp);
  for(int k=0;k<N;++k){auto &s=stages[k];
#define SET(name,field) d_ocp_qp_set_##name(k,const_cast<double*>(s.field),&qp)
   SET(A,a);SET(B,b);SET(b,bias);SET(Q,qmat);SET(R,rmat);SET(S,smat);SET(q,q);SET(r,r);SET(C,c);SET(D,d);
#undef SET
   Vec lower(s.ng),upper(s.ng),lm=Vec::Ones(s.ng),um=Vec::Ones(s.ng);
   for(int j=0;j<s.ng;++j){lower[j]=std::isfinite(s.lower[j])?s.lower[j]:-1e30;upper[j]=std::isfinite(s.upper[j])?s.upper[j]:1e30;if(!std::isfinite(s.lower[j]))lm[j]=0;if(!std::isfinite(s.upper[j]))um[j]=0;}
   d_ocp_qp_set_lg(k,lower.data(),&qp);d_ocp_qp_set_ug(k,upper.data(),&qp);d_ocp_qp_set_lg_mask(k,lm.data(),&qp);d_ocp_qp_set_ug_mask(k,um.data(),&qp);
  }
  d_ocp_qp_sol sol;std::vector<char> sm(d_ocp_qp_sol_memsize(&dim));d_ocp_qp_sol_create(&dim,&sol,sm.data());
  d_ocp_qp_ipm_arg arg;std::vector<char> am(d_ocp_qp_ipm_arg_memsize(&dim));d_ocp_qp_ipm_arg_create(&dim,&arg,am.data());d_ocp_qp_ipm_arg_set_default(ROBUST,&arg);int ric=0;d_ocp_qp_ipm_arg_set_ric_alg(&ric,&arg);d_ocp_qp_ipm_arg_set_iter_max(&max_iter,&arg);d_ocp_qp_ipm_arg_set_tol_stat(&otol,&arg);d_ocp_qp_ipm_arg_set_tol_eq(&ftol,&arg);d_ocp_qp_ipm_arg_set_tol_ineq(&ftol,&arg);d_ocp_qp_ipm_arg_set_tol_comp(&otol,&arg);
  d_ocp_qp_ipm_ws ws;std::vector<char> wm(d_ocp_qp_ipm_ws_memsize(&dim,&arg));d_ocp_qp_ipm_ws_create(&dim,&arg,&ws,wm.data());times[1]=ns(t);t=Clock::now();d_ocp_qp_ipm_solve(&qp,&sol,&arg,&ws);times[3]=ns(t);
  int offset=0;for(int k=0;k<N;++k){d_ocp_qp_sol_get_u(k,&sol,controls+k*nu);Vec lo(stages[k].ng),up(stages[k].ng);d_ocp_qp_sol_get_lam_lg(k,&sol,lo.data());d_ocp_qp_sol_get_lam_ug(k,&sol,up.data());Eigen::Map<Vec>(multipliers+offset,stages[k].ng)=up-lo;offset+=stages[k].ng;}
  d_ocp_qp_ipm_get_status(&ws,raw);d_ocp_qp_ipm_get_iter(&ws,iterations);return *raw==0?0:*raw==1?1:3;
 }catch(...){*raw=-998;return 3;}
}
#endif
