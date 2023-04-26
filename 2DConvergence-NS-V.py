'''
Convergence test for a mixed scheme of the Navier--Stokes with variable viscosity equations
The domain is (0,1)x(0,1)

Manufactured smooth solutions
#######################################
strong primal form: 

-div(mu(|grad(u)|)grad(u)) + grad(u)*u + grad(p) = f  in Omega
                                          div(u) = 0  in Omega 

Pure Dirichlet conditions for u 
                                                u = g on Gamma

Lagrange multiplier to fix the average of p
                                            int(p) = 0

######################################

strong mixed form in terms of (t,sigma,u)

                          t = grad(u) in Omega
-mu(|t|)*t - (u otimes u)^d = div(sigma)^d in Omega
                 div(sigma) = f in Omega
                          u = g on Gamma
+ trace of pressure:
 int(tr(sigma+u otimes u)) = 0

'''

from fenics import *
import numpy as np
from dolfin import *

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

fileO = XDMFFile("outputs/2DConvergence-NS-V.xdmf")
fileO.parameters['rewrite_function_mesh']=True
fileO.parameters["functions_share_mesh"] = True

import sympy2fenics as sf
def str2exp(s):
    return sf.sympy2exp(sf.str2sympy(s))
# Constant coefficients 
ndim = 2
Id = Identity(ndim)

# variable viscosity 
'''
def muu(t):
    print(t.get_local_vector())
    norm_tt = 0
    for tt in t :
        for ss in t :
            norm_tt = norm_tt + tt*ss
    norm_tt = pow(norm_tt,0.5)
    print(norm_tt)
    return 2. + norm_tt
'''    

def mu(t):
    return conditional(inner(t, t) <= 1e-14, 2., 2. + pow(1. + pow(inner(t, t), 0.5), -1.))

# Manufactured solutions as strings# 

u_str = '(sin(pi*x)*cos(pi*y),-cos(pi*x)*sin(pi*y))'
p_str = 'x**2 - y**2'

# Initializing vectors for error history 

nkmax = 8; # max refinements

dof = [];
hh = []; dof = []; 
rt = []; ru = []; rsig = [];
et = []; eu = []; esig = [];
rp = []; ep = [];


rt.append(0.0); ru.append(0.0); rsig.append(0.0); rp.append(0.0)

# polynomial degree

k = 0

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1) 
    mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh)
        
    hh.append(mesh.hmax())
    # ********* Finite dimensional spaces ********* #
    Qu = VectorElement('DG', mesh.ufl_cell(), k)
    Htt= VectorElement('DG', mesh.ufl_cell(), k, dim=3)   
    Hs = FiniteElement('RT', mesh.ufl_cell(), k+1) # in FEniCS RTk is understood as RT{k+1} - is a vector-valued space!
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)

    Vh = FunctionSpace(mesh, MixedElement([Qu,Htt,Hs,Hs,R0]))
    dof.append(Vh.dim())
    
    Trial = TrialFunction(Vh)
    Sol   = Function(Vh)


    #Qu, Htt, Hsig, Hsig,  R  
    v,    s_, taux, tauy,  zeta =  TestFunctions(Vh)
    u,    t_, sigx, sigy,  theta = split(Sol)

    tt = as_tensor(((t_[0], t_[1]),(t_[2],-t_[0])))
    ss = as_tensor(((s_[0], s_[1]),(s_[2],-s_[0])))
    tau = as_tensor((taux,tauy))
    sig = as_tensor((sigx,sigy))
    
    # muu(t_)
    #


    # ********* instantiation of exact solutions ****** #
    u_ex   = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex   = Expression(str2exp(p_str), degree=6, domain=mesh)
    
    # Instantiation of exact solutions
    
    tt_ex  = grad(u_ex)
    #c= -(0.5*inner(tr(outer(u_ex,u_ex)),1.)*dx)*Id
    sig_ex =  mu(tt_ex)*tt_ex - outer(u_ex,u_ex) - p_ex*Id #-c*Id


    # source and forcing terms
    
    ff = -div(sig_ex)
    gg = u_ex

    # ********* boundary conditions ******** #

    # all imposed naturally 
    

    # ********* Variational forms ********* #
    #sigd = sig -1./ndim*tr(sig)*Id 
    #taud =tau -1./ndim*tr(tau)*Id 
    #uud = outer(u,u) -1./ndim*tr(outer(u,u))*Id


    A_st= inner(mu(tt)*tt,ss)*dx #
    B1_ssig = inner(dev(sig),ss)*dx 
    c_uu = inner(dev(outer(u,u)), ss)*dx 
    B1_ttau = inner(dev(tau),tt)*dx
    B_tauu =  dot(u,div(tau))*dx
    B_sigv =  dot(v,div(sig))*dx  
    

    GG = dot(tau*n,gg)*ds
    FF = dot(ff,v)*dx


    # ---- Lagrange multiplier to impose trace of sigma --- # 
    ZZ = tr(sig-sig_ex)*zeta*dx + (tr(outer(u,u)-outer(u_ex,u_ex)))*zeta*dx \
        + tr(tau) * theta * dx\

    Nonl = A_st - B1_ssig - c_uu \
                - B1_ttau - B_tauu + GG \
                          - B_sigv - FF \
            + ZZ                        
    
    # Solver specifications (including essential BCs if any)

    Tangent = derivative(Nonl, Sol, Trial)
    Problem = NonlinearVariationalProblem(Nonl, Sol, J=Tangent)
    Solver  = NonlinearVariationalSolver(Problem)
    Solver.parameters['nonlinear_solver']                    = 'newton'
    Solver.parameters['newton_solver']['linear_solver']      = 'umfpack'
    Solver.parameters['newton_solver']['absolute_tolerance'] = 1e-6
    Solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
    Solver.parameters['newton_solver']['maximum_iterations'] = 25

    # Assembling and solving
    #solve(Nonl == 0, Sol)

    Solver.solve()

    uh,    th_, sighx, sighy,  thetah= Sol.split()
    
    th = as_tensor(((th_[0], th_[1]),(th_[2],-th_[0])))
    sigh = as_tensor((sighx,sighy))
    #gammah = as_tensor(((0,gamh_),(-gamh_,0)))

    Ph = FunctionSpace(mesh, 'DG', k)
    Th = TensorFunctionSpace(mesh, 'DG', k)
    # Postprocessing (eq 5.28)
    ph = project(-1/ndim*tr(sigh + outer(uh,uh)),Ph) 

    sig_v = project(sigh, Th)
    t_v = project(th, Th)
   
    # saving to file

    uh.rename("u","u"); fileO.write(uh,nk*1.0)
    t_v.rename("t","t"); fileO.write(t_v,nk*1.0)
    sig_v.rename("sig","sig"); fileO.write(sig_v,nk*1.0)
    ph.rename("p","p"); fileO.write(ph,nk*1.0)
   
    # Error computation (uh,    th_, sighx, sighy,  p)

    E_u = assemble(dot(u_ex-uh,u_ex-uh)**4*dx)  # (|| u - uh ||_L4)**4
    E_tt = assemble(inner(tt_ex - th,tt_ex-th)*dx) # (|| t - th ||_L2)**2
    E_sigma_0 = assemble(inner(sig_ex-sigh,sig_ex-sigh)*dx) # (||sig0 -sig0h||_L2)**2
    E_sigma_div = assemble(dot(div(sig_ex-sigh),div(sig_ex-sigh))**(4./3.)*dx) #(||div(sig0 -sig0h)||_0,4/3)**2
    E_p = assemble((p_ex - ph)**2*dx) # ||ph-p||_L2


#dof = [];
#hh = []; dof = []; 
#rt = []; ru = []; rsig = [];
#et = []; eu = []; esig = [];
#rp = []; ep = [];


    et.append(pow(E_tt,0.5))
    eu.append(pow(E_u,0.25))
    esig.append(pow(E_sigma_0,0.5)+pow(E_sigma_div,0.75))
    ep.append(pow(E_p,0.5))

    # Computing convergence rates
    
    if(nk>0):
        rt.append(ln(et[nk]/et[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsig.append(ln(esig[nk]/esig[nk-1])/ln(hh[nk]/hh[nk-1]))
        ru.append(ln(eu[nk]/eu[nk-1])/ln(hh[nk]/hh[nk-1]))
        rp.append(ln(ep[nk]/ep[nk-1])/ln(hh[nk]/hh[nk-1]))
    


# Generating error history 
print('=====================================================================================')
print('   dof &  hh   &  e(t)  & r(t) & e(sig) & r(sig) & e(u) & r(u)   & e(p)   & r(p)  ')
print('=====================================================================================')

for nk in range(nkmax):
    print('{:6d}  {:.4f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}   {:1.2e}  {:.2f} '.format(dof[nk], hh[nk], et[nk], rt[nk], esig[nk], rsig[nk], eu[nk], ru[nk], ep[nk], rp[nk]))
print('======================================================================================')


