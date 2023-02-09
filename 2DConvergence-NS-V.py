'''
Convergence test for a mixed scheme of the Navier--Stokes with variable viscosity equations
The domain is (0,1)x(0,1)

Manufactured smooth solutions
#######################################
strong primal form: 

-div(mu|grad(u)|grad(u)) + grad(u)*u + grad(p) = f  in Omega
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
#import numpy


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

def mu(s):
        ss = det(s)
        muu = 2 + 1/(1+ss)
        return muu

#lam = Constant(0.2)



# Manufactured solutions as strings 

u_str = '(-cos(pi*x)*sin(pi*y),sin(pi*x)*cos(pi*y))'
p_str = 'x**2 - y**2'

# Initialising vectors for error history 

nkmax = 6; # max refinements
l = 0 # polynomial degree
dof = [];
hh = []; dof = []; 
rt = []; ru = []; rsig = [];
et = []; eu = []; esig = [];
rp = [];


rt.append(0.0); ru.append(0.0); rsig.append(0.0); rp.append(0.0)

# polynomial degree

k = 1

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1) 
    mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh)
        
    hh.append(mesh.hmax())
    # ********* Finite dimensional spaces ********* #
    Hu = VectorElement('DG', mesh.ufl_cell(), k)
    Htt= VectorElement('DG', mesh.ufl_cell(), k, dim=3)   
    Hs = FiniteElement('RT', mesh.ufl_cell(), k+1) # in FEniCS RTk is understood as RT{k+1} - is a vector-valued space!
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)

    Vh = FunctionSpace(mesh, MixedElement([Hu,Htt,Hs,Hs,R0]))
    dof.append(Vh.dim())
    
    Trial = TrialFunction(Vh)
    Sol   = Function(Vh)


    #Hu, Htt, Hsig, Hsig,  R  
    v,    s_, taux, tauy,  zeta = TestFunctions(Vh)
    u,    t_, sigx, sigy,  theta = split(Sol)

    tt = as_tensor(((t_[0], t_[1]),(t_[2],-t_[0])))
    ss = as_tensor(((s_[0], s_[1]),(s_[2],-s_[0])))
    tau = as_tensor((taux,tauy))
    sig = as_tensor((sigx,sigy))


    # ********* instantiation of exact solutions ****** #
    u_ex   = Expression(str2exp(u_str), degree=6, domain=mesh)
    p_ex   = Expression(str2exp(p_str), degree=6, domain=mesh)
    
    # Instantiation of exact solutions
    
    
    tt_ex  = grad(u_ex)
    sig_ex = mu(tt_ex)*tt_ex - outer(u_ex,u_ex) - p_ex*Id


    # source and forcing terms
    
    ff = div(sig_ex)
    gg = u_ex

    # ********* boundary conditions ******** #

    # all imposed naturally 
    

    # ********* Variational forms ********* #
    
    A_st= inner(mu(tt)*tt,ss)*dx #
    B1_ssig = -inner(sig,ss)*dx
    B1_ttau = -inner(tau,tt)*dx
    B_tauu = - dot(u,div(tau))*dx
    B_sigv = - dot(v,div(sig))*dx
    uud = outer(u,u) -1./ndim*tr(outer(u,u))*Id     
    c_uu = inner(uud, ss)*dx 

    GG = dot(tau*n,gg)*ds
    FF = dot(ff,v)*dx


    # ---- Lagrange multiplier to impose trace of sigma --- # 
    ZZ = (tr(sig + outer(u,u))- tr(sig_ex + outer(u_ex,u_ex))) * zeta * dx \
        + tr(tau) * theta * dx

    Nonl  =  A_st   + B1_ssig           + c_uu \
            +B1_ttau           +B_tauu         - GG\
                    + B_sigv                   - FF \
            +ZZ                        
    
    # Solver specifications (including essential BCs if any)

    Tangent = derivative(Nonl, Sol, Trial)
    Problem = NonlinearVariationalProblem(Nonl, Sol, J=Tangent)
    Solver  = NonlinearVariationalSolver(Problem)
    Solver.parameters['nonlinear_solver']                    = 'newton'
    Solver.parameters['newton_solver']['linear_solver']      = 'umfpack'
    Solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    Solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    Solver.parameters['newton_solver']['maximum_iterations'] = 300

    # Assembling and solving
    #solve(Nonl == 0, Sol)

    Solver.solve()
    '''
    th_, sigh1, sigh2,uh,gamh_,xih = Sol.split()
    
    th = as_tensor(((th_[0], th_[1]),(th_[2],-th_[0])))
    sigmah = as_tensor((sigh1,sigh2))
    gammah = as_tensor(((0,gamh_),(-gamh_,0)))

    Ph = FunctionSpace(mesh, 'DG', l)
    Th = TensorFunctionSpace(mesh, 'DG', l)
    # Postprocessing (eq 2.7)
    ph = project(-1/ndim*tr(sigmah + outer(uh,uh)),Ph) 

    sig_v = project(sigmah, Th)
    t_v = project(th, Th)
    
    # saving to file

    #uh.rename("u","u"); fileO.write(uh,nk*1.0)
    #t_v.rename("t","t"); fileO.write(t_v,nk*1.0)
    #sig_v.rename("sig","sig"); fileO.write(sig_v,nk*1.0)
    #ph.rename("p","p"); fileO.write(ph,nk*1.0)
   
    # Error computation

    E_t = assemble(inner(t_ex - th,t_ex-th)*dx)
    E_sigma_0 = assemble(inner(sigma_ex-sigmah,sigma_ex-sigmah)*dx)
    E_sigma_div = assemble(dot(div(sigma_ex-sigmah),div(sigma_ex-sigmah))**(2./3.)*dx)
    E_u = assemble(dot(u_ex-uh,u_ex-uh)**2*dx)
    E_gamma = assemble(inner(gamma_ex-gammah,gamma_ex-gammah)*dx)
    E_p = assemble((p_ex - ph)**2*dx)

    errt.append(pow(E_t,0.5))
    errsigma.append(pow(E_sigma_0,0.5)+pow(E_sigma_div,0.75))
    erru.append(pow(E_u,0.25))
    errgamma.append(pow(E_gamma,0.5))
    errp.append(pow(E_p,0.5))

    # Computing convergence rates
    
    if(nk>0):
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rategamma.append(ln(errgamma[nk]/errgamma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        '''
       

'''
# Generating error history 
print('==============================================================================================================')
print('   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  ')
print('==============================================================================================================')

for nk in range(nkmax):
    print('{:6d}  {:.4f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f}  {:1.2e}  {:.2f} '.format(nvec[nk], hvec[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk], erru[nk], rateu[nk], errgamma[nk], rategamma[nk], errp[nk], ratep[nk]))
print('==============================================================================================================')

'''
'''

   305 & 0.5000 & 6.72e-01 & 0.00 & 2.61e+00 & 0.00 & 2.96e-01 & 0.00 & 8.28e-01 & 0.00 & 1.68e-01 & 0.00 
  1185 & 0.2500 & 3.10e-01 & 1.12 & 1.30e+00 & 1.00 & 1.61e-01 & 0.87 & 4.16e-01 & 0.99 & 7.24e-02 & 1.22 
  4673 & 0.1250 & 1.53e-01 & 1.02 & 6.45e-01 & 1.01 & 8.13e-02 & 0.99 & 2.09e-01 & 1.00 & 2.81e-02 & 1.36 
 18561 & 0.0625 & 7.65e-02 & 1.00 & 3.22e-01 & 1.00 & 4.07e-02 & 1.00 & 1.04e-01 & 1.00 & 1.27e-02 & 1.15 
 73985 & 0.0312 & 3.82e-02 & 1.00 & 1.61e-01 & 1.00 & 2.04e-02 & 1.00 & 5.22e-02 & 1.00 & 6.17e-03 & 1.04 
295425 & 0.0156 & 1.91e-02 & 1.00 & 8.05e-02 & 1.00 & 1.02e-02 & 1.00 & 2.61e-02 & 1.00 & 3.06e-03 & 1.01 

   697 & 0.5000 & 1.27e-01 & 0.00 & 5.59e-01 & 0.00 & 6.38e-02 & 0.00 & 1.60e-01 & 0.00 & 2.59e-02 & 0.00 
  2737 & 0.2500 & 3.24e-02 & 1.98 & 1.35e-01 & 2.05 & 1.77e-02 & 1.85 & 4.13e-02 & 1.96 & 7.39e-03 & 1.81 
 10849 & 0.1250 & 7.93e-03 & 2.03 & 3.35e-02 & 2.01 & 4.46e-03 & 1.99 & 1.04e-02 & 1.99 & 1.77e-03 & 2.06 
 43201 & 0.0625 & 1.95e-03 & 2.02 & 8.32e-03 & 2.01 & 1.12e-03 & 2.00 & 2.60e-03 & 2.00 & 3.96e-04 & 2.16 
172417 & 0.0312 & 4.83e-04 & 2.01 & 2.07e-03 & 2.01 & 2.80e-04 & 2.00 & 6.51e-04 & 2.00 & 8.99e-05 & 2.14 
688897 & 0.0156 & 1.20e-04 & 2.01 & 5.17e-04 & 2.00 & 6.99e-05 & 2.00 & 1.63e-04 & 2.00 & 2.11e-05 & 2.09 


'''
