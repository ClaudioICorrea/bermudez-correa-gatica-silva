'''
Convergence test for a mixed scheme of the Navier-Stokes- with variable viscosity equations
The domain is (0,1)x(0,0.5)x(0,0.5)

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
 int(tr(sigma+ u otimes u)) = 0

'''

from fenics import *
import sympy2fenics as sf

str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 6

fileO = XDMFFile("outputs/3DConvergence-NS-V.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

# ****** Constant coefficients ****** #
ndim = 3
Id = Identity(ndim)
lam = Constant(0.2)

# variable viscosity 
mu  = lambda t: 2. + pow((1.+np.linalg.norm(t, ord=2)),-1)

# ******* Exact solutions for error analysis ****** #

u_str = '(sin(pi*x)*cos(pi*y)*cos(pi*z), -2*cos(pi*x)*sin(pi*y)*cos(pi*z), cos(pi*x)*cos(pi*y)*sin(pi*z))'
p_str = 'sin(x*y*z)'


# ****** Constant coefficients ****** #

nkmax = 4; k = 0

dof = [];
hh = []; dof = []; 
rt = []; ru = []; rsig = [];
et = []; eu = []; esig = [];
rp = []; ep = [];

rt.append(0.0); ru.append(0.0); rsig.append(0.0); rp.append(0.0)


# ***** Error analysis ***** #

for nk in range(nkmax):
    print("....... Refinement level : nk = ", nk)
    nps = pow(2,nk+1)
    mesh = UnitCubeMesh(nps,nps,nps)
    n   = FacetNormal(mesh)
        
    hh.append(mesh.hmax())
    # ********* Finite dimensional spaces ******
    Qu = VectorElement('DG', mesh.ufl_cell(), k)
    Htt= VectorElement('DG', mesh.ufl_cell(), k, dim=8)   
    Hs = FiniteElement('RT', mesh.ufl_cell(), k+1) # in FEniCS RTk is understood as RT{k+1} - is a vector-valued space!
    R0 = FiniteElement('R', mesh.ufl_cell(), 0)

    Vh = FunctionSpace(mesh, MixedElement([Qu,Htt,Hs,Hs,Hs,R0]))
    dof.append(Vh.dim())
    
    Trial = TrialFunction(Vh)
    Sol   = Function(Vh)

    #Qu, Htt, Hsig, Hsig, Hsig  R  
    v,    s_, taux, tauy, tauz, zeta  =  TestFunctions(Vh)
    u,    t_, sigx, sigy, sigz, theta = split(Sol)


    tt=as_tensor(((t_[0],t_[1],t_[2]),(t_[3],t_[4],t_[5]),(t_[6],t_[7],-t_[0]-t_[4])))
    ss=as_tensor(((s_[0],s_[1],s_[2]),(s_[3],s_[4],s_[5]),(s_[6],s_[7],-s_[0]-s_[4])))


    sig = as_tensor((sigx,sigy,sigz)) 
    tau   = as_tensor((taux,tauy,tauz)) 

    
    # instantiation of exact solutions
    
    u_ex  = Expression(str2exp(u_str), degree=k+4, domain=mesh)
    p_ex  = Expression(str2exp(p_str), degree=k+4, domain=mesh)
    
    tt_ex  = grad(u_ex)
    sig_ex =  mu(tt_ex)*tt_ex - outer(u_ex,u_ex) - p_ex*Id
    
    # source and forcing terms

    ff = -div(sig_ex)
    gg = u_ex
    


    # ********* Variational forms ********* #


    A_st= inner(mu(tt)*tt,ss)*dx #
    B1_ssig = -inner(dev(sig),ss)*dx 
    c_uu = inner(dev(outer(u,u)), ss)*dx 
    B1_ttau = inner(dev(tau),tt)*dx
    B_tauu =  dot(u,div(tau))*dx
    B_sigv =  -dot(v,div(sig))*dx  
    

    GG = dot(tau*n,gg)*ds
    FF = dot(ff,v)*dx


    # ---- Lagrange multiplier to impose trace of sigma --- # 
    ZZ = tr(sig-sig_ex)*zeta*dx + (tr(outer(u,u)-outer(u_ex,u_ex)))*zeta*dx \
        + tr(tau) * theta * dx\

    Nonl = A_st + B1_ssig - c_uu \
            + B1_ttau + B_tauu - GG \
            + B_sigv - FF \
            + ZZ   
    

    
    Tang = derivative(Nonl, Sol, Trial)
    Problem = NonlinearVariationalProblem(Nonl, Sol, J=Tang) # In this case, no need for essential BCs
    solver  = NonlinearVariationalSolver(Problem)
    solver.parameters['nonlinear_solver']                    = 'newton'
    solver.parameters['newton_solver']['linear_solver']      = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 1e-8
    solver.parameters['newton_solver']['relative_tolerance'] = 1e-8
    solver.parameters['newton_solver']['maximum_iterations'] = 25
    
    solver.solve()
    '''
    th_, sigh1, sigh2, sigh3, bsolh, uh, gamh_, xih = Usol.split()

    th=as_tensor(((th_[0],th_[1],th_[2]),(th_[3],th_[4],th_[5]),(th_[6],th_[7],-th_[0]-th_[4])))
    sigmah = as_tensor((sigh1,sigh2,sigh3)) + curlTen(bsolh) 
    gammah=as_tensor(((   0.0, gamh_[0], gamh_[1]),
                     (-gamh_[0],   0.0, gamh_[2]),
                     (-gamh_[1],-gamh_[2],  0.0)))
    
    # dimension-dependent
    ph = project(-1./ndim*tr(sigmah+outer(uh,uh)),Ph)
    Th = TensorFunctionSpace(mesh, 'DG', l)
    sig_v = project(sigmah, Th)
    t_v = project(th, Th)

    # saving to file

    uh.rename("u","u"); fileO.write(uh,nk*1.0)
    t_v.rename("t","t"); fileO.write(t_v,nk*1.0)
    sig_v.rename("sig","sig"); fileO.write(sig_v,nk*1.0)
    ph.rename("p","p"); fileO.write(ph,nk*1.0)

    E_t   = assemble((th-t_ex)**2*dx)
    E_sig1 = assemble((sigma_ex-sigmah)**2*dx)
    E_sig2 = assemble(dot(div(sigma_ex)-div(sigmah),div(sigma_ex)-div(sigmah))**(2./3.)*dx)# norm div,4/3
    E_u   = assemble(dot(uh-u_ex,uh-u_ex)**2*dx) # norm 0,4
    E_gam = assemble((gammah-gamma_ex)**2*dx)
    E_p   = assemble((ph-p_ex)**2*dx)
    
    errt.append(pow(E_t,0.5))
    errsigma.append(pow(E_sig1,0.5)+pow(E_sig2,0.75)) # norm div,4/3
    erru.append(pow(E_u,0.25)) # norm 0,4
    errgamma.append(pow(E_gam,0.5))
    errp.append(pow(E_p,0.5))
    
    if(nk>0):
        ratet.append(ln(errt[nk]/errt[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratesigma.append(ln(errsigma[nk]/errsigma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rateu.append(ln(erru[nk]/erru[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        rategamma.append(ln(errgamma[nk]/errgamma[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        ratep.append(ln(errp[nk]/errp[nk-1])/ln(hvec[nk]/hvec[nk-1]))
        
        

# ********  Generating error history **** #
print('==============================================================================================================')
print('   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  ')
print('==============================================================================================================')

for nk in range(nkmax):
    print('{:6d} & {:.4f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} & {:1.2e} & {:.2f} '.format(nvec[nk], hvec[nk], errt[nk], ratet[nk], errsigma[nk], ratesigma[nk], erru[nk], rateu[nk], errgamma[nk], rategamma[nk], errp[nk], ratep[nk]))
print('==============================================================================================================')



'''
'''
l = 0 

==============================================================================================================
   nn  &   hh   &   e(t)   & r(t) &  e(sig)  & r(s) &   e(u)   & r(u) &  e(gam)  & r(g) &   e(p)   & r(p)  
==============================================================================================================
  1111 & 1.7321 & 3.38e+00 & 0.00 & 1.75e+01 & 0.00 & 9.86e-01 & 0.00 & 4.23e+00 & 0.00 & 1.12e+00 & 0.00 
  8698 & 0.8660 & 1.84e+00 & 0.88 & 8.32e+00 & 1.08 & 5.65e-01 & 0.80 & 1.46e+00 & 1.54 & 4.88e-01 & 1.20 
 69016 & 0.4330 & 9.77e-01 & 0.91 & 4.26e+00 & 0.97 & 3.02e-01 & 0.90 & 5.18e-01 & 1.49 & 3.52e-01 & 0.47 
550156 & 0.2165 & 4.96e-01 & 0.98 & 2.13e+00 & 1.00 & 1.55e-01 & 0.96 & 1.50e-01 & 1.79 & 1.91e-01 & 0.88 
4393876 & 0.1083 & 2.50e-01 & 0.99 & 1.07e+00 & 1.00 & 7.80e-02 & 0.99 & 5.19e-02 & 1.53 & 9.55e-02 & 1.00 

l=1

 2266 & 1.7321 & 1.72e+00 & 0.00 & 1.05e+01 & 0.00 & 5.35e-01 & 0.00 & 1.38e+00 & 0.00 & 1.00e+00 & 0.00 
 17632 & 0.8660 & 5.67e-01 & 1.60 & 2.91e+00 & 1.85 & 2.40e-01 & 1.15 & 3.93e-01 & 1.82 & 2.00e-01 & 2.32 
139372 & 0.4330 & 1.61e-01 & 1.82 & 7.57e-01 & 1.94 & 6.58e-02 & 1.87 & 1.04e-01 & 1.92 & 5.39e-02 & 1.89 
1108756 & 0.2165 & 4.43e-02 & 1.86 & 1.94e-01 & 1.97 & 1.71e-02 & 1.95 & 3.68e-02 & 1.50 & 1.41e-02 & 1.93 


'''
