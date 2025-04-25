   
    #=
    # self._mean_value = self._mean(self._t)
        # self._log_det = np.sum(np.log(self._d))
        #     self._norm = -0.5 * (
        #         self._log_det + self._size * np.log(2 * np.pi)
        #     )
   # def mean(self):
   #      alpha = self.gp._do_solve((self.y - self.gp._mean_value)[:, None])[
   #          :, 0
   #      ]

   #      if self.t is None and self.kernel is None:
   #          mu = self.y - self.gp._diag * alpha
   #          if not self.include_mean:
   #              mu -= self.gp._mean_value
   #          return mu

   #      mu = self.gp._zeros_like(self._xs)
   #      mu = self._do_dot(alpha, mu)

   #      if self.include_mean:
   #          mu += self.gp._mean(self._xs)
   #      return mu
    # @property
    # def variance(self):
    #     if self.kernel is None:
    #         kernel = self.gp.kernel
    #     else:
    #         kernel = self.kernel
    #     return kernel.get_value(0.0) - self._diagdot(self.KxsT, self.Kinv_KxsT)

    # @property
    # def covariance(self):
    #     if self.kernel is None:
    #         kernel = self.gp.kernel
    #     else:
    #         kernel = self.kernel
    #     neg_cov = -kernel.get_value(self._xs[:, None] - self._xs[None, :])
    #     neg_cov = self._do_dot(self.Kinv_KxsT, neg_cov)
    #     return -neg_cov
function _dodot(gp)
    σ = sqrt.(diag(gp.Σy))
    U1, V1, phi1, A=_init_matrices(gp.kernel,gp.x,σ)
    # U1 = gp.U
    # V1 = gp.W    
end

   def _do_dot_tril(self, y):
        z = y * np.sqrt(self._d)[:, None]
        return driver.matmul_lower(self._t, self._c, self._U, self._W, z, z)

    def _do_norm(self, y):
        alpha = y[:, None]
        alpha = driver.solve_lower(
            self._t, self._c, self._U, self._W, alpha, alpha
        )[:, 0]
        return np.sum(alpha**2 / self._d)
    import numpy as np

def general_matmul_lower(t1, t2, c, U, V, Y, Z):
    # Request buffers
    t1buf = t1
    t2buf = t2
    cbuf = c
    Ubuf = U
    Vbuf = V
    Ybuf = Y
    Zbuf = Z

    # Parse dimensions
    if t1buf.ndim <= 0:
        raise ValueError("Invalid number of dimensions: t1")
    N = t1buf.shape[0]
    if t2buf.ndim <= 0:
        raise ValueError("Invalid number of dimensions: t2")
    M = t2buf.shape[0]
    if cbuf.ndim <= 0:
        raise ValueError("Invalid number of dimensions: c")
    J = cbuf.shape[0]
    if Ybuf.ndim <= 1:
        raise ValueError("Invalid number of dimensions: Y")
    nrhs = Ybuf.shape[1]

    # Check shapes
    if t1buf.ndim != 1 or t1buf.shape[0] != N:
        raise ValueError("Invalid shape: t1")
    if t2buf.ndim != 1 or t2buf.shape[0] != M:
        raise ValueError("Invalid shape: t2")
    if cbuf.ndim != 1 or cbuf.shape[0] != J:
        raise ValueError("Invalid shape: c")
    if Ubuf.ndim != 2 or Ubuf.shape[0] != N or Ubuf.shape[1] != J:
        raise ValueError("Invalid shape: U")
    if Vbuf.ndim != 2 or Vbuf.shape[0] != M or Vbuf.shape[1] != J:
        raise ValueError("Invalid shape: V")
    if Ybuf.ndim != 2 or Ybuf.shape[0] != M or Ybuf.shape[1] != nrhs:
        raise ValueError("Invalid shape: Y")
    if Zbuf.ndim != 2 or Zbuf.shape[0] != N or Zbuf.shape[1] != nrhs:
        raise ValueError("Invalid shape: Z")

    # Perform matrix multiplication
    if nrhs == 1:
        Y_ = Ybuf.flatten()
        Z_ = Zbuf.flatten()
        Z_[:] = celerite2.core.general_matmul_lower(t1buf, t2buf, cbuf, Ubuf, Vbuf, Y_, Z_)
    else:
        Y_ = Ybuf
        Z_ = Zbuf
        Z_[:] = celerite2.core.general_matmul_lower(t1buf, t2buf, cbuf, Ubuf, Vbuf, Y_, Z_)

    return Z
    def _do_general_matmul(self, c, U1, V1, U2, V2, inp, target):
        target = driver.general_matmul_lower(
            self._xs, self.gp._t, c, U2, V1, inp, target
        )
        target = driver.general_matmul_upper(
            self._xs, self.gp._t, c, V2, U1, inp, target
        )
        return target
 def _do_dot(self, inp, target):
        if self.kernel is None:
            kernel = self.gp.kernel
            U1 = self.gp._U
            V1 = self.gp._V
        else:
            kernel = self.kernel
            if self._U1 is None or self._V1 is None:
                _, _, self._U1, self._V1 = kernel.get_celerite_matrices(
                    self.gp._t,
                    self.gp._zeros_like(self.gp._t),
                    U=self._U1,
                    V=self._V1,
                )
            U1 = self._U1
            V1 = self._V1

        if self._c2 is None or self._U2 is None or self._V2 is None:
            self._c2, _, self._U2, self._V2 = kernel.get_celerite_matrices(
                self._xs,
                self.gp._zeros_like(self._xs),
                c=self._c2,
                U=self._U2,
                V=self._V2,
            )
        c = self._c2
        U2 = self._U2
        V2 = self._V2

        is_vector = False
        if inp.ndim == 1:
            is_vector = True
            inp = inp[:, None]
            target = target[:, None]

        target = self._do_general_matmul(c, U1, V1, U2, V2, inp, target)

        if is_vector:
            return target[:, 0]
        return target

=#


    rng=MersenneTwister(42)
    # The input coordinates must be sorted
    x = sort(cat(1, 3.8 .* rand(57), 5.5 .+ 4.5 .* rand(68);dims=1));
    yerr = 0.08 .+ (0.22-0.08) .*rand(length(x));
    y = 0.2.*(x.-5.0) .+ sin.(3.0.*x .+ 0.1.*(x.-5.0).^2) .+ yerr .* randn(length(x));

    true_x=collect(range(0,stop=10,length=126))
    true_y = 0.2 .*(true_x.-5) .+ sin.(3 .*true_x .+ 0.1.*(true_x.-5).^2);

    # non periodic component
    Q = 1.0/sqrt(2.0) ; w0 = 3.0
    S0 = var(y) ./ (w0 * Q)
    comp_1=Celerite2.SHOKernel(S0,Q,w0)

    # periodic component
    Q = 1.0; w0 = 3.0
    S0 = var(y) ./ (w0 * Q)
    comp_2=Celerite2.SHOKernel(S0,Q,w0)

    kernel = comp_1 + comp_2
    gp=Celerite2.CeleriteGP(kernel,x,yerr)
    logL=logpdf(gp,y)

    Q = 1.0 / sqrt(2.0)
    w0 = 3.0
    S0 = var(y) ./ (w0 * Q)
    kernel1 = celerite.SHOTerm(log(S0), log(Q), log(w0))
    Q = 1.0
    w0 = 3.0
    S0 = var(y) ./ (w0 * Q)
    orig_kernel = kernel1 + celerite.SHOTerm(log(S0), log(Q), log(w0))
    orig_gp = celerite.Celerite(orig_kernel)
    celerite.compute_ldlt!(orig_gp, x, yerr) 

    orig_logL= celerite.log_likelihood_ldlt(orig_gp, y)
    # println("Difference in logLikelihood: ",orig_logL-logL)
    # @test isapprox(orig_logL,logL)

    matrixK = Celerite2._k_matrix(gp,true_x,gp.x)
    orig_matrixK = celerite.get_matrix(orig_gp,true_x,orig_gp.x)
    # println("Diff. in K matrices: ",max(matrixK .- orig_matrixK))

    # predict GP
    M = 126*4
    tpred = sort!(rand(M)) #.* 200
    orig_ypred_ldlt = celerite.predict_ldlt!(orig_gp, x, y, true_x)
    # tpred2 = zeros(tpred)
    ypred = Celerite2.predict!(gp, gp.x, y, true_x)
    println("Max diff in predicted times:",maximum(abs.(ypred - orig_ypred_ldlt)))
    ## optimize GP
    # init_kernel = copy(kernel)
    init_vector = Celerite2.get_kernel(kernel)
    mask = ones(Bool,size(kernel))
    # We don't want to fit the first Q
    mask[2] = false
    function logL_wrapper(params)
        init_vector[mask] = params
        Celerite2.set_kernel!(kernel,init_vector)
        # @show kernel
        gp_trial=Celerite2.CeleriteGP(kernel,x,yerr)
        logL_trial=logpdf(gp_trial,y)
        return -logL_trial
    end
    # @time res_NM = optimize(logL_wrapper, init_vector[mask])
    # @time res_LBFGS = optimize(logL_wrapper, init_vector[mask],LBFGS())
    # @time res_auto = optimize(logL_wrapper, init_vector[mask],BFGS();autodiff = :forward)
    # @show res.minimizer
    Celerite2.set_kernel!(gp.kernel,vector)

    vector = celerite.get_parameter_vector(orig_gp.kernel)
    mask = ones(Bool, length(vector))
    mask[2] = false  # Don't fit for the first Q
    function nll_ldlt(params)
    vector[mask] = params
    celerite.set_parameter_vector!(orig_gp.kernel, vector)
    celerite.compute_ldlt!(orig_gp, x, yerr)
    return -celerite.log_likelihood_ldlt(orig_gp, y)
    end
    orig_res = Optim.optimize(nll_ldlt, vector[mask], Optim.LBFGS())
    vector[mask] = Optim.minimizer(orig_res)
    celerite.set_parameter_vector!(orig_gp.kernel, vector)

    mu_ldlt, variance_ldlt = celerite.predict_full_ldlt(orig_gp, y, true_x, return_var=true)
    Celerite2.set_kernel!(gp.kernel,exp.(vector))


    function full_Math_mu(gp,y,x)
        mu = mean(x) .+ diag(kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,x))) .* (y .- mean(gp.x))
        return mu
    end

    function full_Math_var(gp,x)
        B = kernelmatrix(gp.kernel,x,x)  - kernelmatrix(gp.kernel,x,gp.x) * inv(kernelmatrix(gp.kernel,gp.x)) * kernelmatrix(gp.kernel,x,gp.x)'
        return B
    end

    full_var = var(diag(full_Math_var(gp,collect(true_x))))
    full_mu = full_Math_mu(gp,y,x)


#   orig_alpha = celerite.apply_inverse_ldlt(orig_gp,y)
#   alpha = Celerite2.apply_inverse(gp,y)
#   println("max diff in alpha :",maximum(abs.(orig_alpha - alpha)))
#   function testmean_and_cov(gp::Celerite2.CeleriteGP,x::AbstractVector,y_train)
#     # # alpha =  _solve!(gp.D, gp.U, gp.W, gp.phi, y)
#     alpha = Celerite2.apply_inverse(gp,y_train)
#     Kxs = Celerite2._k_matrix(gp,x,gp.x)
#     mu = Kxs * alpha
#     # # KxsT = transpose(Kxs)
#     # Cov = _k_matrix(gp,x)
#     # # t1 = cov ; t2 = Kxs ; t3 = apply_inverse(gp,KxsT)
#     # # (size(t1), size(t2), size(t3)) = ((126, 126), (126, 126), (126,))
#     # cov = cov - Kxs * apply_inverse(gp,KxsT) 
#     # # BL: error  "must have singleton at dim 2"  exists in celerite 
#     # # solution? either cov .- Kxs * alpha OR cov - Kxs .* alpha
#     return mu#,Cov
#     end
#   mu_2 = testmean_and_cov(gp,true_x,y)
#   # println("max diff in mu:",maximum(abs.(mu - mu_ldlt)))
#   println("max diff in mu:",maximum(abs.(mu_2 - mu_ldlt)))

#   # mu,variance= Celerite2.predict(gp,y,true_x)
#   # mu = mean(gp)
#     #     ypred_full = celerite.predict_full_ldlt(orig_gp, y0, xpred; return_cov = false)

#   # orig_mu, orig_cov = celerite.predict(orig_gp, y, true_x, return_var=false)
#   # @show orig_res.minimizer
#   # @test maximum(abs.(log.(res.minimizer).-orig_res.minimizer)) < 1e-4
#   # # res_NM
#   # println(orig_mu - mu)
#   # res_LBFGS
#mu, variance = celerite.predict_full(gp, y, true_x, return_var=true)

#=
function _predict!(gp::)
    function predict_ldlt!(gp::Celerite, t, y, x)
# Predict future times, x, based on a 'training set' of values y at times t.
# Runs in O((M+N)J^2) (variance is not computed, though)
    a_real, c_real, a_comp, b_comp, c_comp, d_comp = get_all_coefficients(gp.kernel)
    N = length(y)
    M = length(x)
    J_real = length(a_real)
    J_comp = length(a_comp)
    J = J_real + 2*J_comp

    b = apply_inverse_ldlt(gp,y)
    Q = zeros(J)
    X = zeros(J)
    pred = zeros(x)

    # Forward pass
    m = 1
    while m < M && x[m] <= t[1]
      m += 1
    end
    for n=1:N
        if n < N
          tref = t[n+1]
        else
          tref = t[N]
        end
        Q[1:J_real] = (Q[1:J_real] .+ b[n]).* exp.(-c_real .* (tref - t[n]))
        Q[J_real+1:J_real+J_comp] .= (Q[J_real+1:J_real+J_comp] .+ b[n] .* cos.(d_comp .* t[n])) .*
            exp.(-c_comp .* (tref - t[n]))
        Q[J_real+J_comp+1:J] .= (Q[J_real+J_comp+1:J] .+ b[n] .* sin.(d_comp .* t[n])) .*
            exp.(-c_comp .* (tref - t[n]))

        while m < M+1 && (n == N || x[m] <= t[n+1])
            X[1:J_real] = a_real .* exp.(-c_real .* (x[m] - tref))
            X[J_real+1:J_real+J_comp] .= a_comp .* exp.(-c_comp .* (x[m] - tref)) .* cos.(d_comp .* x[m]) .+
                b_comp .* exp.(-c_comp .* (x[m] - tref)) .* sin.(d_comp .* x[m])
            X[J_real+J_comp+1:J] .= a_comp .* exp.(-c_comp .* (x[m] - tref)) .* sin.(d_comp .* x[m]) .-
                b_comp .* exp.(-c_comp .* (x[m] - tref)) .* cos.(d_comp .* x[m])

            pred[m] = dot(X, Q)
            m += 1
        end
    end

    # Backward pass
    m = M
    while m >= 1 && x[m] > t[N]
        m -= 1
    end
    fill!(Q,0.0)
    for n=N:-1:1
        if n > 1
          tref = t[n-1]
        else
          tref = t[1]
        end
        Q[1:J_real] .= (Q[1:J_real] .+ b[n] .* a_real) .*
            exp.(-c_real .* (t[n]-tref))
        Q[J_real+1:J_real+J_comp] .= (Q[J_real+1:J_real+J_comp] .+ b[n] .* a_comp .* cos.(d_comp .* t[n]) .+
                                      b[n] .* b_comp .* sin.(d_comp .* t[n])) .*
                                      exp.(-c_comp .* (t[n] - tref))
        Q[J_real+J_comp+1:J] .= (Q[J_real+J_comp+1:J] .+ b[n] .* a_comp .* sin.(d_comp .* t[n]) .-
                                 b[n] .* b_comp.*cos.(d_comp .* t[n])) .*
                                 exp.(-c_comp .* (t[n] - tref))

        while m >= 1 && (n == 1 || x[m] > t[n-1])
            X[1:J_real] .= exp.(-c_real .* (tref - x[m]))
            X[J_real+1:J_real+J_comp] .= exp.(-c_comp .* (tref - x[m])) .* cos.(d_comp .* x[m])
            X[J_real+J_comp+1:J] .= exp.(-c_comp .* (tref - x[m])) .* sin.(d_comp .* x[m])

            pred[m] += dot(X, Q)
            m -= 1
        end
    end
  return pred
end

=#
    
#=
function matmul(x,diag,y)
    if size(x,1) != size(y,1)
        throw("Dimension mismatch.")
    end
end
function _mat_mult_lower!(A::Vector{Float64},U::Array{Float64, 2},V::Array{Float64, 2},phi::Array{Float64,2},z::AbstractVector)
    J,N = size(U)
    f = zeros(Float64,J)
    for n =2:N
      f .= phi[:,n-1] .* (f .+ V[:,n-1] .* z[n-1])
      y[n] += dot(U[:,n-1],f)
    end
    return y
end

function _mat_mult_upper!(A::Vector{Float64},U::Array{Float64, 2},V::Array{Float64, 2},phi::Array{Float64,2},z::AbstractVector,y::AbstractVector)
    f = zeros(Float64,J)
    for n = N-1:-1:1
      f .= phi[:,n] .* (f .+  U[:,n] .* z[n+1])
      y[n] += dot(V[:,n],f)
    end
    return y
end
function _do_mat_mult!(A,U,V,phi,z)
    y = A .* z
    z = _mat_mult_upper!(U,V,phi,z,y)
    y = _mat_mult_lower(A,U,V,phi,z,y)   
end
        J,N = size(gp.U)
      z = zeros(Float64,N)
    z[1] = y[1]
      f = zeros(Float64,J)
      for n =2:N
        f .= gp.phi[:,n-1] .* (f .+ gp.W[:,n-1] .* z[n-1])
        z[n] = (y[n] - dot(gp.U[:,n], f))
      end
    # The following solves L^T.z = y for z:
      y = copy(z)
      fill!(z, zero(Float64))
      z[N] = y[N] / gp.D[N]
      fill!(f, zero(Float64))
      for n=N-1:-1:1
        f .= gp.phi[:,n] .* (f .+  gp.U[:,n+1] .* z[n+1])
        z[n] = y[n]/ gp.D[n] - dot(gp.W[:,n], f)
      end
      return z
function _solve_lower(U::Array{Float64, 2},W::Array{Float64, 2},phi::Array{Float64,2},y::Vector{Float64})
    # Solve lower inverse of L.z = y for z:
    J,N = size(U)
    z = zeros(Float64,N)
    z[1] = y[1]
    f=zeros(Float64,J)
    @inbounds for n in 2:N
        f .= phi[:,n-1] .* (f .+ W[:,n-1] .* z[n-1])
        z[n] = (y[n] - dot(U[:,n],f))
    end
    return z
end

function _solve_upper!(U::Array{Float64, 2},W::Array{Float64, 2},phi::Array{Float64,2},z::Vector{Float64})
    # Solve upper inverse of L' .z = y for z:
    J,N = size(U)
    f=zeros(Float64,J)
    @inbounds for n = N-1:-1:1
        f .= phi[:,n] .* (f .+ U[:,n+1] .* z[n+1])
        z[n] -=  dot(W[:,n],f)
    end
end
function _do_solve!(U,W,phi,y)
    z = _solve_lower(U,W,phi,y)   
    z ./= D 
    z = _solve_upper!(U,W,phi,z)
end
=#

#=
function _factor_rewrite!(D::Vector{Float64}, U,W,phi)
    # BL:  Rewrite of cholesky method, assuming that _init_matrices is called first.
    # @warn "Unstable?"
    J,N=size(U)
    W[:,1] ./= D[1] 

    # Allocate array for recursive computation of low-rank matrices:   
    S = zeros(J, J);
    Sk=0.0 ;  phij = 0.0 ; Dn = 0.0 ; Uk=0.0 ;  Uj = 0.0 ; Wj=0.0; tmp=0.0
    @inbounds for n in 2:N # what is diff b/w this loop form and for n in 2:N
        # Update S
        for j in 1:J
            phij=phi[j,n-1]
            Wj=W[j,n-1]
            for k in 1:j
                S[k,j] +=( D[n-1] .* Wj * W[k,n-1])
                S[k,j] = phij * phi[k,n-1] * S[k,j]
                # S[j,k] =  phij * phi[n-1,k] * (S[j,k] .+ (D[n-1] * Wj * W[n-1,k]))
            end 
        end
        # Update W and D
        # zero_out!(Dn)
        Dn = 0.0
        for j in 1:J
            Uj = U[j,n] 
            Wj = W[j,n]
            for k = 1:j-1
                Sk = S[k,j] ; 
                Uk = U[k,n]
                tmp = Uj * Sk
                Dn +=  Uk * tmp 
                Wj -= Uk * Sk 
                W[k,n] -= tmp
            end
            tmp = Uj * S[j,j]
            D[n] +=  Uj * tmp
            W[j,n] = Wj - tmp
        end
       Dn = D[n] 
       D[n] = Dn
       for j in 1:J
            W[j,n] /= Dn
       end
    end

    logdetK = 0.0
    for n in 1:N
    logdetK += log(D[n])
    end
    return logdetK
end

=#