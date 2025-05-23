# BL: should I ignore building of diagonal matrix for speed ??
## CeleriteGP structure of prior
    struct CeleriteGP{Tk<:CeleriteKernel,Tm<:MeanFunction,Tx<:AbstractVector,TΣ } <: AbstractMvNormal
        kernel::Tk
        mean::Tm
        x::Tx
        Σy::TΣ 
        D::Vector{Float64}
        W::Array{Float64}
        U::Array{Float64}
        phi::Array{Float64}
    end

    function CeleriteGP(k::CeleriteKernel,x::AbstractVector, σ::AbstractVector)
        # Get the kernel coefficients
        ar, cr, ac, bc, cc, dc = _get_coefficients(k)
        N = length(x)
        # Number of real and complex components:
        Jr = length(ar);    Jc = length(ac)
        J = Jr + 2*Jc
        σ² = σ.^2
        # Initialize matrices.
        return CeleriteGP(k,ZeroMean(),
            x,Diagonal(σ²),
            zeros(N),zeros(J,N),zeros(J,N),zeros(J,N-1))
    end 

    function CeleriteGP(k::CeleriteKernel,x::AbstractVector, σ::AbstractVector,mean::Real)
        # Allows setting of scalar mean
        ar, cr, ac, bc, cc, dc = _get_coefficients(k)
        N = length(x)
        Jr = length(ar);    Jc = length(ac)
        J = Jr + 2*Jc
        σ² = σ.^2
        return CeleriteGP(k,ConstMean(mean),
            x,Diagonal(σ²),
            zeros(N),zeros(J,N),zeros(J,N),zeros(J,N-1))
    end 
    # Construct abstract CeleriteGP 
    CeleriteGP(k::CeleriteKernel) = CeleriteGP(k,ZeroMean(),zeros(Float64,0),zeros(Float64,0,0),zeros(Float64,0),zeros(Float64,0,0),zeros(Float64,0,0),zeros(Float64,0,0))
    CeleriteGP(k::CeleriteKernel;mean::Real=0.0) = CeleriteGP(k,ConstMean(mean), zeros(Float64,0),zeros(Float64,0,0),zeros(Float64,0),zeros(Float64,0,0),zeros(Float64,0,0),zeros(Float64,0,0))

    function compute!(gp::CeleriteGP,x::AbstractVector,σ::AbstractVector)
        # Call the choleksy function to factorize & update structure:
        N = length(x)
        gp.x = x
        gp.D = _reshape!(gp.D, N)
        gp.U = _reshape!(gp.U, J, N)
        gp.W = _reshape!(gp.W, J, N)
        gp.phi = _reshape!(gp.phi, J, N-1)
        Σy=Diagonal(σ.^2)
        coeffs = _get_coefficients(gp.kernel)
        logdetK = _factorize!(gp.D, gp.U, gp.W, gp.phi, coeffs , x, Σy)
        return logdetK
    end

# Compute the marginalized likelihood of the CeleriteGP 
    function Distributions.logpdf(gp::CeleriteGP,y::Vector{Float64})
        J,N = size(gp.U)
        y0 = copy(y)
        coeffs = _get_coefficients(gp.kernel)
        # Do cholesky decomposition and apply the inverse:
        logdetK = _factorize!(gp.D, gp.U, gp.W, gp.phi, coeffs , gp.x, gp.Σy)
        invKy =  _solve!(gp.D, gp.U, gp.W, gp.phi, y0)
        logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
        return logL
    end
# Sample from CeleriteGP prior where q is vector of draws from Normal(0,1)
    function _sample_gp(gp::CeleriteGP,q::AbstractVector) 
        if gp.D == zeros(length(gp.x)) || size(gp.D) != size(gp.x)
            throw("CeleriteGP must be computed for sorted input coordinates first.")
        end
        return _simulate_gp(gp.D,gp.U,gp.W,gp.phi,q)
    end
    Random.rand(rng::AbstractRNG,gp::CeleriteGP) = _sample_gp(gp,randn(rng,length(gp.x)))
    Random.rand(gp::CeleriteGP,N::Int) = _sample_gp(gp,randn(Random.GLOBAL_RNG,N))
    function Random.rand(rng::AbstractRNG,gp::CeleriteGP, N::Int)
        q = randn(rng,N)
        return _sample_gp(gp,q)
    end
# Wrappers to compute invKy and covariance matrix.
    # Compute α = K-¹ y 
    function apply_inverse(gp::CeleriteGP,y)
        if gp.D == zeros(length(gp.x)) || size(gp.D) != size(gp.x)
            throw("CeleriteGP must be computed for sorted input coordinates first.")
        end
      @assert(size(y,1)==length(gp.x))
        return _solve!(gp.D, gp.U, gp.W, gp.phi, y)
    end 
    # Compute full covariance matrix (without the noise)
    function _k_matrix(gp::CeleriteGP,xs...)
        # Can provide autocorrelation or cross-correlation
        @assert length(xs)<=2 
        local x1::Array
        local x2::Array
        if length(xs) >= 1
            x1 = xs[1]
        else
            if gp.D == zeros(length(gp.x)) || size(gp.D) != size(gp.x)
            throw("CeleriteGP must be computed for sorted input coordinates first.")
            end
            x1 = gp.x
        end
        if length(xs) == 2
            x2 = xs[2]
        else
            x2 = x1
        end
        if size(x1, 2) != 1 || size(x2, 2) != 1
            throw("Inputs must be 1D.")
        end
        τ = broadcast(-, reshape(x1, length(x1), 1), reshape(x2, 1, length(x2)))
        k = _get_value(gp.kernel, τ)
        return k
    end

# Compute covariance, variance, and mean of the prior process.
    function Statistics.mean(gp::CeleriteGP)
        # equivalent to AbstractGPs.mean_vector(gp_prior.mean,x)
        if typeof(gp.mean) == ZeroMean{Float64}
        mu = zeros(length(gp.x))
        elseif typeof(gp.mean) == ConstMean{Float64}
        mu = fill(gp.mean.c, length(gp.x))
        end
        return mu
    end
    Statistics.var(gp::CeleriteGP) = diag(gp.Σy)
    Statistics.cov(gp::CeleriteGP) = _reconstruct_K(gp,gp.x) + gp.Σy
    StatsBase.mean_and_cov(gp::CeleriteGP)= (mean(gp),cov(gp))

# Predict future values y*  based on a 'training set' of values y at times x.
    function predict!(gp::CeleriteGP,x_train,y_train,x)
        # Runs in O((M+N)J^2) but does not compute variance.
        ar, cr, ac, bc, cc, dc = _get_coefficients(gp.kernel)
        N = length(y_train) ;
        M = length(x)
        Jr = length(ar);    Jc = length(ac)
        J = Jr + 2*Jc

        b = apply_inverse(gp,y_train) # invKy
        Q = zeros(J)
        X = zeros(J)
        pred = similar(x) 
        fill!(pred,0)

        # Forward pass
        m = 1
        while m < M && x[m] <= x_train[1]
          m += 1
        end
        for n=1:N
            if n < N
              tref = x_train[n+1]
            else
              tref = x_train[N]
            end
            # Update Q:
            Q[1:Jr] = (Q[1:Jr] .+ b[n]) .* exp.(-cr .* (tref - x_train[n]))
            Q[Jr+1:Jr+Jc] .= (Q[Jr+1:Jr+Jc] .+ b[n] .* cos.(dc .* x_train[n])) .* 
                            exp.(-cc .* (tref - x_train[n]))
            Q[Jr+Jc+1:J] .= (Q[Jr+Jc+1:J] .+ b[n] .* sin.(dc .* x_train[n])) .* 
                            exp.(-cc .* (tref - x_train[n]))
            # Update X and m:
            while m < M+1 && (n == N || x[m] <= x_train[n+1])
                X[1:Jr] = ar .* exp.(-cr .* (x[m] - tref))
                X[Jr+1:Jr+Jc] .= ac .* exp.(-cc .* (x[m] - tref)) .* cos.(dc .* x[m]) .+ bc .* exp.(-cc .* (x[m] - tref)) .* sin.(dc .* x[m])
                X[Jr+Jc+1:J] .= ac .* exp.(-cc .* (x[m] - tref)) .* sin.(dc .* x[m]) .- bc .* exp.(-cc .* (x[m] - tref)) .* cos.(dc .* x[m])

                pred[m] = dot(X, Q)
                m += 1
            end
        end

        # Backward pass
        m = M
        while m >= 1 && x[m] > x_train[N]
            m -= 1
        end
        fill!(Q,0.0)
        for n=N:-1:1
            if n > 1
              tref = x_train[n-1]
            else
              tref = x_train[1]
            end
            Q[1:Jr] .= (Q[1:Jr] .+ b[n] .* ar) .*
                exp.(-cr .* (x_train[n]-tref))
            Q[Jr+1:Jr+Jc] .= (Q[Jr+1:Jr+Jc] .+ b[n] .* ac .* cos.(dc .* x_train[n]) .+
                                          b[n] .* bc .* sin.(dc .* x_train[n])) .*
                                          exp.(-cc .* (x_train[n] - tref))
            Q[Jr+Jc+1:J] .= (Q[Jr+Jc+1:J] .+ b[n] .* ac .* sin.(dc .* x_train[n]) .-
                                     b[n] .* bc.*cos.(dc .* x_train[n])) .*
                                     exp.(-cc .* (x_train[n] - tref))

            while m >= 1 && (n == 1 || x[m] > x_train[n-1])
                X[1:Jr] .= exp.(-cr .* (tref - x[m]))
                X[Jr+1:Jr+Jc] .= exp.(-cc .* (tref - x[m])) .* cos.(dc .* x[m])
                X[Jr+Jc+1:J] .= exp.(-cc .* (tref - x[m])) .* sin.(dc .* x[m])

                pred[m] += dot(X, Q)
                m -= 1
            end
        end
      return pred
    end

# Compute a vector of Normal distributions representing the marginals of the CeleriteGP 
    function AbstractGPs.marginals(gp::CeleriteGP)
        m = mean(gp)
        v = var(gp)
        return Normal.(m, sqrt.(v))
    end

    function AbstractGPs.posterior(gp::CeleriteGP,y::AbstractVector)
        mu = mean(gp)
        δ = y - mu
        α=apply_inverse(gp,y)
        return AbstractGPs.PosteriorGP(gp,(x=gp.x,α=α,δ=δ))
    end
# Construct the posterior process implied by conditioning CeleriteGP at observations of y at x. <- is equivalent to?  -> # Compute covariance, variance, and mean of the predictive distribution.
    function mean_and_var(gp::CeleriteGP,y_train::AbstractVector,x::AbstractVector)
        alpha = apply_inverse(gp,y_train)
        Kxs = _k_matrix(gp,x,gp.x)
        mu = Kxs * alpha
        KxsT = transpose(Kxs)

        v=zeros(length(x))
        for i=1:length(x)
        v[i] = -sum(KxsT[:,i] .* apply_inverse(gp, KxsT[:,i]))
        end
        σ² =  _get_value(gp.kernel,[0.0])[1] .+ v
        return mu, σ²[1,:]
    end

    function cov(gp::CeleriteGP,y_train::AbstractVector,x::AbstractVector)
        Kxs = _k_matrix(gp,x,gp.x)
        cov = _k_matrix(gp,x)
        cov -= Kxs * apply_inverse(gp,KxsT) 
        # # BL: error  "must have singleton at dim 2"  exists in celerite 
    end

# Reconstruct cholesky factor from low-rank decomposition.
    function _reconstruct_K(gp::CeleriteGP,x)
        if gp.D == zeros(length(gp.x)) || size(gp.D) != size(gp.x)
            throw("CeleriteGP must be computed for sorted input coordinates first.")
        end
        ar, cr, ac, bc, cc, dc = _get_coefficients(gp.kernel)
        Jr = length(ar);    Jc = length(ac)
        N = length(x)
        @assert N == length(gp.x)
        J = Jr + 2*Jc
        # Reconstruct cholesky factor from low-rank decomposition:
        Umat = copy(gp.U)
        Wmat = copy(gp.W)
        for n=1:N
          if Jr > 0
            for j=1:Jr
              Umat[j,n] *= exp(-cc[j]*x[n])
              Wmat[j,n] *= exp( cc[j]*x[n])
            end
          end
          if (J-Jr) > 0
            for j=1:Jc
              Umat[Jr+2j-1,n] *= exp(-cc[j]*x[n])
              Umat[Jr+2j  ,n] *= exp(-cc[j]*x[n])
              Wmat[Jr+2j-1,n] *= exp( cc[j]*x[n])
              Wmat[Jr+2j  ,n] *= exp( cc[j]*x[n])
            end
          end
        end
        L = tril(*(Umat', Wmat), -1)
        # Add identity matrix, then multipy by D^{1/2}:
        for i=1:N
          L[i,i] = 1.0
          for j=1:N
            L[i,j] *=sqrt(gp.D[j])
          end
        end
        K = L*L'
        return K
    end