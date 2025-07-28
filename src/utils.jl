# for sampling
RealKernel(u::ForwardDiff.Dual,v::ForwardDiff.Dual) = RealKernel(ForwardDiff.value.(u),ForwardDiff.value.(v))
SHOKernel(u::ForwardDiff.Dual,v::ForwardDiff.Dual,z::ForwardDiff.Dual) = SHOKernel(ForwardDiff.value.(u),ForwardDiff.value.(v),ForwardDiff.value.(z))
ComplexKernel(s::ForwardDiff.Dual,u::ForwardDiff.Dual,v::ForwardDiff.Dual,z::ForwardDiff.Dual) = ComplexKernel(ForwardDiff.value.(s),ForwardDiff.value.(u),ForwardDiff.value.(v),ForwardDiff.value.(z))
RotationKernel(s::ForwardDiff.Dual,t::ForwardDiff.Dual,u::ForwardDiff.Dual,v::ForwardDiff.Dual,z::ForwardDiff.Dual) = RotationKernel(ForwardDiff.value.(s),ForwardDiff.value.(t),ForwardDiff.value.(u),ForwardDiff.value.(v),ForwardDiff.value.(z))

SHOKernel(u::ForwardDiff.Dual,v::Float64,z::ForwardDiff.Dual) = SHOKernel(ForwardDiff.value.(u),v,ForwardDiff.value.(z))

## deprecated
    struct logSHOKernel{T} <: CeleriteKernel
    # Simple harmonic oscillator (SHO) kernel
        logS0::Vector{T}    # ∝ power spectral density at ω0
        logQ::Vector{T}     # oscillator quality factor
        logω0::Vector{T}    # frequency of undamped oscillator
    end

    function logSHOKernel(logS0::T,logQ::T,logω0::T) where T<:Float64
        return logSHOKernel([logS0],[logQ],[logω0])
    end

    function _get_coefficients(k::logSHOKernel) 
        eps = 1e-5 # for numerical stability
        # overdamped if Q < 0.5
        S0 = exp(only(k.logS0)); Q = exp(only(k.logQ)) ; ω0 = exp(only(k.logω0))
        # @assert Q > 0.0
        if Q < 0.5 
            # f = sqrt(max(1 - 4 * Q^2,eps))
            f = sqrt(1 - 4 * Q^2)
            a = 0.5 * S0 * ω0 * Q
            c = 0.5 * ω0 / Q
            return ([a * (1 + 1 / f), a * (1 - 1 / f)], [c * (1 - f), c * (1 + f)], 
                zeros(0), zeros(0), zeros(0),zeros(0))
        end
        # f = sqrt(max(4 * Q^2 - 1,eps))
        f = sqrt(4 * Q^2 - 1)
        a =  S0 * ω0 * Q
        c = 0.5 * ω0 / Q
        return (zeros(0), zeros(0), [a], [a / f], [c], [c * f])
    end
    Base.size(k::logSHOKernel) = 3
    get_kernel(k::logSHOKernel) = [only(k.logS0),only(k.logQ),only(k.logω0)]
    function set_kernel!(kernel::logSHOKernel,vector)
        kernel.logS0 .= [vector[1]] ; kernel.logQ .= [vector[2]] ; kernel.logω0 .= [vector[3]]
    end

## Utils adopted from celerite.jl
    function _full_solve(k::CeleriteKernel, x::Vector,σ::Vector)
        # Compute the full covariance matrix.
        N=length(x)
        Nmax=1000 
        if N > Nmax
            @warn "You are attempting to create the full covariance matrix for a dataset with length greater than $(Nmax). Do not use this method with large datasets."
        else
            @assert(N <= Nmax)
            @assert(N ==length(σ))
            ar, cr, ac, bc, cc, dc=_get_coefficients(k)
            # Diagonal components:
            A = σ.^2 .+ sum(ac) .+ sum(ar)
            K = zeros(Float64,N,N)
            for n=1:N
                for m=1:N
                # Compute the time lag, τ
                τ_nm = abs(x[n] - x[m])
                # Compute the kernel matrix:
                K[n,m] = sum(ac .* exp.(-cc .* τ_nm) .* cos.(dc .* τ_nm) .+ bc .* exp.(-cc .* τ_nm) .* sin.(dc .* τ_nm))
                end
                K[n,n]=A[n]
            end
            return K
        end
    end

    function _semi_separable_kernelmatrix(k::CeleriteKernel,x::Vector,σ::Vector)
        ar, cr, ac, bc, cc, dc=_get_coefficients(k)
        N=length(x)
        U,V = _compute_UV(k,x)
        A = σ.^2 .+ sum(ac) .+ sum(ar)
        K0 = tril(*(u, v'), -1) + triu(*(v, u'), 1)
        for i=1:N
            K0[i,i] = A[i]
        end
        return K0
    end

    function _semi_seperable_error(K,K0)
        @info "Semiseparable error: ", maximum(abs.(K .- K0))
    end

## Utils adopted from celerite2
    function _diag_dot(a,b)
        return ein"ij,ij->j"(a, b)
    end

    function make_test_data(N=10)
        rng = MersenneTwister(42)
        x = sort(N .* rand(rng,N));
        y = sin.(x);
        yerr = 0.01 .+ rand(rng,N) ./ 100;
        # Z=rand(rng,N,2)
        return N,x,yerr,y
    end
    function zero_out!(input::AbstractVecOrMat)
        input.=zeros(size(input))
        return 
    end

    function _reshape!(A::Array{Float64}, dims...)
    # Allocates arrays if size is not correct
    if size(A) != dims
        A = Array{Float64}(undef,dims...)
    end
    return A
    end
    # if check_sorted 
    #     if issorted(x)
    #     else
    #         println("Inputs are not sorted. Sorting.")
    #         x = sort(x)
    #     end
    # end
    function _compute_UV(k::CeleriteKernel, x::Vector)
        N=length(x)
        ar, cr, ac, bc, cc, dc=_get_coefficients(k)
        # Number of real and complex components:
        Jr = length(ar);    Jc = length(ac)
        # Rank of semi-separable components:
        J = Jr + 2*Jc
        # Compute the full matrices for U and V:
        U = zeros(Float64,N,J)
        V = zeros(Float64,N,J)
            for n=1:N
                for j=1:Jc
                    expct = exp(-cc[j]*x[n])
                    cosdt = cos(dc[j]*x[n])
                    sindt = sin(dc[j]*x[n])
                    U[n,j*2-1]=ac[j]*expct*cosdt+bc[j]*expct*sindt
                    U[n,j*2  ]=ac[j]*expct*sindt-bc[j]*expct*cosdt
                    V[n,j*2-1]=cosdt/expct
                    V[n,j*2  ]=sindt/expct
                end
            end
        return U,V
    end

# function KernelFunctions.kappa(term::SHOKernel,x)
#   t = abs.(x)
#   η = sqrt(abs(1/(1.0 - 4 * term.Q^2)))   
#   f=0.0 ;
#   if 0.0 < term.Q < 0.5
#       f = cosh.(η .*term.w0 .*t) .+ (0.5 .*sinh.(η .*term.w0 .*t) ./ η .*term.Q) 
#   elseif term.Q==0.5
#       f = 2 .* (1 .+ term.w0 .*t)
#   elseif 0.5 < term.Q  
#       f = cos.(η .*term.w0 .*t) .+ (0.5 .*sin.(η .*term.w0 .*t) ./ η .* term.Q) 
#   end
#   return f .*term.S0 .*term.w0 .*term.Q.*exp.(.- term.w0 .* t ./ 2 .*term.Q)
# end