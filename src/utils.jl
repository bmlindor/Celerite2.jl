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

    function _init_matrices(kernel::Tk,x::AbstractVector,σ::AbstractVector) where Tk <: CeleriteKernel
        # Initialize matrices for K
        ar, cr, ac, bc, cc, dc = _get_coefficients(kernel)
        # Compute the dimensions of the problem:
        N=length(x)
        # Number of real and complex components:
        Jr = length(ar);    Jc = length(ac)
        # Rank of semi-separable components:
        J = Jr + 2*Jc
        # Sum over the diagonal kernel amplitudes to get elements of the diagonal:
        A = σ.^2 .+ (sum(ar) + sum(ac))
        # Compute time lag:
        dx = x[2:N] - x[1:N-1]
        trig_arg = x * dc'
        cosdt = cos.(trig_arg)
        sindt = sin.(trig_arg) 
        # Compute the real and complex components of U,V, and ϕ :
        ϕc = exp.(-dx * cc')
        U = cat(    (ones(N) * ar')', 
                    (cosdt .* ac' + sindt .* bc')' ,
                    (sindt .* ac' - cosdt .* bc')' ,dims=1)
        V = cat(    ones(N,Jr)' ,  cosdt'  , sindt' ,dims=1)
        ϕ = cat(  (ones(N-1) .* exp.( -dx * cr'))', ϕc',ϕc',dims=1)
        return U, V, ϕ, A
    end

    function _factor_after_init!(A::Vector{Float64}, U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2})
        # at input A = D, and V = W
        J,N=size(U)
        W[:,1] ./= A[1] 

        # Allocate array for recursive computation of low-rank matrices:   
        S::Array{Float64, 2} =zeros(J, J);
        Dn = 0.0 ; Sk = 0.0 ;  ; tmp = 0.0 ;Wj = 0.0 ; ϕj = 0.0 ;  Uj = 0.0
        @inbounds for n in 2:N 
            Dn = A[n-1]
            for j in 1:J, k in 1:j
                S[k,j] = (ϕ[j,n-1] * ϕ[k,n-1]) * (S[k,j] + (Dn * W[j,n-1] * W[k,n-1]))
            end
            Dn = 0.0
            for j in 1:J
                Uj= U[j,n]
                Wj = W[j,n]
                for k in 1:j-1
                    Sk = S[k,j]
                    tmp = Uj * Sk
                    Dn +=  tmp * U[k,n]
                    Wj -= U[k,n] * Sk
                    W[k,n] -= tmp
                end
                tmp = Uj * S[j,j]
                Dn += .5 * Uj * tmp
                W[j,n] = Wj - tmp 
            end
            Dn = A[n] - 2 * Dn
            A[n] = Dn
            if Dn <= 0
                @warn "Diagonal is not positive definite." 
                # This should only happen during parameter inference.
            return 0
                # break?
                #  apply sturm's theorem ?
            end
            for j in 1:J
                W[j,n] /= Dn 
            end
        end # n loop
        logdetK=sum(log.(A))
        return logdetK
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