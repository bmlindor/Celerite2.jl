function _init_matrices(kernel::Tk,x::AbstractVector,σ::AbstractVector) where Tk <: Celerite2.CeleriteKernel
    # Initialize matrices for K
    ar, cr, ac, bc, cc, dc = Celerite2._get_coefficients(kernel)
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
                # Uk = 
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

function _factor_grad(A::Vector{Float64}, U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},
                    dA,dV)
    J,N=size(U)
    dU = zeros(J,N)
    dϕ = zeros(N-1,J)
    dV ./= A
    # v_n = 
    @inbounds for n in 2:N 
        dA[n] -= W[:,n] * dV[:,n] 
    end
end

function _solve_grad()
    
end
# include("Celerite2.jl")
# using Main.Celerite2
# filename = string("research/Celerite2.jl/test/simulated_gp_data.txt")
# data=readdlm(filename,comments=true)
# x = data[:,1];
# y = data[:,2];
# yerr = data[:,3];
# Q = 1.0/sqrt(2.0) ; w0 = 3.0
# S0 = var(y) ./ (w0 * Q)

# comp_1=Celerite2.SHOKernel(log(S0),log(Q),log(w0))
# U,V,ϕ,A=_init_matrices(comp_1,x,yerr)
# time1 = time()
# logdetK = _factor_after_init!(A,U,V,ϕ)
# N = length(x)
# invKy =  Celerite2._solve!(A, U, V, ϕ, y)
# logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
# time2 = time() - time1 
# println("New?",time2)
# function old()
# gp=Celerite2.CeleriteGP(comp_1,x,yerr)
# logL0=logpdf(gp,y)
# return logL0
# end
# @time "current" old()
# @show logL0
# coeffs = Celerite2._get_coefficients(gp.kernel)
# logdetK0 = _factorize_actual!(gp.D, gp.U, gp.W, gp.phi, coeffs , collect(gp.x), gp.Σy)
# println("Difference in U: ",maximum(abs.(gp.U  - U)))
