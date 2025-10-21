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
    # Compute the real and complex components of Ũ,Ṽ, and ϕ :
    ϕc = exp.(-dx * cc')
    Ũ = cat(    (ones(N) * ar')', 
                (cosdt .* ac' + sindt .* bc')' ,
                (sindt .* ac' - cosdt .* bc')' ,dims=1)
    Ṽ = cat(    ones(N,Jr)' ,  cosdt'  , sindt' ,dims=1)
    ϕ = cat(  (ones(N-1) .* exp.( -dx * cr'))', ϕc',ϕc',dims=1)
    return Ũ, Ṽ, ϕ, A
end
function _factor_rewrite!(D::AbstractVector, U::AbstractMatrix,W::AbstractMatrix,ϕ::AbstractMatrix)
    # BL:  Rewrite of cholesky method, assuming that _init_matrices is called first.
    # @warn "Unstable?"
    J,N=size(U)
    W[:,1] ./= D[1] 

    # Allocate array for recursive computation of low-rank matrices:   
    S = zeros(J, J);
    Sk=0.0 ;  ϕj = 0.0 ; Dn = 0.0 ; Uk=0.0 ;  Uj = 0.0 ; Wj=0.0; tmp=0.0
    @inbounds for n in 2:N # what is diff b/w this loop form and for n in 2:N
        # Update S
        for j in 1:J
            ϕj=ϕ[j,n-1]
            Wj=W[j,n-1]
            for k in 1:j
                S[k,j] +=( D[n-1] .* Wj * W[k,n-1])
                S[k,j] = ϕj * ϕ[k,n-1] * S[k,j]
                # S[j,k] =  ϕj * ϕ[n-1,k] * (S[j,k] .+ (D[n-1] * Wj * W[n-1,k]))
            end 
        end
        # Update W and D
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
    return W,S,logdetK
end

function _factor_test!(A::AbstractVector, Ũ::AbstractMatrix,W̃::AbstractMatrix,ϕ::AbstractMatrix)
    J,N=size(U)
    D = similar(A)
    D[1] = A[1]
    W̃[:,1] ./= A[1] 
    # tmp = zeros(eltype(W̃), J, 2)
    # Allocate array for recursive computation of low-rank matrices:   
    S = zeros(J, J);
    Dn = 0.0 ; Sk = 0.0 ;  ; tmp = 0.0 ;Wj = 0.0 ;ϕj = 0.0 ; 
     @inbounds for n in 2:N 
        Dn = D[n-1]
        for j=1:J
            ϕj= ϕ[j,n-1]
            W̃j = W̃[j,n-1]
            for k=1:j
            S[k,j] = (ϕj * ϕ[k,n-1]) * (S[k,j] + (Dn * W̃j * W̃[k,n-1]))
            end
        end
        @show S
        # update W and D
        Dn = 0.0
        for j=1:J
            for k=1:j
                tmp = Ũ[j,n] * S[k,j]
                Dn = A[n] - Ũ[k,n] * tmp
                W̃[j,n] -=  Ũ[k,n] * S[k,j]
                W̃[k,n] -= tmp
            end
            tmp = Ũ[j,n] * S[j,j]
            Dn = A[n] - Ũ[j,n] * tmp
            W̃[j,n] -= tmp
            W̃[j,n] /= Dn
        end
        # if Dn <= 0
        #     @warn "Diagonal is not positive definite." 
        #     # This should only happen during parameter inference.
        #     return 0
        #     # break?
        #     #  apply sturm's theorem ?
        # end
     end
    logdetK=sum(log.(D))
    println(logdetK)
    return W̃,S 
end

# include("Celerite2.jl")
# using Main.Celerite2
filename = string("research/Celerite2.jl/test/simulated_gp_data.txt")
data=readdlm(filename,comments=true)
x = data[:,1];
y = data[:,2];
yerr = data[:,3];
Q = 1.0/sqrt(2.0) ; w0 = 3.0
S0 = var(y) ./ (w0 * Q)

comp_1=Celerite2.SHOKernel(log(S0),log(Q),log(w0))
U,V,ϕ,A=_init_matrices(comp_1,x,yerr)
# W_re,S_re,logdetK0=_factor_rewrite!(A,U,V,ϕ)

_factor_test!(A,U,V,ϕ)
# test3=factor(reshape(U,126,2),reshape(ϕ,125,2),A,reshape(V,126,2))
# invKy =  Celerite2._solve!(A, U, V, ϕ, y)
# logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
# @show logL
gp=Celerite2.CeleriteGP(comp_1,x,yerr)
logL0=logpdf(gp,y)
coeffs = Celerite2._get_coefficients(gp.kernel)
logdetK = Celerite2._factorize!(gp.D, gp.U, gp.W, gp.phi, coeffs , collect(gp.x), gp.Σy)
