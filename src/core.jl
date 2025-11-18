function _factorize!(D::Vector{Float64}, U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},coeffs::NTuple{6,Vector{T}}, x::Vector{Float64}, Σy::Diagonal{Float64}) where T <: Real
    # Do cholesky decomposition.
    ar, cr, ac, bc, cc, dc=coeffs
    N = size(x,1)
    Jr = length(ar);    Jc = length(ac)
    J = Jr + 2*Jc
    # Allocate array for recursive computation of low-rank matrices:   
    S::Array{Float64, 2} = zeros(J, J)
    # A =  σ.^2 
    A = diag(Σy) .+ (sum(ar) + sum(ac))
    # Compute the first element:
    D[1] = A[1]
    value = 1.0 / D[1]
    for j in 1:Jr
        U[j,1] = ar[j]
        W[j,1] = value
    end
    # EA: we compute these recursively to save time
    cosdt = zeros(Float64,Jc) ; sindt = zeros(Float64,Jc)
    for j in 1:Jc
        cosdt[j] = cos(dc[j]*x[1]) 
        sindt[j] = sin(dc[j]*x[1]) 
        U[Jr+2*j-1, 1] = ac[j] * cosdt[j] + bc[j] * sindt[j] 
        U[Jr+2*j,   1] = ac[j] * sindt[j] - bc[j] * cosdt[j]
        W[Jr+2*j-1, 1] = cosdt[j] * value
        W[Jr+2*j,   1] = sindt[j] * value
    end
    # Allocate temporary variables:
    dcd = 0.0 ; dsd = 0.0 ; cdj= 0.0 ; 
    Uj = 0.0 ; Uk = 0.0 ; Wj = 0.0 ;ϕj = 0.0 ; 
    Dn = 0.0 ; Sk = 0.0 ;  ; tmp = 0.0 ;

    @inbounds for n in 2:N
        # Compute time lag:
        tn = x[n]
        dx = tn - x[n-1]
        # Compute the real and complex components of Ũ, W̃ , and ϕ :
        for j in 1:Jr
            ϕ[j,n-1] = exp(-cr[j] * dx)
            U[j,n] = ar[j]
            W[j,n] = 1.0
        end
        for j in 1:Jc
            expdx = exp(-cc[j] * dx)
            ϕ[Jr+2*j-1,   n-1] = expdx
            ϕ[Jr+2*j,     n-1] = expdx
            cdj = cosdt[j]
            dcd = cos(dc[j] * dx) ; dsd = sin(dc[j] * dx) ;
            cosdt[j] = cdj * dcd - sindt[j]*dsd
            sindt[j] = sindt[j] * dcd + cdj * dsd

            U[Jr+2*j-1, n] = ac[j] * cosdt[j] + bc[j] * sindt[j] 
            U[Jr+2*j,   n] = ac[j] * sindt[j] - bc[j] * cosdt[j] 
            W[Jr+2*j-1, n] = cosdt[j] #cos(dc[j] * tn)
            W[Jr+2*j,   n] = sindt[j] #sin(dc[j] * tn)
        end
        # Compute S via recursion
        Dn = D[n-1]
        for j in 1:J
            ϕj= ϕ[j,n-1]
            Wj = W[j,n-1]
            for k in 1:j
                S[k,j] =  ϕj * ϕ[k,n-1] * (S[k,j] + (Dn * Wj * W[k,n-1]))
            end
        end
        # Update W and D
        Dn = 0.0
        for j in 1:J
            Uj = U[j,n]
            Wj = W[j,n]
            for k in 1:j-1
                Sk = S[k,j]
                tmp = Uj * Sk
                Uk = U[k,n]
                Dn += Uk * tmp
                Wj -= Uk * Sk
                W[k,n] -= tmp
            end
            tmp = Uj * S[j,j]
            Dn += .5 * Uj * tmp
            W[j,n] = Wj - tmp
        end

        # Finalize computation of D and W:
        Dn = A[n] - 2 * Dn
        D[n] = Dn
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
    # Compute the log determinant 
    logdetK=sum(log.(D))
    return logdetK 
end

function _solve!(D::Vector{Float64}, U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},y::AbstractVecOrMat)
    # Solves K.b=y for b with LDLT decomposition by applying the inverse
    J,N = size(U)
    z = zeros(Float64,N)
    z[1] = y[1]
    f=zeros(Float64,J)
    # Solve L.z = y for z:
    @inbounds for n in 2:N
        f .= ϕ[:,n-1] .* (f .+ W[:,n-1] .* z[n-1])
        z[n] = (y[n] - dot(U[:,n],f))
    end
    # Solve L' .z = y for z:
    z ./= D
    fill!(f,0.0)
    @inbounds for n = N-1:-1:1
        f .= ϕ[:,n] .* (f .+ U[:,n+1] .* z[n+1])
        z[n] -=  dot(W[:,n],f)
    end
    # Returns solution of L.L' z = y for z:
    return z
end

function _simulate_gp(D::Vector{Float64},U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},q::Vector)
    J,N = size(U)
    y=zeros(Float64,N)
    f=zeros(Float64,J)
    # Multiply lower Cholesky factor by random normal deviates:
    tmp = sqrt(D[1]) * q[1]
    y[1] = tmp
    @inbounds for n=2:N
        f .= ϕ[:,n-1] .* (f .+ W[:,n-1] .* tmp)
        tmp = sqrt(D[n]) * q[n] # error where it's attempting to access q[N+1]
        y[n] = tmp + dot( U[:,n],f)
    end
    # Returns simulated correlated noise
    return y
end
# BL: what if q is matrix?
function _simulate_gp(D::Vector{Float64},U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},q::AbstractArray)
    J,N = size(U)
    nrhs = size(q, 2)
    y=zeros(Float64,N)
    f=zeros(Float64,J,nrhs)
    # Multiply lower Cholesky factor by random normal deviates:
    tmp = sqrt(D[1]) * q[:,1]
    y[:,1] = tmp
    @inbounds for n=2:N
        f .= ϕ[:,n-1] .* (f .+ W[:,n-1] .* tmp)
        tmp = sqrt(D[n]) .* q[:,n] 
        y[:,n] = tmp +  dot(U[:,n],f)
    end
    # Returns simulated correlated noise
    return y
end

function _mat_mult(A::Vector{Float64},U::Array{Float64, 2},V::Array{Float64, 2},ϕ::Array{Float64,2},z::AbstractVector)
    # Compute y =  K . z 
    J,N = size(U)
    y = A .* z
    # Sweep upwards in n:
    f = zeros(Float64,J)
    for n =2:N
      f .= ϕ[:,n-1] .* (f .+ V[:,n-1] .* z[n-1])
      y[n] += dot(U[:,n-1],f)
    end
    # Sweep downwards in n:
    fill!(f, 0)
    for n = N-1:-1:1
      f .= ϕ[:,n] .* (f .+  U[:,n] .* z[n+1])
      y[n] += dot(V[:,n],f)
    end
    return y
end