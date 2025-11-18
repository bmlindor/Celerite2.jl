module DFMCeleriteGP
using Celerite2,Distributions,AbstractGPs,LinearAlgebra
# function factor_grad!(A::Vector{Float64}, U::Array{Float64, 2},W::Array{Float64, 2},ϕ::Array{Float64,2},
#                     dA::Vector{Float64},dV::Array{Float64, 2})
#     J,N=size(U)

#     dU = zeros(J,N)
#     dϕ = zeros(N-1,J)
#     S::Array{Float64, 2} =zeros(J, J);
#     dS::Array{Float64, 2} =zeros(J, J);
#     for j in 1:J
#         dV[j,:] ./= A
#     end

#     dS_ = copy(dS)
#     S_ = copy(S)
#     bSWT = similar(W[1, :])

#     # v_n = 
#     for n in N:-1:2
#         for j in 1:J
#             for k in 1:j
#                 dA[n] -= W[j,n] .* dV[j,n] 
#                 dU[:,n] .-= (dV[:,n] .+ 2 *dA[n] * U[:,n]) * S_
#         dS_ .-= U[:,n]' * (dV[:,n] .+ dA[n] .* U[:,n])
#     end
# end
# end
#     return 
# end

function get_matrices(kernel::Tk,x::AbstractVector,σ::AbstractVector) where Tk <: Celerite2.CeleriteKernel
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
    U = cat(    (ones(N) * ar'), 
                (cosdt .* ac' + sindt .* bc') ,
                (sindt .* ac' - cosdt .* bc') ,dims=2)
    V = cat(    ones(N,Jr) ,  cosdt  , sindt ,dims=2)
    ϕ = cat(  (ones(N-1) .* exp.( -dx * cr')), ϕc,ϕc,dims=2)
    return U, V, ϕ, A
end
function factor!(
  U::AbstractArray{T, 2},
  P::AbstractArray{T, 2},
  d::AbstractArray{T, 1},
  W::AbstractArray{T, 2},
  Sn::AbstractArray{T, 2}
) where T
  N, J = size(U)
  W[1, :] ./= d[1]

  @inbounds for n in 2:N
    Sn += d[n - 1] .* W[n - 1, :] * transpose(W[n - 1, :])
    Sn = P[n - 1, :] .* Sn .* transpose(P[n - 1, :])
    tmp = transpose(U[n, :]) * Sn
    d[n] -= tmp * U[n, :]
    if (d[n] <= 0)
      throw("matrix not positive definite")
    end
    W[n, :] -= transpose(tmp)
    W[n, :] ./= d[n]
  end
#   return Sn
    # Compute the log determinant 
    logdetK=sum(log.(d))
    return logdetK 
end

function solve_DFM!(
  U::AbstractArray{T, 2},
  P::AbstractArray{T, 2},
  d::AbstractArray{T, 1},
  W::AbstractArray{T, 2},
  Z::AbstractArray{T}
) where T

  N, J = size(U)
  nrhs = size(Z, 2)
  Fn = zeros(J, nrhs)

  @inbounds for n in 2:N
    Fn = P[n - 1, :] .* (Fn + W[n - 1, :] * transpose(Z[n - 1, :]))
    Z[n, :] -= transpose(Fn) * U[n, :]
  end

  Z ./= d

  fill!(Fn, 0)
  @inbounds for n in N-1:-1:1
    Fn = P[n, :] .* (Fn + U[n + 1, :] * transpose(Z[n + 1, :]))
    Z[n, :] -= transpose(Fn) * W[n, :]
  end
  return Z
end


function solve_grad!(
    U::AbstractMatrix,  # (N, J)
    P::AbstractMatrix,  # (N-1, J)
    d::AbstractVector,  # (N)
    W::AbstractMatrix,  # (N, J)
    Z::AbstractArray,  # (N, Nrhs); initially set to Y
    F::AbstractMatrix,  # (J, Nrhs)
    G::AbstractMatrix   # (J, Nrhs)
)
    N = size(U, 1)

    fill!(F, 0)
    fill!(G, 0)

    for n in 2:N
        F .+= W[n-1, :]' * Z[n-1, :]
        F .= Diagonal(P[n-1, :]) * F
        Z[n, :] .-= U[n, :] * F
    end

    for j in 1:size(Z, 2)
        Z[:, j] ./= d
    end

    for n in (N-1):-1:1
        G .+= U[n+1, :]' * Z[n+1, :]
        G .= Diagonal(P[n, :]) * G
        Z[n, :] .-= W[n, :] * G
    end
    return Z
end

function factor_grad!(
    U::AbstractMatrix,   # (N, J)
    P::AbstractMatrix,   # (N-1, J)
    d::AbstractVector,   # (N)
    W::AbstractMatrix,   # (N, J)
    S::AbstractMatrix,   # (J, J)
    
    bS::AbstractMatrix,  # (J, J)
    bU::AbstractMatrix,  # (N, J)
    bP::AbstractMatrix,  # (N-1, J)
    ba::AbstractVector,  # (N)
    bV::AbstractMatrix   # (N, J)
)
    N = size(U, 1)
    # make local copies of gradients
    bS_ = copy(bS)
    S_ = copy(S)
    bSWT = similar(W[1, :])

    # Element-wise division of each column of bV by d
    for j in 1:size(bV, 2)
        bV[:, j] ./= d
    end

    for n in (N-1):-1:1
        # Step 6
        ba[n] -= dot(W[n, :], bV[n, :])
        # @show size(ba),size(bV[n,:]),size(U[n,:]),size(S_),size(dU[n,:])
        bU[n, :] .-= (bV[n, :] .+ 2.0 * ba[n] .* U[n, :]) * S_
        bS_ .-= U[n, :]' * (bV[n, :] .+ ba[n] .* U[n, :])

        # Step 4
        inv_diagP = Diagonal(1.0 ./ P[n-1, :])
        S_ = S_ * inv_diagP
        bP[n-1, :] .+= diag(bS_ * S_ + S_' * bS_)

        # Step 3
        diagP = Diagonal(P[n-1, :])
        bS_ = diagP * bS_ * diagP
        bSWT = bS_ * W[n-1, :]'
        ba[n-1] += dot(W[n-1, :], bSWT)
        bV[n-1, :] .+= W[n-1, :] * (bS_ + bS_')

        # Downdate S
        S_ = inv_diagP * S_
        S_ .-= d[n-1] * (W[n-1, :]' * W[n-1, :])
    end

    ba[1] -= dot(bV[1, :], W[1, :]')
end

import AbstractGPs: MeanFunction
mutable struct InitCeleriteGP{Tk<:Celerite2.CeleriteKernel,Tm<:MeanFunction,Tx<:AbstractVector,TΣ} <: AbstractMvNormal
        kernel::Tk
        mean::Tm
        x::Tx
        Σy::TΣ 
        D::Array{Float64,1}
        W::Array{Float64,2}
        U::Array{Float64,2}
        ϕ::Array{Float64,2}
        S::Array{Float64,2}
        initialized::Bool
        dD::Array{Float64,1}
        dW::Array{Float64,2}
        dU::Array{Float64,2}
        dϕ::Array{Float64,2}
        dS::Array{Float64,2}
end

function InitCeleriteGP(k::Celerite2.CeleriteKernel,x::AbstractVector, σ::AbstractVector) 
    # Get the kernel coefficients
    ar, cr, ac, bc, cc, dc = Celerite2._get_coefficients(k)
    N = length(x)
    # Number of real and complex components:
    Jr = length(ar);    Jc = length(ac)
    J = Jr + 2*Jc
    σ² = σ.^2
    # Number of real and complex components:
    σ² = σ.^2
    # Initialize matrices.

    U,V,ϕ,A=get_matrices(k,x,σ)
    return InitCeleriteGP(k,ZeroMean(),
        x,Diagonal(σ²),
        A,V,U,ϕ,zeros(J,J),
        true,
        zeros(N),zeros(N,J),zeros(N,J),zeros(N-1,J),zeros(J,J))
end 

function loglike(gp::InitCeleriteGP,y)
    N = length(gp.x)
    y0 = copy(y)
    logdetK = factor!(gp.U, gp.ϕ, gp.D, gp.W,gp.S)
    invKy = solve_DFM!(gp.U, gp.ϕ, gp.D, gp.W, y0)
    logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
    return logL
end

function grad_loglike(gp::InitCeleriteGP,y)
    N, J = size(gp.U)
    Fn = zeros(J, J) ; Gn = zeros(J,J)
    y0 = copy(y)
    factor_grad!(gp.U, gp.ϕ, gp.D, gp.W, gp.S, gp.dS, gp.dU, gp.dϕ, gp.dD, gp.dW)
    invKy = solve_grad!(gp.U,gp.ϕ, gp.D, gp.W,y0,Fn, Gn)
    logdetK = sum(log.(gp.D))
    logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
    return logL
end
export InitCeleriteGP
end
# Compute finite difference derivatives to check loglikelihood gradient
# dθ=big.(1e-8)
# grad_num = zeros(length(grad_ll))
# θ_num = (copy(θ_init))
# for i in eachindex(grad_ll)
#     θ_num .= θ_init
#     θ_num[i] += dθ
#     llp = loglikelihood(θ_num)
#     θ_num[i] -= 2*dθ
#     llm = loglikelihood(θ_num)
#     grad_num[i] = (llp - llm)/(2*dθ)
# end

# grad_a = grad_loglikelihood(θ_init)[2]
# [grad_num grad_a]
# include("Celerite2.jl")
# using Main.Celerite2
#=

# J,N=size(U)
comp_1=Celerite2.SHOKernel(log(S0),log(Q),log(w0))
U,V,ϕ,A=get_matrices(comp_1,x,yerr)

@time "Init" U_DFM,V_DFM,ϕ_DFM,A_DFM=get_matrices(comp_1,x,yerr)
@time "factor" Sn = factor!(U_DFM,ϕ_DFM,A_DFM,V_DFM)

# time1 = time()
logdetK = _factor_after_init!(A,U,V,ϕ)
# @show A_DFM
# @show A
dA = zeros(N)
dV = zeros(N,J)
dS = zeros(J,J)
dϕ = zeros(N-1,J)
dU = zeros(N,J)
# factor_grad(A,U,V,ϕ,dA,dV)
factor_grad(U_DFM,ϕ_DFM,A_DFM,V_DFM,Sn,dS,dU,dϕ,dA,dV)

# invKy =  Celerite2._solve!(A, U, V, ϕ, y)
# logL =  -0.5 *((logdetK + N * log(2*pi)) + (y' * invKy))
# time2 = time() - time1 
# println("New?",time2)
# function old()

# return logL0
# end
# @time "current" old()
# @show logL0
# coeffs = Celerite2._get_coefficients(gp.kernel)
# logdetK0 = _factorize_actual!(gp.D, gp.U, gp.W, gp.phi, coeffs , collect(gp.x), gp.Σy)
# println("Difference in U: ",maximum(abs.(gp.U  - U)))

=#