""" WIP 
# using Plots
@recipe function  f(x::Vector{Float64},post_gp::PosteriorCeleriteGP)
    x_train = post_gp.data
    y_train = post_gp.data.δ + mean(post_gp.prior)
    μ, σ2 = mean_and_var(post_gp.prior,y_train,x) 
    scale::Float64 = pop!(plotattributes, :ribbon_scale, 1.0)
    ribbon := scale .* sqrt.(σ2)
    fillalpha --> 0.3
    linewidth --> 2
    return x, μ
end
"""