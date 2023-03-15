using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling



function generateBasket(basket_volume::Int,T::Int,S₀::Array{Float64},mu::Array{Float64},sigma::Array{Float64},epsilon::Matrix{Float64})
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for t in 1:T
        assets[:,t] = S₀.*exp.((mu .- 0.5.*sigma.^2).*(1/T) .+ sigma.*sqrt(1/T).*epsilon[t,:])
        S₀ = assets[:,t]
    end
    return assets
end

function price_everest_normal(T::Int, r::Float64, C::Float64, basket_volume::Int, 
                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    worst_performing::Int = argmin(@views (assets[:,end] - S₀)./S₀)

    return (C+assets[worst_performing,end])*ℯ^(-r*T)
    
end

function price_everest_LHS(T::Int, r::Float64, C::Float64, basket_volume::Int, 
                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})

    Z::Matrix{Float64} = scaleLHC(randomLHC(T,basket_volume),[(-1,1),(-1,1),(-1,1)])'
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)

    worst_performing::Int = argmin(@views (assets[:,end] - S₀)./S₀)

    return (C+assets[worst_performing,end])*ℯ^(-r*T)
    
end

function price_everest_moment_matching(num_of_sim::Int, α::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    
    assets_to_optimise::Array{Float64} = zeros(num_of_sim)
    d = Normal()
    worst_performing::Int = 0;
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,T))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
        worst_performing = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])
        assets_to_optimise[iteration] = assets[worst_performing,end]
    end
    S₀df::Float64 = S₀[worst_performing]*ℯ^(r*T)
    assets_to_calculate::Array{Float64} = assets_to_optimise.*S₀df./mean(assets_to_optimise)
    all_values::Array{Float64} = (C.+assets_to_calculate)*ℯ^(-r*T)
    θ::Float64 = mean(all_values)
    s::Float64 = std(all_values)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
    
end


function price_everest_quasi_monte_carlo(T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},sobolSeq)
    s = sobolSeq
    Z::Matrix{Float64} = reshape(reduce(hcat, next!(s) for i = 1:T*basket_volume),basket_volume,T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)

    worst_performing::Int = argmin(@views (assets[:,end] - S₀)./S₀)

    return (C+assets[worst_performing,end])*ℯ^(-r*T)
    
end

function price_everest_antithetic_variates(T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    antithetic_assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,antithetic_delta)


    worst_performing::Int = argmin(@views (assets[:,end] - S₀)./S₀)
    antithetic_worst_performing::Int = argmin(@views (antithetic_assets[:,end] - S₀)./S₀)

    return C+0.5*(assets[worst_performing,end]*ℯ^(-r*T)+assets[antithetic_worst_performing,end]*ℯ^(-r*T))

end


function everest_option_monte_carlo(num_of_sim::Int, α::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},method="basic")
    len = num_of_sim
    if method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)
    sobolSeq = SobolSeq(-1,1)

    if method == "basic"
        rtrn = [price_everest_normal(T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_everest_antithetic_variates(T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_everest_quasi_monte_carlo(T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix, sobolSeq) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_everest_LHS(T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end