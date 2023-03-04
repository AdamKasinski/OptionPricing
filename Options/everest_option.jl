using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function price_everest_normal(Notional::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    worst_performing::Int = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])

    return Notional*(C+assets[worst_performing,end])*ₑ^(-r*T)
    
end

function price_everest_LHS(Notional::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    worst_performing::Int = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])

    return Notional*(C+assets[worst_performing,end])*ₑ^(-r*T)
    
end

function price_everest_moment_matching(Notional::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    S₀s::Matrix{Float64} = reshape(repeat(assets[:,1],T),basket_volume,T)
    S₀df::Matrix{Float64} = S₀s.*[ℯ^(-r*t) for t in 1:T]'
    assets =  assets.*S₀df./mean(assets,dims=2)
    
    worst_performing::Int = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])

    return Notional*(C+assets[worst_performing,end])*ₑ^(-r*T)
    
end


function price_everest_quasi_monte_carlo(Notional::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    s = SobolSeq(basket_volume)
    Z::Matrix{Float64} = reduce(hcat, next!(s) for i = 1:T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    worst_performing::Int = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])

    return Notional*(C+assets[worst_performing,end])*ₑ^(-r*T)
    
end

function price_everest_antithetic_variates(Notional::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    antithetic_assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T]
        antithetic_assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(antithetic_delta[1:k-1,asset])) for k in 1:T]
    end

    worst_performing::Int = minimum(@views (assets[:,end] - assets[:,1])./assets[:,1])
    antithetic_worst_performing::Int = minimum(@views (antithetic_assets[:,end] - antithetic_assets[:,1])/antithetic_assets[:,1])

    return Notional*0.5*(C+worst_performing+antithetic_worst_performing)*ₑ^(-r*T)
    
end


function atlas_option_monte_carlo(num_of_sim::Int, α::Float64, Notional::Float64,T::Int, r::Float64, C::Float64, basket_volume::Int, 
                                                S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},method="basic")
    len = num_of_sim
    if method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)

    if method == "basic"
        rtrn = [price_everest_normal(Notional, T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_everest_antithetic_variates(Notional, T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_everest_quasi_monte_carlo(Notional, T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "moment_matching"
        rtrn = [price_everest_moment_matching(Notional, T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_everest_LHS(Notional, T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end