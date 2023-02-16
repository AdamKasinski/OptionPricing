using LinearAlgebra, Distributions, Statistics, Sobol


function price_everest_normal(Notional,T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    worst_performing = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])

    return Notional*(C+assets[worst_performing,end])*ₑ^(-r*T)
    
end


function price_everest_quasi_monte_carlo(Notional,T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    s = SobolSeq(basket_volume)
    Z = reduce(hcat, next!(s) for i = 1:T)
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    worst_performing = argmin(@views (assets[:,end] - assets[:,1])./assets[:,1])

    return Notional*(C+assets[worst_performing,end])*ₑ^(-r*T)
    
end

function price_everest_antithetic_variates(Notional,T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    antithetic_delta = -Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    antithetic_assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T]
        antithetic_assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(antithetic_delta[1:k-1,asset])) for k in 1:T]
    end

    worst_performing = minimum(@views (assets[:,end] - assets[:,1])./assets[:,1])
    antithetic_worst_performing = minimum(@views (antithetic_assets[:,end] - antithetic_assets[:,1])/antithetic_assets[:,1])

    return Notional*0.5*(C+worst_performing+antithetic_worst_performing)*ₑ^(-r*T)
    
end


function everest_option_monte_carlo(num_of_sim, α , Notional,T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix,method)
    
    if method == "basic"
        len = num_of_sim
    elseif method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)
    if method == "basic"
        rtrn = [price_everest_normal(Notional,T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_everest_normal(Notional,T, r, C, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    end
    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end