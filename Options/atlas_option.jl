using LinearAlgebra, Distributions, Statistics


function price_atlas_normal(Notional, T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) # TODO: add types, n1/2 - amount of worst/best performing
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    remaining_stocks = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]

    return Notional * (1+maximum(0,mean(assets[remaining_stocks,end])-K))*ₑ^(-r*T)

end



function price_atlas_antithetic_variates(Notional, T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) # TODO: add types
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

    remaining_stocks = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]
    antithetic_remaining_stocks = sortperm(@views (antithetic_assets[:,end] - antithetic_assets[:,1])./antithetic_assets[:,1])[n1:end-n2-1]

    return 0.5*(Notional + maximum(0,mean(assets[remaining_stocks,end])-K) + Notional + maximum(0,mean(antithetic_assets[antithetic_remaining_stocks,end])-K))*ₑ^(-r*T)

end



function atlas_option_monte_carlo(num_of_sim, α, Notional, T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2)
    
    if method == "basic"
        len = num_of_sim
    elseif method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)
    if method == "basic"
        rtrn = [price_atlas_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_atlas_antithetic_variates(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    end
    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end