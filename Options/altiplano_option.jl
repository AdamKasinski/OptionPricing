using LinearAlgebra, Distributions, Statistics


function price_altiplano_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    if any(x->x < treshold, assets./S₀)
        return max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function price_altiplano_antithetic_variates(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
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

    if any(x->x < treshold, assets./S₀) || any(x->x < treshold, antithetic_assets./S₀)
        return 0.5*(max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0) + max(sum(antithetic_assets[:,periods]./S₀)*ℯ^(-r*T)-K,0))
    else
        return C*ℯ^(-r*T)
    end
end


function altiplano_option_monte_carlo(num_of_sim, α, T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix,method)
    
    if method == "basic"
        len = num_of_sim
    elseif method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)
    if method == "basic"
        rtrn = [price_altiplano_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_altiplano_antithetic_variates(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    end
    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end

 
cov_matrix = [1.0 0.3 0.4; 0.3 1.0 0.1; 0.4 0.1 1.0] 

normal = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.02,10,10,[1,3,7],3,[10,10,10],[0.02,0.03,0.03],[0.03,0.03,0.03],cov_matrix,"basic") for i in 1:10]
antithetic = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.02,10,10,[1,3,7],3,[10,10,10],[0.03,0.03,0.03],[0.03,0.03,0.03],cov_matrix,"antithetic") for i in 1:10]

mean_elements = mean([normal[i][1] for i in eachindex(normal)])
bottom_elements = mean([normal[i][2] for i in eachindex(normal)])
upper_elements = mean([normal[i][3] for i in eachindex(normal)])

mean_antithetic = mean([antithetic[i][1] for i in eachindex(antithetic)])
bottom_antithetic = mean([antithetic[i][2] for i in eachindex(antithetic)])
upper_antithetic = mean([antithetic[i][3] for i in eachindex(antithetic)])

length_normal = upper_elements-bottom_elements
length_antithetic = upper_antithetic - bottom_antithetic
println("basic mean : $mean_elements length: $length_normal")
println("antithetic mean : $mean_antithetic length_antithetic: $length_antithetic")
