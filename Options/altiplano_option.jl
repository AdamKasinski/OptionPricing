using LinearAlgebra, Distributions


function simulate_basket_normal(T, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    # 1) sample vector T x basket volume
    # 2) calculate Cholesky Matrix
    # 3) calculate dot product of Cholesky matrix and random sample
    # 4) generate assets with GBM with correlated sample, mu, sigma of every asset price
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end
    return assets
end

function price_altiplano_option(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix)
    # check if any price fall below a threshold
    # if True:
    #   return C
    # else:
    #   return Asian style payment
    
    basket_prices = simulate_basket_normal(T, basket_volume, S₀, mu, sigma, correlation_matrix)
    if any(x->x < treshold, basket_prices./S₀)
        return max(sum(basket_prices[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end

end

function altiplano_option_monte_carlo(num_of_sim, α, T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix)
    
    rtrn::Array{Float64,1} = [price_altiplano_option(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)
end

 
cov_matrix = [1.0 0.3 0.4; 0.3 1.0 0.1; 0.4 0.1 1.0] 

println(altiplano_option_monte_carlo(1000,0.05,10,1,.02,10,10,[1,3,7],3,[10,10,10],[0.03,0.03,0.03],[0.02,0.02,0.02],cov_matrix)) 
    
