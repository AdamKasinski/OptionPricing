using LinearAlgebra, Distributions, Statistics, Sobol

function price_himalayan_normal(r, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    #
    T = basket_volume*250
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end
    
    highest_values = zeros(basket_volume)
    available_stocks = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)#*ℯ^(-r*T) #TODO
end


function price_himalayan_quasi_monte_carlo(r, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    #
    T = basket_volume*250
    s = SobolSeq(basket_volume)
    Z = reduce(hcat, next!(s) for i = 1:T)
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end
    
    highest_values = zeros(basket_volume)
    available_stocks = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)#*ℯ^(-r*T) #TODO
end

function price_himalayan_antithetic(r, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    
    T = basket_volume*250
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    delta_antithetic = -Z*cholesky_matrix
    assets = zeros(basket_volume,T)
    antithetic_assets = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
        antithetic_assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(antithetic_delta[1:k-1,asset])) for k in 1:T]
    end
    
    highest_values = zeros(basket_volume)
    available_stocks = [i for i in 1:basket_volume]
    
    highest_values_antithetic = zeros(basket_volume)
    available_stocks_antithetic = [i for i in 1:basket_volume]

    for year in 0:basket_volume-1
        
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        highest_price_antithetic, stock_antithetic = findmax(antithetic_assets[available_stocks_antithetic,year*4+1:year*4+3+1])
        highest_values_antithetic[year+1] = highest_price_antithetic
        filter!(e->e≠stock,available_stocks)
        filter!(e->e≠stock_antithetic,available_stocks_antithetic)
    end

    return 0.5*(mean(highest_values) + mean(highest_values_antithetic))#*ℯ^(-r*T) #TODO
end


function himalayan_option_monte_carlo(num_of_sim, α, r, basket_volume, S₀, mu, sigma, correlation_matrix,method)
    
    if method == "basic"
        len = num_of_sim
    elseif method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)
    if method == "basic"
        rtrn = [price_himalayan_normal(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_himalayan_normal(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    end
    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end



cov_matrix = [1.0 0.3 0.4; 0.3 1.0 0.1; 0.4 0.1 1.0] 

himalayan_option_monte_carlo(100,.05,.02,3,[10,10,10],[.02,.02,.02],[.002,.002,.002],cov_matrix,"basic")