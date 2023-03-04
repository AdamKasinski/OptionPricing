using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function price_himalayan_normal(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64}) 
    T::Int = basket_volume*250
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)*ℯ^(-r*T) 
end

function price_himalayan_LHS(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    T::Int = basket_volume*250
    Z::Matrix{Float64} = randomLHC(basket_volume,T)/10
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)*ℯ^(-r*T) 
end

function price_himalayan_moment_matching(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    T::Int = basket_volume*250
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

    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)*ℯ^(-r*T) 
end


function price_himalayan_quasi_monte_carlo(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    T::Int = basket_volume*250
    s = SobolSeq(basket_volume)
    Z::Matrix{Float64} = reduce(hcat, next!(s) for i = 1:T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)*ℯ^(-r*T)
end

function price_himalayan_antithetic(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    T::Int = basket_volume*250
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    antithetic_assets::Matrix{Float64} = zeros(basket_volume,T)
    
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
        antithetic_assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(antithetic_delta[1:k-1,asset])) for k in 1:T]
    end
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    highest_values_antithetic::Array{Float64} = zeros(basket_volume)
    available_stocks_antithetic::Array{Int} = [i for i in 1:basket_volume]

    for year in 0:basket_volume-1
        
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        highest_price_antithetic, stock_antithetic = findmax(antithetic_assets[available_stocks_antithetic,year*4+1:year*4+3+1])
        highest_values_antithetic[year+1] = highest_price_antithetic
        filter!(e->e≠stock,available_stocks)
        filter!(e->e≠stock_antithetic,available_stocks_antithetic)
    end

    return 0.5*(mean(highest_values) + mean(highest_values_antithetic))*ℯ^(-r*T)
end


function himalayan_option_monte_carlo(num_of_sim::Int, α::Float64,r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, 
                                    sigma::Array{Float64}, correlation_matrix::Matrix{Float64},method::String)
    
    len = num_of_sim
    if method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)

    if method == "basic"
        rtrn = [price_himalayan_normal(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_himalayan_antithetic_variates(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_himalayan_quasi_monte_carlo(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "moment_matching"
        rtrn = [price_himalayan_moment_matching(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_himalayan_LHS(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end
                                
