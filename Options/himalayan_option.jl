using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling

function generateBasket(basket_volume::Int,dt::Float64, N::Int,S₀::Array{Float64},r::Float64,
                        sigma::Array{Float64},epsilon::Matrix{Float64})
    assets::Matrix{Float64} = zeros(basket_volume,N)
    for t in 1:N
        assets[:,t] = S₀.*exp.((r .- 0.5.*sigma.^2).*(dt) .+ sigma.*sqrt(dt).*epsilon[t,:])
        S₀ = assets[:,t]
    end
    return assets
end

function price_himalayan_normal(r::Float64, basket_volume::Int, S₀::Array{Float64}, K::Float64,  sigma::Array{Float64}, correlation_matrix::Matrix{Float64}) 
    T::Int = basket_volume
    N::Int = basket_volume*250
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 1:basket_volume
        highest_price, stock = findmax(assets[available_stocks,year*250])
        highest_values[year] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return max(mean(highest_values) - K, 0.0)
end

function price_himalayan_LHS(r::Float64, basket_volume::Int, S₀::Array{Float64}, K::Float64,  sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    T::Int = basket_volume
    N::Int = basket_volume*250
    dt::Float64 = T/N
    Z::Matrix{Float64} = quantile(Normal(),scaleLHC(randomLHC(N,basket_volume),[(0.001,0.999),(0.001,0.999),(0.001,0.999)])')
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 1:basket_volume
        highest_price, stock = findmax(assets[available_stocks,year*250])
        highest_values[year] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return max(mean(highest_values) - K, 0.0)
end

function price_himalayan_moment_matching(num_of_sim::Int,α::Float64, r::Float64, basket_volume::Int, S₀::Array{Float64}, K::Float64,  sigma::Array{Float64}, 
                                        correlation_matrix::Matrix{Float64})
    
    T::Int = basket_volume
    N::Int = basket_volume*250
    dt::Float64 = T/N
    assets_to_optimise::Matrix{Float64} = zeros(basket_volume,num_of_sim)
    S₀df::Array{Float64} = S₀.*ℯ^(r*T)
    d = Normal()
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,N))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
        highest_values::Array{Float64} = zeros(basket_volume)
        available_stocks::Array{Int} = [i for i in 1:basket_volume]
        for year in 1:basket_volume
            highest_price, stock = findmax(assets[available_stocks,year*250])
            highest_values[year] = highest_price
            filter!(e->e≠stock,available_stocks)
        end
        assets_to_optimise[:,iteration] = highest_values
    end

    assets_to_calculate::Matrix{Float64} = assets_to_optimise*S₀df[1]./mean(assets_to_optimise,dims=2) 
    
    option_prices::Array{Float64} = mean(assets_to_calculate,dims=1) .- K
    option_prices[option_prices.<=0.0] .= 0.0

    θ::Float64 = mean(option_prices*ℯ^(-r*T))
    s::Float64 = std(option_prices*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
    
    
    
end


function price_himalayan_quasi_monte_carlo(r::Float64, basket_volume::Int, S₀::Array{Float64}, K::Float64,  sigma::Array{Float64}, 
                                            correlation_matrix::Matrix{Float64})

    T::Int = basket_volume
    N::Int = basket_volume*250
    dt::Float64 = T/N
    s = SobolSeq(0,1)
    Z::Matrix{Float64} = quantile(Normal(),reshape(reduce(hcat, next!(s) for i = 1:N*basket_volume),basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 1:basket_volume
        highest_price, stock = findmax(assets[available_stocks,year*250])
        highest_values[year] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return max(mean(highest_values) - K, 0.0)*ℯ^(-r*T)
end

function price_himalayan_antithetic(r::Float64, basket_volume::Int, S₀::Array{Float64}, K::Float64,  sigma::Array{Float64}, 
                                    correlation_matrix::Matrix{Float64})
    
    T::Int = basket_volume
    N::Int = basket_volume*250
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    antithetic_assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,antithetic_delta)
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    highest_values_antithetic::Array{Float64} = zeros(basket_volume)
    available_stocks_antithetic::Array{Int} = [i for i in 1:basket_volume]

    for year in 1:basket_volume
        
        highest_price, stock = findmax(assets[available_stocks,year*250])
        highest_values[year] = highest_price
        highest_price_antithetic, stock_antithetic = findmax(antithetic_assets[available_stocks_antithetic,year*250])
        highest_values_antithetic[year] = highest_price_antithetic
        filter!(e->e≠stock,available_stocks)
        filter!(e->e≠stock_antithetic,available_stocks_antithetic)
    end

    highest_values = highest_values
    
    highest_values_antithetic = highest_values_antithetic

    return 0.5*(max(mean(highest_values) - K, 0.0) + max(mean(highest_values) - K, 0.0))
end


function himalayan_option_monte_carlo(num_of_sim::Int, α::Float64,r::Float64, basket_volume::Int, S₀::Array{Float64}, K::Float64,  
                                    sigma::Array{Float64}, correlation_matrix::Matrix{Float64},method::String)
    
    len = num_of_sim
    
    rtrn::Array{Float64,1} = zeros(len)
    
    T::Int = basket_volume

    if method == "quasi_monte_carlo"
        return price_himalayan_quasi_monte_carlo(r, basket_volume, S₀, K,   sigma, correlation_matrix)
    elseif method == "basic"
        rtrn = [price_himalayan_normal(r, basket_volume, S₀, K,   sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_himalayan_antithetic(r, basket_volume, S₀, K,   sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_himalayan_LHS(r, basket_volume, S₀, K,   sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn*ℯ^(-r*T))
    s::Float64 = std(rtrn*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end
                                
