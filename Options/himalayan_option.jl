using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling

function generateBasket(basket_volume::Int,T::Int,S₀::Array{Float64},mu::Array{Float64},sigma::Array{Float64},epsilon::Matrix{Float64})
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for t in 1:T
        assets[:,t] = S₀.*exp.((mu .- 0.5.*sigma.^2).*(1/T) .+ sigma.*sqrt(1/T).*epsilon[t,:])
        S₀ = assets[:,t]
    end
    return assets
end

function price_himalayan_normal(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64}) 
    T::Int = basket_volume*250
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    
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
    Z::Matrix{Float64} = scaleLHC(randomLHC(T,basket_volume),[(-1,1),(-1,1),(-1,1)])'
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    
    highest_values::Array{Float64} = zeros(basket_volume)
    available_stocks::Array{Int} = [i for i in 1:basket_volume]
    
    for year in 0:basket_volume-1
        highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
        highest_values[year+1] = highest_price
        filter!(e->e≠stock,available_stocks)
    end

    return mean(highest_values)*ℯ^(-r*T) 
end

function price_himalayan_moment_matching(num_of_sim::Int,α::Float64, r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, 
                                        correlation_matrix::Matrix{Float64})
    
    T::Int = basket_volume*250
    assets_to_optimise::Array{Float64} = zeros(num_of_sim)
    S₀df::Array{Float64} = S₀.*ℯ^(r*T)
    d = Normal()
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,T))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta).*S₀df
        highest_values::Array{Float64} = zeros(basket_volume)
        available_stocks::Array{Int} = [i for i in 1:basket_volume]
        for year in 0:basket_volume-1
            highest_price, stock = findmax(assets[available_stocks,year*4+1:year*4+3+1])
            highest_values[year+1] = highest_price
            filter!(e->e≠stock,available_stocks)
        end
        assets_to_optimise[iteration] = mean(highest_values)#*ℯ^(-r*T)
    end
    #assets_to_calculate = assets_to_optimise
    assets_to_calculate::Array{Float64} = (assets_to_optimise./mean(assets_to_optimise))*ℯ^(-r*T)
    θ::Float64 = mean(assets_to_calculate)
    s::Float64 = std(assets_to_calculate)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
    
    
    
end


function price_himalayan_quasi_monte_carlo(r::Float64, basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64},sobolSeq)
    T::Int = basket_volume*250
    s = sobolSeq
    Z::Matrix{Float64} = reshape(reduce(hcat, next!(s) for i = 1:T*basket_volume),basket_volume,T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    
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
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    antithetic_assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,antithetic_delta)
    
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
    sobolSeq = SobolSeq(-1,1)

    if method == "basic"
        rtrn = [price_himalayan_normal(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_himalayan_antithetic(r, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_himalayan_quasi_monte_carlo(r, basket_volume, S₀, mu, sigma, correlation_matrix,sobolSeq) for iteration in 1:num_of_sim]
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
                                
