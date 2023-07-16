using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling

function generateBasket(basket_volume::Int,dt::Float64, N::Int,S₀::Array{Float64},mu::Array{Float64},sigma::Array{Float64},epsilon::Matrix{Float64})
    assets::Matrix{Float64} = zeros(basket_volume,N)
    for t in 1:N
        assets[:,t] = S₀.*exp.((mu .- 0.5.*sigma.^2).*(dt) .+ sigma.*sqrt(dt).*epsilon[t,:])
        S₀ = assets[:,t]
    end
    return assets
end

function price_atlas_normal(T::Int,N::Int, r::Float64, K::Float64, basket_volume::Int, 
                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
    remaining_stocks = sortperm(@views (assets[:,end] - S₀)./S₀)[n1:end-n2-1]

    return max(mean(assets[remaining_stocks]) - K, 0.0)

end


function price_atlas_LHS(T::Int, N::Int, r::Float64, K::Float64, basket_volume::Int, 
                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int) 

    dt::Float64 = T/N
    Z::Matrix{Float64} = quantile(Normal(),scaleLHC(randomLHC(N,basket_volume),[(0.001,0.999),(0.001,0.999),(0.001,0.999)])')
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)

    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] - S₀)./S₀)[n1:end-n2-1]

    return max(mean(assets[remaining_stocks]) - K, 0.0)

end

function price_atlas_moment_matching(num_of_sim::Int, α::Float64,T::Int, N::Int, r::Float64, K::Float64, basket_volume::Int, 
                                    S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    dt::Float64 = T/N
    assets_to_optimise::Matrix{Float64} = zeros(basket_volume,num_of_sim)
    d = Normal()
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,N))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
        assets_to_optimise[:,iteration] = assets[:,end]
    end
    S₀df::Array{Float64} = S₀.*ℯ^(r*T)
    remaining_stocks::Matrix{Float64} = assets_to_optimise.*S₀df./mean(assets_to_optimise,dims=2)
    assets_to_calculate::Array{Float64} = zeros(basket_volume-n2-n1,num_of_sim)
    for asst in 1:num_of_sim
        stocks_indices = sortperm(@views ((remaining_stocks[:,asst] .- S₀)./S₀))[n1:end-n2-1]
        assets_to_calculate[:,asst] = remaining_stocks[stocks_indices]
    end
    option_prices::Array{Float64} = mean(assets_to_calculate,dims=1) .- K
    option_prices[option_prices.<=0.0] .= 0.0


    θ::Float64 = mean(option_prices*ℯ^(-r*T))
    s::Float64 = std(option_prices*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]

end

    
function price_atlas_quasi_monte_carlo(T::Int, N::Int, r::Float64, K::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    dt::Float64 = T/N
    s = SobolSeq(0,1)
    Z::Matrix{Float64} = quantile(Normal(),reshape(reduce(hcat, next!(s) for i = 1:N*basket_volume),basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)

    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] -S₀)./S₀)[n1:end-n2-1]

    return max(mean(assets[remaining_stocks]) - K, 0.0)*ℯ^(-r*T)

end



function price_atlas_antithetic_variates(T::Int, N::Int, r::Float64, K::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
    antithetic_assets = generateBasket(basket_volume,dt,N,S₀,mu,sigma,antithetic_delta)

    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] - S₀)./S₀)[n1:end-n2-1]
    antithetic_remaining_stocks::Array{Int} = sortperm(@views (antithetic_assets[:,end] - S₀)./S₀)[n1:end-n2-1]

    return 0.5*((max(mean(assets[remaining_stocks]) - K, 0.0)) + (max(mean(assets[remaining_stocks]) - K, 0.0)))

end



function atlas_option_monte_carlo(num_of_sim::Int, α::Float64,T::Int, N::Int, r::Float64, K::Float64, basket_volume::Int, 
                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int,method="basic")
    len = num_of_sim
    rtrn::Array{Float64,1} = zeros(len)

    if method == "quasi_monte_carlo"
        return price_atlas_quasi_monte_carlo(T, N, r, K, basket_volume, S₀, mu, sigma, correlation_matrix, n1, n2)

    elseif method == "basic"
        rtrn = [price_atlas_normal(T, N, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_atlas_antithetic_variates(T,N, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "LHS"
        rtrn = [price_atlas_LHS(T,N, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn*ℯ^(-r*T))
    s::Float64 = std(rtrn*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end