using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function price_atlas_normal(Notional::Float64,T::Int, r::Float64, K::Float64, basket_volume::Int, 
                            S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    remaining_stocks = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]

    return Notional * (1+maximum(0,mean(assets[remaining_stocks,end])-K))*ₑ^(-r*T)

end

function price_atlas_LHS(Notional::Float64,T::Int, r::Float64, K::Float64, basket_volume::Int, 
                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int) 
    Z::Matrix{Float64} = randomLHC(basket_volume,T)/10
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]

    return Notional * (1+maximum(0,mean(assets[remaining_stocks,end])-K))*ₑ^(-r*T)

end

function price_atlas_moment_matching(Notional::Float64,T::Int, r::Float64, K::Float64, basket_volume::Int, 
                                    S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
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
    
    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]

    return Notional * (1+maximum(0,mean(assets[remaining_stocks,end])-K))*ₑ^(-r*T)

end

function price_atlas_quasi_monte_carlo(Notional::Float64,T::Int, r::Float64, K::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    s = SobolSeq(basket_volume)
    Z::Matrix{Float64} = reduce(hcat, next!(s) for i = 1:T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]

    return Notional * (1+maximum(0,mean(assets[remaining_stocks,end])-K))*ₑ^(-r*T)

end



function price_atlas_antithetic_variates(Notional::Float64,T::Int, r::Float64, K::Float64, basket_volume::Int, 
                                        S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
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

    remaining_stocks::Array{Int} = sortperm(@views (assets[:,end] - assets[:,1])./assets[:,1])[n1:end-n2-1]
    antithetic_remaining_stocks::Array{Int} = sortperm(@views (antithetic_assets[:,end] - antithetic_assets[:,1])./antithetic_assets[:,1])[n1:end-n2-1]

    return 0.5*(Notional + maximum(0,mean(assets[remaining_stocks,end])-K) + Notional + maximum(0,mean(antithetic_assets[antithetic_remaining_stocks,end])-K))*ₑ^(-r*T)

end



function atlas_option_monte_carlo(num_of_sim::Int, α::Float64,Notional::Float64,T::Int, r::Float64, K::Float64, basket_volume::Int, 
                                S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Array{Float64},n1::Int,n2::Int)
    len = num_of_sim
    if method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)

    if method == "basic"
        rtrn = [price_atlas_normal(Notional,T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_atlas_antithetic_variates(Notional,T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_atlas_quasi_monte_carlo(Notional,T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:num_of_sim]
    elseif method == "moment_matching"
        rtrn = [price_atlas_moment_matching(Notional,T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_atlas_LHS(Notional,T, r, K, basket_volume, S₀, mu, sigma, correlation_matrix,n1,n2) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end