using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function price_altiplano_normal(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                                basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    if any(x->x > treshold, assets./S₀)
        return max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function price_altiplano_LHS(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                                    basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    Z::Matrix{Float64} = randomLHC(basket_volume,T)/10
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    if any(x->x > treshold, assets./S₀)
        return max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end


function price_altiplano_antithetic_variates(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                            basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
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

    if any(x->x > treshold, assets./S₀) || any(x->x < treshold, antithetic_assets./S₀)
        return 0.5*(max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0) + max(sum(antithetic_assets[:,periods]./S₀)*ℯ^(-r*T)-K,0))
    else
        return C*ℯ^(-r*T)
    end
end

function price_altiplano_quasi_monte_carlo(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                            basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    s = SobolSeq(basket_volume)
    Z::Matrix{Float64} = reduce(hcat, next!(s) for i = 1:T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    if any(x->x > treshold, assets./S₀)
        return max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function price_altiplano_moment_matching(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                        basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
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
    assets = assets.*S₀df./mean(assets,dims=2)

    if any(x->x > treshold, assets./S₀)
        return max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function altiplano_option_monte_carlo(num_of_sim::Int, α::Float64, T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
        basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64},method="basic")
    len = num_of_sim
    if method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)

    if method == "basic"
        rtrn = [price_altiplano_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_altiplano_antithetic_variates(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_altiplano_quasi_monte_carlo(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "moment_matching"
        rtrn = [price_altiplano_moment_matching(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_altiplano_LHS(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end

 
cov_matrix = [1.0 0.3 0.4; 0.3 1.0 0.1; 0.4 0.1 1.0] 

normal = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.01,10.0,10.0,[1,3,7],3,[10.0,10.0,10.0],[0.07,0.07,0.07],[0.06,0.05,0.09],cov_matrix,"basic") for i in 1:10]
antithetic = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.01,10.0,10.0,[1,3,7],3,[10.0,10.0,10.0],[0.07,0.07,0.07],[0.06,0.05,0.09],cov_matrix,"antithetic") for i in 1:10]
quasi = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.01,10.0,10.0,[1,3,7],3,[10.0,10.0,10.0],[0.07,0.07,0.07],[0.06,0.05,0.09],cov_matrix,"quasi_monte_carlo") for i in 1:10]
moment_matching = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.01,10.0,10.0,[1,3,7],3,[10.0,10.0,10.0],[0.07,0.07,0.07],[0.06,0.05,0.09],cov_matrix,"moment_matching") for i in 1:10]
LHS = [altiplano_option_monte_carlo(1000,0.05,10,0.9,0.01,10.0,10.0,[1,3,7],3,[10.0,10.0,10.0],[0.07,0.07,0.07],[0.06,0.05,0.09],cov_matrix,"LHS") for i in 1:10]

mean_elements = mean([normal[i][1] for i in eachindex(normal)])
bottom_elements = mean([normal[i][2] for i in eachindex(normal)])
upper_elements = mean([normal[i][3] for i in eachindex(normal)])

mean_antithetic = mean([antithetic[i][1] for i in eachindex(antithetic)])
bottom_antithetic = mean([antithetic[i][2] for i in eachindex(antithetic)])
upper_antithetic = mean([antithetic[i][3] for i in eachindex(antithetic)])

mean_quasi = mean([quasi[i][1] for i in eachindex(quasi)])
bottom_quasi = mean([quasi[i][2] for i in eachindex(quasi)])
upper_quasi = mean([quasi[i][3] for i in eachindex(quasi)])

mean_moment_matching = mean([moment_matching[i][1] for i in eachindex(moment_matching)])
bottom_moment_matching = mean([moment_matching[i][2] for i in eachindex(moment_matching)])
upper_moment_matching = mean([moment_matching[i][3] for i in eachindex(moment_matching)])

mean_LHS = mean([LHS[i][1] for i in eachindex(LHS)])
bottom_LHS = mean([LHS[i][2] for i in eachindex(LHS)])
upper_LHS = mean([LHS[i][3] for i in eachindex(LHS)])

length_normal = upper_elements-bottom_elements
length_antithetic = upper_antithetic - bottom_antithetic
length_quasi = upper_quasi-bottom_quasi
length_moment_matching = upper_moment_matching - bottom_moment_matching
length_LHS = upper_LHS-bottom_LHS


println("basic mean : $mean_elements length: $length_normal")
println("antithetic mean : $mean_antithetic length_antithetic: $length_antithetic")
println("quasi mean : $mean_quasi length: $length_quasi")
println("moment matching mean : $mean_moment_matching length_antithetic: $length_moment_matching")
println("LHS mean : $mean_LHS length_antithetic: $length_LHS")