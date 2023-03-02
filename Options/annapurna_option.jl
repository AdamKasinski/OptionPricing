using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function price_annapurna_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
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


function price_annapurna_LHS(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    Z = randomLHC(T,basket_volume)/10
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


function price_annapurna_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
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

function price_annapurna_moment_matching(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    d = Normal()
    Z = rand(d,(basket_volume,T))
    cholesky_matrix = cholesky(correlation_matrix).L
    delta = Z'cholesky_matrix
    assets = zeros(basket_volume,T)
    for asset in 1:basket_volume
        assets[asset,:] = [S₀[asset] * exp((mu[asset] - 0.5 * sigma[asset]^2) * k + sigma[asset] * sum(delta[1:k-1,asset])) for k in 1:T] # if dt != 1 the formula will be changed
    end

    S₀s = reshape(repeat(assets[:,1],T),basket_volume,T)
    S₀df = S₀s.*[ℯ^(-r*t) for t in 1:T]'
    assets =  assets.*S₀df./mean(assets,dims=2)
    
    if any(x->x < treshold, assets./S₀)
        return max(sum(assets[:,periods]./S₀)*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function price_annapurna_quasi_monte_carlo(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
    s = SobolSeq(basket_volume)
    Z = reduce(hcat, next!(s) for i = 1:T)
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

function price_annapurna_antithetic_variates(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) # TODO: add types
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


function annapurna_option_monte_carlo(num_of_sim, α, T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix,method)
    
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

 

