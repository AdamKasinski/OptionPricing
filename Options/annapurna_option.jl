using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function generateBasket(basket_volume::Int,T::Int,S₀::Array{Float64},mu::Array{Float64},sigma::Array{Float64},epsilon::Matrix{Float64})
    assets::Matrix{Float64} = zeros(basket_volume,T)
    for t in 1:T
        assets[:,t] = S₀.*exp.((mu .- 0.5.*sigma.^2).*(1/T) .+ sigma.*sqrt(1/T).*epsilon[t,:])
        S₀ = assets[:,t]
    end
    return assets
end

function price_annapurna_normal(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                                basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    if any(x->x < treshold, assets./S₀)
        return max(mean(assets[:,end])*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function price_annapurna_LHS(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                                    basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})

    Z::Matrix{Float64} = scaleLHC(randomLHC(T,basket_volume),[(-1,1),(-1,1),(-1,1)])'
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    if any(x->x < treshold, assets./S₀)
        return max(mean(assets[:,end])*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end

function price_annapurna_antithetic_variates(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                            basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,T))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    antithetic_assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,antithetic_delta)
    if any(x->x < treshold, assets./S₀) || any(x->x < treshold, antithetic_assets./S₀)
        return 0.5*(max(mean(assets[:,end])*ℯ^(-r*T)-K,0) + max(mean(antithetic_assets[:,end])*ℯ^(-r*T)-K,0))
    else
        return C*ℯ^(-r*T)
    end
end

function price_annapurna_quasi_monte_carlo(T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                            basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64},sobolSeq)
    s = sobolSeq
    Z::Matrix{Float64} = reshape(reduce(hcat, next!(s) for i = 1:T*basket_volume),basket_volume,T)
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
    if any(x->x < treshold, assets./S₀)
        return max(mean(assets[:,end])*ℯ^(-r*T)-K,0)
    else
        return C*ℯ^(-r*T)
    end
end


function price_annapurna_moment_matching(num_of_sim::Int,α::Float64,T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
                        basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    
    coupons::Array{Float64} = [] #it is not possible to say what capacity should be 
    assets_to_optimise::Matrix{Float64} = zeros(basket_volume,num_of_sim) #create matrix for pesimistic scenario where each path reach barrier 
    how_many_assets_to_optimise::Int = 0
    d = Normal()
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,T))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,T,S₀,mu,sigma,delta)
        
        if any(x->x < treshold, assets./S₀)
            how_many_assets_to_optimise+=1
            assets_to_optimise[:,how_many_assets_to_optimise] = assets[:,end]
        else
            push!(coupons,C*ℯ^(-r*T))
        end
    end
    
    S₀df::Array{Float64} = S₀.*ℯ^(r*T)
    assets_to_calculate::Matrix{Float64} = assets_to_optimise[:,1:how_many_assets_to_optimise].*S₀df./mean(assets_to_optimise[:,1:how_many_assets_to_optimise],dims=2)
    all_values::Array{Float64} = mean(assets_to_calculate,dims=1)*ℯ^(-r*T) .- K
    all_values[all_values.<=0] .=0
    if length(coupons) !=0
        all_values = hcat(all_values,coupons')
    end
    θ::Float64 = mean(all_values)
    s::Float64 = std(all_values)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end



function annapurna_option_monte_carlo(num_of_sim::Int, α::Float64, T::Int, treshold::Float64, r::Float64, K::Float64, C::Float64,periods::Array{Int}, 
        basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64},method="basic")
    len = num_of_sim
    if method == "antithetic"
        len = Int(round(num_of_sim/2))
    end
    rtrn::Array{Float64,1} = zeros(len)
    sobolSeq = SobolSeq(-1,1)

    if method == "basic"
        rtrn = [price_annapurna_normal(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_annapurna_antithetic_variates(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:Int(round(num_of_sim)/2)]
    elseif method == "quasi_monte_carlo"
        rtrn = [price_annapurna_quasi_monte_carlo(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix,sobolSeq) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_annapurna_LHS(T, treshold, r, K, C,periods, basket_volume, S₀, mu, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end

 
