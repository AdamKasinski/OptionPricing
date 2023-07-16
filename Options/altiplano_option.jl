using LinearAlgebra, Distributions, Statistics, Sobol, LatinHypercubeSampling


function generateBasket(basket_volume::Int,dt::Float64, N::Int,S₀::Array{Float64},mu::Array{Float64},
                        sigma::Array{Float64},epsilon::Matrix{Float64})
    assets::Matrix{Float64} = zeros(basket_volume,N)
    for t in 1:N
        assets[:,t] = S₀.*exp.((mu .- 0.5.*sigma.^2).*(dt) .+ sigma.*sqrt(dt).*epsilon[t,:])
        S₀ = assets[:,t]
    end
    return assets
end

function price_altiplano_normal(T::Int, N::Int, treshold::Float64, r::Float64, K::Float64, C::Float64, 
                    basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, 
                    correlation_matrix::Matrix{Float64})
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
    if any(x->x > treshold, assets./S₀)
        return max(mean(assets[:,end])-K,0)
    else
        return C
    end
end


function price_altiplano_LHS(T::Int,N::Int, treshold::Float64, r::Float64, K::Float64, C::Float64, 
                basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, 
                correlation_matrix::Matrix{Float64})

    dt::Float64 = T/N
    Z::Matrix{Float64} = quantile(Normal(),scaleLHC(randomLHC(N,basket_volume),[(0.0001,0.9999),
                                                                (0.0001,0.9999), (0.0001,0.9999)])')
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
    if any(x->x > treshold, assets./S₀)
        return max(mean(assets[:,end])-K,0)
    else
        return C
    end
end

function price_altiplano_antithetic_variates(T::Int, N::Int, treshold::Float64, r::Float64, K::Float64, 
                                            C::Float64, basket_volume::Int, S₀::Array{Float64}, 
                                            mu::Array{Float64}, sigma::Array{Float64}, 
                                            correlation_matrix::Matrix{Float64})

    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
    antithetic_assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,antithetic_delta)
    if any(x->x > treshold, assets./S₀) || any(x->x > treshold, antithetic_assets./S₀)
        return 0.5*(max(mean(assets[:,end])-K,0) + max(mean(antithetic_assets[:,end])-K,0))
    else
        return C
    end
end

function price_altiplano_quasi_monte_carlo(T::Int,N::Int, treshold::Float64, r::Float64, 
                                    K::Float64, C::Float64, basket_volume::Int, S₀::Array{Float64}, 
                                    mu::Array{Float64}, sigma::Array{Float64}, 
                                    correlation_matrix::Matrix{Float64})
    dt::Float64 = T/N
    sobolSeq = SobolSeq(0,1)
    Z::Matrix{Float64} = quantile(Normal(),reshape(reduce(hcat, next!(sobolSeq) for i = 1:N*basket_volume),
                basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
    if any(x->x > treshold, assets./S₀)
        return max(mean(assets[:,end])-K,0)*ℯ^(-r*T)
    else
        return C*ℯ^(-r*T)
    end
end

function price_altiplano_moment_matching(num_of_sim::Int,α::Float64,T::Int,N::Int, treshold::Float64, r::Float64, K::Float64, C::Float64, 
                        basket_volume::Int, S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, correlation_matrix::Matrix{Float64})
    dt::Float64 = T/N
    coupons::Array{Float64} = [] #it is not possible to say what capacity should be 
    assets_to_optimise::Matrix{Float64} = zeros(basket_volume,num_of_sim) #create matrix for pesimistic scenario where each path reach barrier 
    how_many_assets_to_optimise::Int = 0
    d = Normal()
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,N))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,mu,sigma,delta)
        
        if any(x->x > treshold, assets./S₀)
            how_many_assets_to_optimise+=1
            assets_to_optimise[:,how_many_assets_to_optimise] = assets[:,end]
        else
            push!(coupons,C)
        end
    end
    
    S₀df::Array{Float64} = S₀*ℯ^(r*T)
    assets_to_calculate::Matrix{Float64} = assets_to_optimise[:,1:how_many_assets_to_optimise].*S₀df./mean(assets_to_optimise[:,1:how_many_assets_to_optimise],dims=2)
    all_values::Array{Float64} = mean(assets_to_calculate,dims=1) .- K
    all_values[all_values.<=0] .=0
    if length(coupons) !=0
        all_values = hcat(all_values,coupons')
    end
    θ::Float64 = mean(all_values*ℯ^(-r*T))
    s::Float64 = std(all_values*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end

function altiplano_option_monte_carlo(num_of_sim::Int, α::Float64, T::Int, N::Int, treshold::Float64, 
                                    r::Float64, K::Float64, C::Float64, basket_volume::Int, 
                                    S₀::Array{Float64}, mu::Array{Float64}, sigma::Array{Float64}, 
                                    correlation_matrix::Matrix{Float64}, method="basic")
    len = num_of_sim

    rtrn::Array{Float64,1} = zeros(len)

    if method == "quasi_monte_carlo"
        return price_altiplano_quasi_monte_carlo(T, N, treshold, r, K, C, basket_volume, S₀, mu, sigma, 
                                                                                    correlation_matrix)
    elseif method == "basic"
        rtrn = [price_altiplano_normal(T, N, treshold, r, K, C, basket_volume, S₀, mu, sigma, 
                                        correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_altiplano_antithetic_variates(T,N, treshold, r, K, C, basket_volume, S₀, mu, sigma, 
                                                    correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_altiplano_LHS(T,N, treshold, r, K, C, basket_volume, S₀, mu, sigma, 
                                    correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn*ℯ^(-r*T))
    s::Float64 = std(rtrn*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end

