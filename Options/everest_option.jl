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

function price_everest_normal(T::Int, N::Int, r::Float64, C::Float64, basket_volume::Int, 
                            S₀::Array{Float64},  sigma::Array{Float64}, correlation_matrix::Array{Float64})
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    worst_performing::Int = argmin(@views (assets[:,end])./S₀)

    return (C+assets[worst_performing,end])
    
end

function price_everest_LHS(T::Int, N::Int, r::Float64, C::Float64, basket_volume::Int, 
                        S₀::Array{Float64},  sigma::Array{Float64}, correlation_matrix::Array{Float64})

    dt::Float64 = T/N
    Z::Matrix{Float64} = quantile(Normal(),scaleLHC(randomLHC(N,basket_volume),[(0.001,0.999),(0.001,0.999),(0.001,0.999)])')
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    worst_performing::Int = argmin(@views (assets[:,end])./S₀)

    return (C+assets[worst_performing,end])
    
end

function price_everest_moment_matching(num_of_sim::Int, α::Float64,T::Int, N::Int, r::Float64, C::Float64, basket_volume::Int, 
                                        S₀::Array{Float64},  sigma::Array{Float64}, correlation_matrix::Array{Float64})
    dt::Float64 = T/N
    assets_to_optimise::Array{Float64} = zeros(num_of_sim)
    d = Normal()
    worst_performing::Int = 0;
    for iteration in 1:num_of_sim
        Z::Matrix{Float64} = rand(d,(basket_volume,N))
        cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
        delta::Matrix{Float64} = Z'cholesky_matrix
        assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
        worst_performing = argmin(@views (assets[:,end])./S₀)
        assets_to_optimise[iteration] = assets[worst_performing,end]
    end
    S₀df::Float64 = S₀[1]*ℯ^(r*T)
    assets_to_calculate::Array{Float64} = assets_to_optimise.*S₀df./mean(assets_to_optimise,dims=1)
    all_values::Array{Float64} = (C.+assets_to_calculate)
    θ::Float64 = mean(all_values*ℯ^(-r*T))
    s::Float64 = std(all_values*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)
    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
    
end


#cov_matrix = [1.0 0.3 0.4; 0.3 1.0 0.1; 0.4 0.1 1.0]
#print(price_everest_moment_matching(2,.05,1,250,.03,9.0,3,[10.0,10.0,10.0],[0.02,0.02,0.02],cov_matrix))



function price_everest_quasi_monte_carlo(T::Int, N::Int, r::Float64, C::Float64, basket_volume::Int, 
                                        S₀::Array{Float64},  sigma::Array{Float64}, correlation_matrix::Array{Float64})
    dt::Float64 = T/N
    s = SobolSeq(0,1)
    Z::Matrix{Float64} = quantile(Normal(),reshape(reduce(hcat, next!(s) for i = 1:N*basket_volume),basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    assets::Matrix{Float64} =  generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)

    worst_performing::Int = argmin(@views (assets[:,end])./S₀)

    return (C+assets[worst_performing,end])*ℯ^(-r*T)
    
end

function price_everest_antithetic_variates(T::Int,N::Int, r::Float64, C::Float64, basket_volume::Int, 
                                            S₀::Array{Float64},  sigma::Array{Float64}, correlation_matrix::Array{Float64})
    dt::Float64 = T/N
    d = Normal()
    Z::Matrix{Float64} = rand(d,(basket_volume,N))
    cholesky_matrix::Matrix{Float64} = cholesky(correlation_matrix).L
    delta::Matrix{Float64} = Z'cholesky_matrix
    antithetic_delta::Matrix{Float64} = -Z'cholesky_matrix
    assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,delta)
    antithetic_assets::Matrix{Float64} = generateBasket(basket_volume,dt,N,S₀,r,sigma,antithetic_delta)


    worst_performing::Int = argmin(@views (assets[:,end])./S₀)
    antithetic_worst_performing::Int = argmin(@views (assets[:,end])./S₀)

    return C+0.5*(assets[worst_performing,end]+assets[antithetic_worst_performing,end])

end


function everest_option_monte_carlo(num_of_sim::Int, α::Float64,T::Int,N::Int, r::Float64, C::Float64, basket_volume::Int, 
                                S₀::Array{Float64},  sigma::Array{Float64}, correlation_matrix::Array{Float64},method="basic")
    len = num_of_sim
    rtrn::Array{Float64,1} = zeros(len)

    if method == "quasi_monte_carlo"
        return price_everest_quasi_monte_carlo(T, N, r, C, basket_volume, S₀, sigma, correlation_matrix)
    elseif method == "basic"
        rtrn = [price_everest_normal(T, N, r, C, basket_volume, S₀, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "antithetic"
        rtrn = [price_everest_antithetic_variates(T, N, r, C, basket_volume, S₀, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    elseif method == "LHS"
        rtrn = [price_everest_LHS(T, N, r, C, basket_volume, S₀, sigma, correlation_matrix) for iteration in 1:num_of_sim]
    else
        return "no method found"
    end

    θ::Float64 = mean(rtrn*ℯ^(-r*T))
    s::Float64 = std(rtrn*ℯ^(-r*T))
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return [θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)]
end
