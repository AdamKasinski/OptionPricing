∑(X::Vector{Float64}) = sum(X)

function asian_option_monte_carlo(S₀::Float64, num_of_sim::Int64, T::Int64, μ::Float64, σ::Float64,K::Float64,r::Float64,α::Float64)

    Sₖ::Array{Float64,1} = zeros(Float64,num_of_sim)

    d = Normal()
    
    for iteration in 1:num_of_sim
        Random.seed!(iteration)
        b::Vector{Float64} = rand(d, T)
        asset_prices::Vector{Float64} = [S₀ * exp((μ - 0.5 * σ^2) * k + σ * ∑(b[1:k-1])) for k in 1:T]
        Sₖ[iteration] = max(mean(asset_prices)-K,0)*ℯ^(-r*T) #TO DO: add variable: moments when value of asset is frozen and took to average calculation
    end


    θ::Float64 = mean(Sₖ)
    s::Float64 = std(Sₖ)
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)
end

function asian_option_antithetic_variates(S₀::Float64, num_of_sim::Int64, T::Int64, μ::Float64, σ::Float64,K::Float64,r::Float64,α::Float64)

    Sₖ::Array{Float64,1} = zeros(Float64,Int(num_of_sim/2))

    d = Normal()
    
    q = Int(num_of_sim/2)
    for iteration in 1:q
        Random.seed!(iteration)
        b::Vector{Float64} = rand(d, T)
        c::Vector{Float64} = -b#[1-i for i in b]
        asset_prices::Vector{Float64} = [S₀ * exp((μ - 0.5 * σ^2) * k + σ * ∑(b[1:k-1])) for k in 1:T]
        antithetic_asset_prices::Vector{Float64} = [S₀ * exp((μ - 0.5 * σ^2) * k + σ * ∑(c[1:k-1])) for k in 1:T]
        Sₖ[iteration] = 0.5*(max(mean(asset_prices)-K,0)*ℯ^(-r*T)+max(mean(antithetic_asset_prices)-K,0)*ℯ^(-r*T))
    end


    θ::Float64 = mean(Sₖ)
    s::Float64 = std(Sₖ)
    confidence::Float64 = quantile(Normal(), 1-α/2)

    return θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)

end