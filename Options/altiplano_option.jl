using LinearAlgebra, Distributions

function altiplano_option_monte_carlo(C::Float64,mu::Float64, A::Matrix{Float64}, companies::Int64, T::Int64, barrier::Float64, periods::Array{Int64}, S₀::Array{Float64}, K::Float64, 
    num_of_sim::Int64,r::Float64)
    
    rtrn::Array{Float64,1} = zeros(Float32,num_of_sim)
    for iteration in num_of_sim
        
        d = Normal()
        Z::Matrix{Float64} = rand(d,(companies,T)) 
        cholesky_matrix::Matrix{Float64} = cholesky(A).L
        returns::Matrix{Float64} = Z'cholesky_matrix .+mu
        
        if any(x->x < barrier, returns./S₀)
            rtrn[iteration] = max(sum(A[:,periods]./S₀)*ℯ^(-r*T)-K,0)
        else
            rtrn[iteration] = C*ℯ^(-r*T)
        end

    end

    θ::Float64 = mean(rtrn)
    s::Float64 = std(rtrn)
    confidence::Float64 = quantile(Normal(), 1-α/2)
    
    return θ, θ - confidence*s/sqrt(num_of_sim), θ + confidence*s/sqrt(num_of_sim)
end

 
cov_matrix = [1.0 0.3 0.4; 0.3 1.0 0.1; 0.4 0.1 1.0] 

println(altiplano_option_monte_carlo(100.0,3.0,cov_matrix,3,10,0.80,[4,6],[1.0,1.0,1.0],30.0,10,0.02)) 
    
