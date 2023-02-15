using LinearAlgebra
using QuantumOptics
using SparseArrays
using Arpack
using PyPlot
using Distributions
using Statistics
using DelimitedFiles
using ForwardDiff
using QuadGK
using Roots
using Interpolations
using DifferentialEquations
using FFTW
using CSV
using Plots
using Folds
using BenchmarkTools
using Evolutionary

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 12;
rcParams["font.sans-serif"] = "Arial";
rcParams["figure.figsize"] = (2,2);

# Fundamental constants
ħ = 1.05e-34;
c_light = 3e8;
ϵ0 = 8.85e-12;

mutable struct beamsplitter
    modes; 
    t::Float64;
end

mutable struct fiber
    fiber_mode::Int64;
    fiber_index::Float64; 
    fiber_length::Float64;
    fiber_dispersion::Float64;
    fiber_nonlinearity::Float64;
end

mutable struct sim
    num_modes::Int64
    N_z::Int64
    z_grid #k_grid is derived from z-grid
end

mutable struct state
    mean_fields
    V_matrix ## the V matrix
end
     

function fftfreq(n,fs)
    if mod(n,2) == 0
        freq = [0:n÷2-1; -n÷2:-1]  * fs/n; # if n is even
        else 
        freq = [0:(n-1)÷2; -(n-1)÷2:-1]  * fs/n  # if n is odd
    end

    return 2*pi*freq
end

function get_row_index(i,σ)
    index = (σ-1)*N_z + i;
    return index
end

function get_column_index(i,σ)
    index = (σ-1)*2*N_z + i;
    return index
end

function get_ranges(mode_indices)
    m_i = mode_indices[1];
    m_j = mode_indices[2];
    index_i_init = (m_i - 1)*N_z + 1;
    index_i_final = (m_i)*N_z;
    index_j_init = 2*(m_j - 1)*N_z + 1;
    index_j_final = 2*(m_j)*N_z;
    ranges = index_i_init:index_i_final,index_j_init:index_j_final
    return ranges
end

function get_submatrix(mode_indices,V)
    m_i = mode_indices[1];
    m_j = mode_indices[2];
    index_i_init = (m_i - 1)*N_z + 1;
    index_i_final = (m_i)*N_z;
    index_j_init = 2*(m_j - 1)*N_z + 1;
    index_j_final = 2*(m_j)*N_z;
    M = V[index_i_init:index_i_final,index_j_init:index_j_final];
    return M
end

function get_μνmatrix(V,sim)
    N_z = sim.N_z;
    N_σ = sim.num_modes
    μ_mat = 1.0im*zeros(N_z*N_σ,N_z*N_σ);
    ν_mat = 1.0im*zeros(N_z*N_σ,N_z*N_σ);
    for ii=1:N_z
        for jj=1:N_z
            for ss = 1:N_σ
                for rr = 1:N_σ
                    μ_mat[N_z*(ss-1)+ii,N_z*(rr-1)+jj] = V[N_z*(ss-1)+ii,2*N_z*(rr-1)+jj];
                    ν_mat[N_z*(ss-1)+ii,N_z*(rr-1)+jj] = V[N_z*(ss-1)+ii,2*N_z*(rr-1)+jj+N_z]
                end
            end
        end
    end
    return μ_mat,ν_mat
end


function check_commutator(state,sim)
    V = state.V_matrix;
    μ,ν = get_μνmatrix(V,sim);
    val = norm(μ*adjoint(μ) - ν*adjoint(ν) - UniformScaling(1.0));
    return val
end


function vacuum_V(sim)
    N_σ = sim.num_modes;
    N_z = sim.N_z;
    
    V_matrix = 1.0im*zeros(N_z*N_σ,2*N_z*N_σ);

    for ii = 1:N_z
        for ss = 1:N_σ
            M_matrix = get_submatrix([ss,ss],V_matrix)
            M_matrix[ii,ii] = 1.0+0.0*im;
            V_matrix[get_ranges([ss,ss])[1],get_ranges([ss,ss])[2]] .= M_matrix;
        end
    end

    V_init = copy(V_matrix);
    
    return V_init
end

function n2_exp(state,mode,sim)
    
    α = get_meanfield_i(state,mode)
    V = state.V_matrix;
    μ,ν = get_μνmatrix(V,sim)
    N_z = sim.N_z;
    
    term1 = transpose(conj.(α))*(μ*transpose(ν))[1:N_z,1:N_z]*(conj.(α));
    term2 = transpose(conj.(α))*(μ*adjoint(μ))[1:N_z,1:N_z]*(α);
    term3 = transpose(α)*(conj.(ν)*transpose(ν))[1:N_z,1:N_z]*conj.(α);
    term4 = transpose(α)*(conj.(ν)*adjoint(μ))[1:N_z,1:N_z]*α;
    
    n2 = term1 + term2 + term3 + term4;
    return n2
end

function x2_exp(state,mode,sim)
    
    α = get_meanfield_i(state,mode)
    V = state.V_matrix;
    μ,ν = get_μνmatrix(V,sim)
    N_z = sim.N_z;
    
    term1 = (μ*transpose(ν))[1:N_z,1:N_z];
    term2 = (μ*adjoint(μ))[1:N_z,1:N_z];
    term3 = (conj.(ν)*transpose(ν))[1:N_z,1:N_z];
    term4 = (conj.(ν)*adjoint(μ))[1:N_z,1:N_z];
    
    x2 = term1 + term2 + term3 + term4;
    return x2
end

function p2_exp(state,mode,sim)
    
    α = get_meanfield_i(state,mode)
    V = state.V_matrix;
    μ,ν = get_μνmatrix(V,sim)
    N_z = sim.N_z;
    
    term1 = (μ*transpose(ν))[1:N_z,1:N_z];
    term2 = -(μ*adjoint(μ))[1:N_z,1:N_z];
    term3 = -(conj.(ν)*transpose(ν))[1:N_z,1:N_z];
    term4 = (conj.(ν)*adjoint(μ))[1:N_z,1:N_z];
    
    p2=term1 + term2 + term3 + term4;
    return -p2
end

function BS_meanfield(state,sim,BS)
    t = BS.t;
    r = sqrt(1-t^2);
    N_z = sim.N_z;
    modes = BS.modes;
    range_α = get_row_index(1,modes[1]):get_row_index(N_z,modes[1])
    range_β = get_row_index(1,modes[2]):get_row_index(N_z,modes[2])
    α = state.mean_fields[range_α];
    β = state.mean_fields[range_β];
    α_prime = -r*α + 1im*t*β;
    β_prime = 1im*t*α - r*β;
    state.mean_fields[range_α] .= α_prime;
    state.mean_fields[range_β] .= β_prime;
end

function split_step_prop_singlestep(α,sim,fiber,dt,method)
    # note that k_grid was already centered with fftshift
    
    z_grid = sim.z_grid;
    Δz = z_grid[2] - z_grid[1];
    K_fiber = fiber.fiber_nonlinearity / Δz;
    fiber_dispersion = fiber.fiber_dispersion;
    k_grid = fftfreq(sim.N_z,1/Δz)
    
    if method == "nd"
        α += 1im*K_fiber*(abs2.(α)).*α*dt;
        α = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt)).*fft(α));
    elseif method == "dn"
        α = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt)).*fft(α));
        α += 1im*K_fiber*(abs2.(α)).*α*dt;
    elseif method == "dnd"
        α = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt/2)).*fft(α));
        α = exp.(1im*K_fiber*(abs2.(α))*dt).*α;
        α = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt/2)).*fft(α));
    elseif method == "ndn"
        α = exp.(1im*K_fiber*(abs2.(α))*dt/2).*α;
        α = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt)).*fft(α));
        α = exp.(1im*K_fiber*(abs2.(α))*dt/2).*α;
    end

end

function split_step_prop(α_init,sim,fiber,t_grid,method)
    
    z_grid = sim.z_grid;
    Δz = z_grid[2] - z_grid[1];
    K_fiber = fiber.fiber_nonlinearity / Δz;
    fiber_dispersion = fiber.fiber_dispersion;
    k_grid = fftfreq(sim.N_z,1/Δz)
    N_steps = length(t_grid);
    dt = t_grid[2] - t_grid[1];
    
    α_zt = 1.0im*zeros(N_steps,N_z);
    α = copy(α_init);
    for ii=1:N_steps
        α = split_step_prop_singlestep(α,sim,fiber,dt,method)
        α_zt[ii,:] = α;
    end
    return α_zt;
end 

function fiber_meanfield(state,sim,fiber,t_grid)
    mode = fiber.fiber_mode;
    range_α = get_row_index(1,mode):get_row_index(N_z,mode)
    α_init = state.mean_fields[range_α];
    α_prime = split_step_prop(α_init,sim,fiber,t_grid,"ndn")
    state.mean_fields[range_α] .= α_prime[end,:];
    return α_prime;
end

function BS_transform_broadband(V,t,modes,N_z,N_σ)
    V_prime = copy(V);
    α = modes[1];
    β = modes[2];
    V_cp = copy(V);
    r = sqrt(1-t^2);
    for ii=1:N_z
        for jj=1:N_z
            for ss=1:N_σ
                μ_iαjσ = V_cp[N_z*(α-1)+ii,2*N_z*(ss-1)+jj];
                μ_iβjσ = V_cp[N_z*(β-1)+ii,2*N_z*(ss-1)+jj];
                ν_iαjσ = V_cp[N_z*(α-1)+ii,2*N_z*(ss-1)+jj+N_z];
                ν_iβjσ = V_cp[N_z*(β-1)+ii,2*N_z*(ss-1)+jj+N_z];             
                V_prime[N_z*(α-1)+ii,2*N_z*(ss-1)+jj] = -r*μ_iαjσ + 1im*t*μ_iβjσ;
                V_prime[N_z*(β-1)+ii,2*N_z*(ss-1)+jj] = 1im*t*μ_iαjσ - r*μ_iβjσ;
                V_prime[N_z*(α-1)+ii,2*N_z*(ss-1)+jj+N_z] = -r*ν_iαjσ + 1im*t*ν_iβjσ;
                V_prime[N_z*(β-1)+ii,2*N_z*(ss-1)+jj+N_z] = 1im*t*ν_iαjσ - r*ν_iβjσ;
            end
        end
    end
    return V_prime
end

function BS_fluctuations(state,sim,BS)
    t = BS.t;
    r = sqrt(1-t^2);
    N_z = sim.N_z;
    BS_modes = BS.modes;
    N_σ = sim.num_modes;
    V = copy(state.V_matrix);
    V = BS_transform_broadband(V,t,BS_modes,N_z,N_σ)
    state.V_matrix = V;
end


function propagate_fluctuations_single_step(V_init,α_zt_fn,sim,fiber,t,dt)
    # V_init is a 2N-dimensional vector and needs to be specified for each j,σ,σ'. For initial coherent states
    # μ,ν are basis vectors (single 1).
    
    z_grid = sim.z_grid;
    Δz = z_grid[2] - z_grid[1];
    K_fiber = fiber.fiber_nonlinearity / Δz;
    fiber_dispersion = fiber.fiber_dispersion;
    k_grid = fftfreq(sim.N_z,1/Δz)
    
    # upgraded to second-order midpoint method;
    μ_tmp = V_init[1:N_z];
    ν_tmp = V_init[N_z+1:2*N_z];
    
    μ_tmp = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt)).*fft(μ_tmp));
    ν_tmp = ifft(exp.(-1im/2*fiber_dispersion*(k_grid).^2*(dt)).*fft(ν_tmp));
    
    # half-step advance for first-part of midpoint method
    α_zt = α_zt_fn.(t,z_grid);
    μ_tmp_half = μ_tmp + (2im*K_fiber*abs2.(α_zt).*μ_tmp + 1im*K_fiber*((α_zt).^2).*(conj.(ν_tmp)))*(dt/2);
    ν_tmp_half = ν_tmp + (2im*K_fiber*abs2.(α_zt).*ν_tmp + 1im*K_fiber*((α_zt).^2).*(conj.(μ_tmp)))*(dt/2);
    
    # second part of midpoint
    α_zt = α_zt_fn.(t+dt/2,z_grid);
    μ_tmp = μ_tmp + (2im*K_fiber*abs2.(α_zt).*μ_tmp_half + 1im*K_fiber*((α_zt).^2).*(conj.(ν_tmp_half)))*(dt);
    ν_tmp = ν_tmp + (2im*K_fiber*abs2.(α_zt).*ν_tmp_half + 1im*K_fiber*((α_zt).^2).*(conj.(μ_tmp_half)))*(dt);

    # symmetrizing FT-SQZ-FT didn't appear to be too important here
    μ = μ_tmp;
    ν = ν_tmp;
    V = [μ; ν]
    return V
end

function fluctuation_prop(V_init,α_zt_fn,sim,fiber,t_grid)
    dt = t_grid[2] - t_grid[1]; 
    V = V_init;
    for tt in t_grid
        V = propagate_fluctuations_single_step(V,α_zt_fn,sim,fiber,tt,dt)
    end
    return V
end

function fiber_fluctuations_step(V_sub_tmp, N_z, sim, fiber, t_grid, α_fn)
    U_init = [V_sub_tmp[:,ii]; V_sub_tmp[:,ii+N_z]];
    U_tmp = fluctuation_prop(U_init,α_fn,sim,fiber,t_grid[1:end-1]);
    V_sub_tmp[:,ii] = U_tmp[1:N_z];
    V_sub_tmp[:,ii+N_z] = U_tmp[N_z+1:2*N_z];
end
    
function fiber_fluctuations(state,sim,fiber,t_grid,α_fn)
    
    mode = fiber.fiber_mode;
    V_initial = copy(state.V_matrix);
    N_z = sim.N_z;
    N_σ = sim.num_modes;
    
    for ss=1:N_σ
        V_sub = get_submatrix([mode,ss],V_initial)
        V_sub_tmp = copy(V_sub);
        for ii=1:N_z
            U_init = [V_sub_tmp[:,ii]; V_sub_tmp[:,ii+N_z]];
            U_tmp = fluctuation_prop(U_init,α_fn,sim,fiber,t_grid[1:end-1]);
            V_sub_tmp[:,ii] = U_tmp[1:N_z];
            V_sub_tmp[:,ii+N_z] = U_tmp[N_z+1:2*N_z];
        end
        state.V_matrix[get_ranges([mode,ss])[1],get_ranges([mode,ss])[2]] .= V_sub_tmp;
    end 
end

function get_meanfield_i(state,mode)
    range_mode = get_row_index(1,mode):get_row_index(N_z,mode);
    field = state.mean_fields[range_mode];
    return field
end


function BS_transform_spectral(V,S,modes)
    
    V_prime = copy(V);
    α = modes[1];
    β = modes[2];
    V_cp = copy(V);
    
    for ii=1:N_z
        for jj=1:N_z
            for ss=1:N_σ
                for kk=1:N_z
                    
                    S11 = S(ii,kk)[1,1];
                    S12 = S(ii,kk)[1,2];
                    S21 = S(ii,kk)[2,1];
                    S22 = S(ii,kk)[2,2];
                    
                    μ_kαjσ = V_cp[N_z*(α-1)+kk,2*N_z*(ss-1)+jj];
                    μ_kβjσ = V_cp[N_z*(β-1)+kk,2*N_z*(ss-1)+jj];
                    ν_kαjσ = V_cp[N_z*(α-1)+kk,2*N_z*(ss-1)+jj+N_z];
                    ν_kβjσ = V_cp[N_z*(β-1)+kk,2*N_z*(ss-1)+jj+N_z];
                
                    V_prime[N_z*(α-1)+ii,2*N_z*(ss-1)+jj] += S11*μ_kαjσ + S12*μ_kβjσ;
                    V_prime[N_z*(β-1)+ii,2*N_z*(ss-1)+jj] += S21*μ_kαjσ + S22*μ_kβjσ;
                    V_prime[N_z*(α-1)+ii,2*N_z*(ss-1)+jj+N_z] += S11*ν_kαjσ + S12*ν_kβjσ;
                    V_prime[N_z*(β-1)+ii,2*N_z*(ss-1)+jj+N_z] += S21*ν_kαjσ + S22*ν_kβjσ;
                end
            end
        end
    end
    
    return V_prime
end

function prop_system_mean(components,state,sim,t_grid)
    
    z_grid = sim.z_grid
    
    for ii=1:length(components)
        component = components[ii];
        if typeof(component) == beamsplitter
            BS_meanfield(state,sim,component);
        elseif typeof(component) == fiber
            α_zt = fiber_meanfield(state,sim,component,t_grid);
        end
    end

end

function prop_system(components,state,sim,t_grid)
    
    z_grid = sim.z_grid
    
    for ii=1:length(components)
        component = components[ii];
        if typeof(component) == beamsplitter
            BS_meanfield(state,sim,component);
            BS_fluctuations(state,sim,component);
        elseif typeof(component) == fiber
            α_zt = fiber_meanfield(state,sim,component,t_grid);
            α_zt_fn = LinearInterpolation((t_grid,z_grid),α_zt);
            fiber_fluctuations(state,sim,component,t_grid,α_zt_fn);
        end
    end

end


