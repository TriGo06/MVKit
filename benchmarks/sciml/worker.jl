# Native SciML worker; binary files use little-endian Float64 in C order.
using StochasticDiffEqLowOrder, SciMLBase, DiffEqNoiseProcess, JSON3, Statistics, LinearAlgebra

function main(directory, repeats)
    c = JSON3.read(read(joinpath(directory, "case.json"), String))
    n, d, steps = Int(c.n), Int(c.dim), Int(c.steps)
    width = c.name == "pairwise" ? 2d : d
    a, b, sigma, beta = Float64(c.a), Float64(c.b), Float64(c.sigma), Float64(c.beta)
    pairwise, multiplicative = c.name == "pairwise", c.name == "multiplicative"
    m = zeros(width)
    x0 = permutedims(reshape(reinterpret(Float64, read(joinpath(directory, "x0.bin"))), width, n))
    z = permutedims(reshape(reinterpret(Float64, read(joinpath(directory, "z.bin"))), width, n, steps), (2, 1, 3))
    dt = c.t_final / steps
    times = collect(range(0.0, c.t_final; length=steps + 1))
    # NoiseGrid receives cumulative Brownian values at exactly the solver grid.
    values = [zeros(n, width)]
    for k in 1:steps
        push!(values, values[end] + sqrt(dt) * z[:, :, k])
    end
    function drift!(du, u, p, t)
        if pairwise
            Threads.@threads for i in 1:n
                for k in 1:d
                    du[i, k] = u[i, d + k]
                    du[i, d + k] = 0.0
                end
                for j in 1:n
                    r2 = 0.0
                    for k in 1:d
                        r2 += (u[j, k] - u[i, k])^2
                    end
                    weight = (1 + r2)^(-beta)
                    for k in 1:d
                        du[i, d + k] += weight * (u[j, d + k] - u[i, d + k])
                    end
                end
                for k in 1:d
                    du[i, d + k] /= n
                end
            end
        else
            for k in 1:width
                total = 0.0
                for i in 1:n
                    total += u[i, k]
                end
                m[k] = total / n
            end
            Threads.@threads for i in 1:n
                for k in 1:width
                    du[i, k] = a * u[i, k] + b * m[k]
                end
            end
        end
        nothing
    end
    function diffusion!(du, u, p, t)
        if multiplicative
            @. du = sigma * u
        elseif pairwise
            fill!(du, 0.0)
            du[:, d+1:end] .= sigma
        else
            fill!(du, sigma)
        end
        nothing
    end
    function simulate()
        # New mutable noise object per solve, without copying precomputed values.
        noise = NoiseGrid(times, values)
        problem = SDEProblem(drift!, diffusion!, x0, (0.0, Float64(c.t_final)); noise)
        # EM defaults to split=true, which evaluates diffusion after the drift.
        # Explicit false matches the other workers' simultaneous Euler update.
        solution = solve(problem, EM(false); dt, adaptive=false, tstops=times[2:end],
            save_everystep=false, save_start=true, save_end=true, save_noise=false,
            dense=false, maxiters=steps + 1)
        string(solution.retcode) == "Success" || error("Solver failed: $(solution.retcode)")
        return solution
    end
    start = time_ns()
    result = simulate()
    first = (time_ns() - start) / 1e9
    elapsed = Float64[]
    for _ in 1:repeats
        result = nothing
        start = time_ns()
        result = simulate()
        push!(elapsed, (time_ns() - start) / 1e9)
    end
    open(joinpath(directory, "result.bin"), "w") do io
        write(io, permutedims(result.u[1]))
        write(io, permutedims(result.u[end]))
    end
    metadata = Dict("first_call_s" => first, "warm_s" => elapsed,
        "backend_version" => string(pkgversion(StochasticDiffEqLowOrder)),
        "julia_version" => string(VERSION), "threads" => Threads.nthreads(),
        "self_peak_rss_bytes" => Sys.maxrss(),
        "input_transfer_s" => 0.0, "device" => "cpu", "interface" => "Julia")
    write(joinpath(directory, "timing.json"), JSON3.write(metadata) * "\n")
end

length(ARGS) == 2 || error("usage: worker.jl INPUT_DIRECTORY REPEATS")
main(ARGS[1], parse(Int, ARGS[2]))
