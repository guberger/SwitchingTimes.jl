using LinearAlgebra
using Printf
using JuMP
using MosekTools
using PyPlot
using Random

include(joinpath(@__DIR__, "../../src/SwitchingTimes.jl"))
SwT = SwitchingTimes

matplotlib.rc("legend", fontsize = 30)
matplotlib.rc("axes", labelsize = 30)
matplotlib.rc("xtick", labelsize = 20)
matplotlib.rc("ytick", labelsize = 20)
matplotlib.rc("text", usetex = true)
matplotlib.rc("text.latex", preamble = "\\usepackage{amsmath,amssymb}")
_cols = repeat(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"], 10, 1)

function compute_ylims(yvals)
    if !isempty(yvals)
        ymax = maximum(x -> maximum(y -> isinf(y) ? -1.0 : y, x[2]), yvals)
        ymin = minimum(x -> minimum(x[2]), yvals)
    else
        return (1/10.0, 10.0)
    end
    if ymax <= 0
        return (1/10.0, 10.0)
    end
    if ymin ≈ ymax
        return (ymin/10.0, 10.0*ymax)
    end
    return (ymin, ymax)
end

function method_leg_label(method::SwT.OptimMethod)
    if Int(method) == 1
        return "Quad_bound_pre_sep"
    elseif Int(method) == 2
        return "Quad_bound_pre_all"
    elseif Int(method) == 3
        return L"Model with $G(V)=-1$"
    elseif Int(method) == 4
        return L"Model with $G(V)=-2\gamma V$"
    end
end

## Compare

function MakeRandomTestsUpperBoundsCompare(d, R0, R1, obj,
        method_list, origin, maxSamples)
    count = 0
    κ_list = Float64[]
    bounds_list = Vector{Float64}[]
    cputimes_list = [zeros(maxSamples) for i in eachindex(method_list)]

    while count < maxSamples
        println(count + 1)
        A = SwT.GenerateMatrixDiagonal(d)
        κ = cond(A)
        if κ > 50_000
            continue
        end

        problem = SwT.ProblemSpecifications(A, R0, R1, obj)
        bounds = fill(Inf, length(method_list))
        γ = SwT.StabilityMargin(A)/2
        optim_solver = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
        for (i, method) in enumerate(method_list)
            if count == 0
                println("Warm-up...")
                SwT.OptimUpperBoundCrossingTime(problem, method, origin, γ,
                    optim_solver, do_print = false)
            end
            cputimes_list[i][count+1] = @elapsed (sol, TSt, PSt, DSt) =
                SwT.OptimUpperBoundCrossingTime(problem, method, origin, γ,
                optim_solver, do_print = true)
            if !all(Int.(PSt) .==  Int.(DSt) .== 1)
                gmin = SwT.CheckUpperBoundCrossingTime(
                    sol, problem, method, origin, γ)
                if gmin < -1e-8*norm(sol.EP)
                    @warn("Failed")
                    display(gmin)
                    display(norm(sol.EP))
                    continue
                end
            end
            V0 = SwT.EvalQuad(sol.EP)(x0)
            if !isnothing(sol.rmax)
                τ_max = SwT.UpperBoundCrossingTime(
                    method, sol.rmax, sol.rmin, sol.EP, γ)
                @printf("  -> %d) Worst-case cross time = %f\n", i, τ_max)
            else
                @printf("  -> %d) Worst-case cross time = Nothing\n", i)
            end
            τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
            @printf("τ = %f\n", τ)
            bounds[i] = τ
        end

        push!(κ_list, κ)
        push!(bounds_list, bounds)
        count += 1
    end

    println(cputimes_list)
    display(sum.(cputimes_list)./maxSamples)
    return κ_list, bounds_list, cputimes_list
end

function PlotUpperBoundsCompare(κ_list, bounds_list, obj, method_list,
        ref_index, others_index; location = nothing)
    #---------------------------------------------------------------------------
    norm_bounds_list = Tuple{Float64, Vector{Float64}}[]
    fail_list = Tuple{Float64, Int}[]
    zero_list = Tuple{Float64, Int}[]
    nothers = length(others_index)
    count = 1
    for (κ, bounds) in zip(κ_list, bounds_list)
        bounds_norm = fill(Inf, nothers)
        bound_ref = bounds[ref_index]
        if isinf(bound_ref)
            @printf("WARNING -> fail 1: i = %d, κ = %f\n", count, κ)
            for (k, i)  in enumerate(others_index)
                if isinf(bounds[i])
                    bounds_norm[k] = 1.0
                else
                    push!(zero_list, (κ, k))
                end
            end
        elseif bound_ref < 1e-5
            for (k, i)  in enumerate(others_index)
                if bounds[i] < 1e-5
                    bounds_norm[k] = 1.0
                else
                    push!(fail_list, (κ, k))
                end
            end
        else
            for (k, i)  in enumerate(others_index)
                if isinf(bounds[i])
                    push!(fail_list, (κ, k))
                elseif bounds[i] < 1e-5
                    push!(zero_list, (κ, k))
                else
                    bounds_norm[k] = bounds[i]/bound_ref
                end
            end
        end
        push!(norm_bounds_list, (κ, bounds_norm))
        count += 1
    end
    println()
    println(fail_list)
    println(zero_list)

    fig = figure(figsize = (18.0, 8.5))
    ax = fig.add_subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")

    if isempty(κ_list)
        return fig
    end

    α = 1.2
    β = 0.03

    ylims = compute_ylims(norm_bounds_list)
    ylog = log.(ylims)
    ylog_c = sum(ylog)/2
    ylog_d = ylog .- ylog_c
    nbottom = isempty(zero_list) ? 0 : nothers + 1
    nup = isempty(fail_list) ? 0 : nothers + 1
    ylims_in = exp.(α.*ylog_d .+ ylog_c)
    ylims_out = exp.((α .+ β.*(nbottom, nup)).*ylog_d .+ ylog_c)

    for x in norm_bounds_list
        κ = x[1]
        ax.plot((κ, κ), ylims_out, lw = 1, c = "gray")
        for (i, bound) in enumerate(x[2])
            if !isinf(bound)
                ax.plot(κ, bound, marker = "o", ms = 7, c = _cols[i])
            end
        end
    end
    for x in fail_list
        yval = exp.((α + β*x[2]).*ylog_d .+ ylog_c)
        ax.plot(x[1], yval[2], marker = "x", ms = 7, mew = 4, c = _cols[x[2]])
    end
    for x in zero_list
        yval = exp.((α + β*x[2]).*ylog_d .+ ylog_c)
        ax.plot(x[1], yval[1], marker = "x", ms = 7, mew = 4, c = _cols[x[2]])
    end

    ax.set_ylim(ylims_out)
    xlims = ax.get_xlim()
    if !isempty(zero_list)
        ax.plot(xlims, (ylims_in[1], ylims_in[1]),
            ls = "--", lw = 1, c = "gray")
    end
    if !isempty(fail_list)
        ax.plot(xlims, (ylims_in[2], ylims_in[2]),
            ls = "--", lw = 1, c = "gray")
    end
    ax.set_xlim(xlims)
    if location == "top"
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xticks([], minor = true)
    else
        ax.set_xlabel(L"Condition number of $A$")
    end
    ax.set_ylabel(L"Our bounds $\big/$ bound of Rabi (2020)")

    if location != "bottom"
        LH = [matplotlib.patches.Patch(
            fc = _cols[k], label = method_leg_label(method_list[i]))
            for (k, i) in enumerate(others_index)]
        ax.legend(handles = LH, ncol = nothers,
            loc = "lower center", bbox_to_anchor = (0.5, 1.0))
    end

    return fig
end

function PlotComputationTimes(cputimes_list_inside, cputimes_list_outside,
        labels)
    #---------------------------------------------------------------------------
    fig = figure(figsize = (18.0, 8.5))
    axIn, axOut = fig.subplots(1, 2, gridspec_kw = Dict("wspace" => 0.3))
    axIn.boxplot(cputimes_list_inside, showfliers = false)
    axIn.set_title("Case I (origin inside)", fontsize = 25, pad = 15)
    axOut.boxplot(cputimes_list_outside, showfliers = false)
    axOut.set_title("Case II (origin outside)", fontsize = 25, pad = 15)
    for ax in (axIn, axOut)
        ax.set_xticklabels(labels, fontsize = 25)
        ax.tick_params(axis = "x", pad = 10, length = 0)
        ax.tick_params(axis = "y", labelsize = 25)
        ax.set_ylabel("Computation time (sec.)", labelpad = 10)
    end
    return fig
end

## Accuracy

function MakeRandomTestsUpperboundsAccuracy(d, R0, R1, obj,
        method_list, origin, Tmin, Tmax, maxSamples)
    #---------------------------------------------------------------------------
    count = 0
    κ_list = Float64[]
    bounds_list = Vector{Float64}[]
    crosstime_list = Float64[]

    while count < maxSamples
        println(count + 1)
        A = SwT.GenerateMatrixDiagonal(d)
        κ = cond(A)
        if κ > 50_000
            continue
        end

        problem = SwT.ProblemSpecifications(A, R0, R1, obj)
        bounds = fill(Inf, length(method_list))
        γ = SwT.StabilityMargin(A)/2
        optim_solver = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
        for (i, method) in enumerate(method_list)
            (sol, TSt, PSt, DSt) = SwT.OptimUpperBoundCrossingTime(problem,
                method, origin, γ, optim_solver, do_print = true)
            if !all(Int.(PSt) .==  Int.(DSt) .== 1)
                gmin = SwT.CheckUpperBoundCrossingTime(
                    sol, problem, method, origin, γ)
                if gmin < -1e-8*norm(sol.EP)
                    @warn("Failed")
                    display(gmin)
                    display(norm(sol.EP))
                    continue
                end
            end
            V0 = SwT.EvalQuad(sol.EP)(x0)
            if !isnothing(sol.rmax)
                τ_max = SwT.UpperBoundCrossingTime(
                    method, sol.rmax, sol.rmin, sol.EP, γ)
                @printf("  -> %d) Worst-case cross time = %f\n", i, τ_max)
            else
                @printf("  -> %d) Worst-case cross time = Nothing\n", i)
            end
            τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
            @printf("τ = %f\n", τ)
            bounds[i] = τ
        end

        τ_min = minimum(bounds)
        Time = min(max(τ_min, Tmin), Tmax)
        TCross = SwT.FindCrossingTime(problem, x0, (0.0, Time), 1e-5)
        @printf("TCross = %f\n", TCross)
        if (Int(obj) == 1 && Int(origin) == 1) ||
                (Int(obj) == 2 && Int(origin) == 2) && TCross == Inf
            @warn("TCross == Inf")
        end

        push!(κ_list, κ)
        push!(bounds_list, bounds)
        push!(crosstime_list, TCross)
        count += 1
    end

    return κ_list, bounds_list, crosstime_list
end

function PlotUpperBoundsAccuracy(κ_list, bounds_list, crosstime_list, obj,
        method_list, ratios_index; location = nothing)
    #---------------------------------------------------------------------------
    norm_bounds_list = Tuple{Float64, Vector{Float64}}[]
    abs_bounds_list = Tuple{Float64, Vector{Float64}}[]
    fail_list = Tuple{Float64, Int}[]
    zero_list = Tuple{Float64, Int}[]
    nratios = length(method_list)
    count = 1
    for (κ, bounds, TCross) in zip(κ_list, bounds_list, crosstime_list)
        bounds_norm = fill(Inf, nratios)
        bounds_abs = fill(Inf, nratios)
        if TCross < 1e-5 || isinf(TCross)
            for (k, i)  in enumerate(ratios_index)
                if isinf(bounds[i])
                    push!(fail_list, (κ, k))
                elseif bounds[i] < 1e-5
                    push!(zero_list, (κ, k))
                else
                    bounds_abs[k] = bounds[i]
                end
            end
        else
            for (k, i)  in enumerate(ratios_index)
                if isinf(bounds[i])
                    push!(fail_list, (κ, k))
                else
                    bounds_norm[k] = bounds[i]/TCross
                end
            end
        end
        push!(norm_bounds_list, (κ, bounds_norm))
        push!(abs_bounds_list, (κ, bounds_abs))
        count += 1
    end
    println()
    println(fail_list)
    println(zero_list)

    fig = figure(figsize = (18.0, 8.5))
    ax = fig.add_subplot()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax3 = ax.twinx()

    if isempty(κ_list)
        return fig
    end

    α = 1.2
    β = 0.03

    ylims = compute_ylims(norm_bounds_list)
    ylog = log.(ylims)
    ylog_c = sum(ylog)/2
    ylog_d = ylog .- ylog_c
    nbottom = isempty(zero_list) ? 0 : nratios + 1
    nup = isempty(fail_list) ? 0 : nratios + 1
    ylims_in = exp.(α.*ylog_d .+ ylog_c)
    ylims_out = exp.((α .+ β.*(nbottom, nup)).*ylog_d .+ ylog_c)

    for x in norm_bounds_list
        κ = x[1]
        ax.plot((κ, κ), ylims_out, lw = 1, c = "gray")
        for (i, bound) in enumerate(x[2])
            if !isinf(bound)
                ax.plot(κ, bound, marker = "o", ms = 7, c = _cols[i])
            end
        end
    end
    for x in abs_bounds_list
        κ = x[1]
        for (i, bound) in enumerate(x[2])
            if !isinf(bound)
                ax2.plot(κ, bound, marker = "d", ms = 7, c = _cols[i])
            end
        end
    end
    for x in fail_list
        yval = exp.((α + β*x[2]).*ylog_d .+ ylog_c)
        ax.plot(x[1], yval[2], marker = "x", ms = 7, mew = 4, c = _cols[x[2]])
    end
    for x in zero_list
        yval = exp.((α + β*x[2]).*ylog_d .+ ylog_c)
        ax.plot(x[1], yval[1], marker = "x", ms = 7, mew = 4, c = _cols[x[2]])
    end

    ax.set_ylim(ylims_out)
    xlims = ax.get_xlim()
    if !isempty(zero_list)
        ax.plot(xlims, (ylims_in[1], ylims_in[1]),
            ls = "--", lw = 1, c = "gray")
    end
    if !isempty(fail_list)
        ax.plot(xlims, (ylims_in[2], ylims_in[2]),
            ls = "--", lw = 1, c = "gray")
    end
    ylims2 = compute_ylims(abs_bounds_list)
    ylog2 = log.(ylims2)
    ylog2_c = sum(ylog2)/2
    ylog2_d = ylog2 .- ylog2_c
    ylims2_in = exp.(α.*ylog2_d .+ ylog2_c)
    ylims2_out = exp.((α .+ β.*(nbottom, nup)).*ylog2_d .+ ylog2_c)
    ax2.set_ylim(ylims2_out)

    ax.set_xlim(xlims)
    if location == "top"
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xticks([], minor = true)
    else
        ax.set_xlabel(L"Condition number of $A$")
    end
    ax.set_ylabel(L"Bounds $\big/$ actual escaping time")
    ax3.set_ylabel("Bounds [sec.]", labelpad=36)
    ax3.set_yticklabels([])
    ax3.set_yticks([], minor = true)
    ax3.set_yticks([])

    if location != "bottom"
        LH = [matplotlib.patches.Patch(
            fc = _cols[k], label = method_leg_label(method_list[i]))
            for (k, i) in enumerate(ratios_index)]
        ax.legend(handles = LH, ncol = nratios,
            loc = "lower center", bbox_to_anchor = (0.5, 1.0))
    end

    return fig
end
