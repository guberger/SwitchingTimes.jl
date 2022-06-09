using LinearAlgebra
using Printf
using JuMP
using MosekTools
using PyPlot

include(joinpath(@__DIR__, "../../src/SwitchingTimes.jl"))
SwT = SwitchingTimes

matplotlib.rc("font", size = 25)
matplotlib.rc("text", usetex = true)
matplotlib.rc("text.latex", preamble = "\\usepackage{amsmath,amssymb}")
_FC_(c, a) =  matplotlib.colors.colorConverter.to_rgba(c, alpha = a)

function PlotRegions2D(problem, method, origin, x0, xlims, ylims, np)
    γ = SwT.StabilityMargin(problem.A)/2
    optim_solver = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)
    sol = SwT.OptimUpperBoundCrossingTime(
        problem, method, origin, γ, optim_solver)[1]
    V0 = SwT.EvalQuad(sol.EP)(x0)
    if !isnothing(sol.rmax)
        τ_max = SwT.UpperBoundCrossingTime(method, sol.rmax, sol.rmin, sol.EP, γ)
        @printf("Worst-case cross time = %f\n", τ_max)
    else
        @printf("Worst-case cross time = Nothing\n")
    end
    τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
    @printf("τ = %f\n", τ)
    if Int(problem.obj) == 1
        error("Not implemented yet.")
    end
    if Int(problem.obj) == 2
        if Int(origin) == 1
            fig = PLOT_escape_inside(
                problem, method, sol, x0, γ, xlims, ylims, np, τ, true)
        elseif Int(origin) == 2
            fig = PLOT_escape_outside(
                problem, method, sol, x0, γ, xlims, ylims, np, τ, true)
        end
    end
    return fig
end

sum_func(F_list, λ_list) = let F_list = F_list, λ_list = λ_list
    x -> sum(y -> y[2]*y[1](x), zip(F_list, λ_list))
end
max_func(F_list) = let F_list = F_list
    x -> maximum(y -> y(x), F_list)
end

function plot_negativeset!(ax, X1, X2, Y, c, a; H = "none")
    ymin = minimum(Y)
    ymax = maximum(Y)
    delta = ymax - ymin
    if ymin >= -1e-7*delta
        @warn(string("Negative level set close to singular. Ymin = ",
            ymin/delta, "*ΔY",))
        return
    end
    ax.contour(X1, X2, Y, levels = (ymin, 0.0), colors = c)
    # https://github.com/matplotlib/matplotlib/issues/2789/
    h = ax.contourf(X1, X2, Y, levels = (ymin, 0.0),
        colors = "none", hatches = H)
    for coll in h.collections
        coll.set_facecolor(_FC_(c, a))
        coll.set_edgecolor(_FC_(c, 1.0))
        coll.set_linewidth(0.)
    end
end

function plot_negativeset_list!(ax, X1, X2, Y_list, c, a; H = "none")
    for Y in Y_list
        plot_negativeset!(ax, X1, X2, Y, c, a, H = H)
    end
end

function compute_trajectory(A, x0, Tmax, np)
    ttraj = range(0.0, stop = Tmax, length = np)
    X = map(t -> exp(A*t)*x0, ttraj)
    return ntuple(i -> map(x -> x[i], X), 2)
end

function build_grid(xlims, ylims, np)
    c1 = range(xlims[1], stop = xlims[2], length = np)
    c2 = range(ylims[1], stop = ylims[2], length = np)
    Xftemp = collect(Iterators.product(c1, c2))
    Xf = map(x -> [x...], Xftemp)
    X1 = map(x -> x[1], Xf)
    X2 = map(x -> x[2], Xf)
    return (Xf, X1, X2)
end

function compute_deriv_RHS(method, YP, sol, γ)
    if Int(method) in (1, 2)
        return sol.rmin*SwT.InnerRadius(sol.EP)
    elseif Int(method) == 3
        return 1.0
    elseif Int(method) == 4
        return 2*γ*YP
    end
end

function deriv_leg_label(method)
    if Int(method) in (1, 2)
        return L"$\mathcal{L}_AV(x)\leq-r/\lVert P\rVert$"
    elseif Int(method) == 3
        return L"$\mathcal{L}_AV(x)\leq-1$"
    elseif Int(method) == 4
        return L"$\mathcal{L}_AV(x)\leq-2\gamma V(x)$"
    end
end

function PLOT_escape_inside(problem, method, sol, x0, γ,
        xlims, ylims, np, Tmax, leg)
    #---------------------------------------------------------------------------
    fig = figure(figsize = (18.0, 8.5))
    ax1, ax2 = fig.subplots(1, 2,
        gridspec_kw = Dict("wspace" => 0.2),
        subplot_kw = Dict("aspect" => "equal"))
    xtraj1, xtraj2 = compute_trajectory(problem.A, x0, Tmax, 500)
    Xf, X1, X2 = build_grid(xlims, ylims, np)
    F_E0 = (SwT.EvalQuad(E) for E in problem.R0.Es)
    F_E1 = (SwT.EvalQuad(E) for E in problem.R1.Es)
    YP = map(SwT.EvalQuad(sol.EP), Xf)
    YdP = map(SwT.EvalDerivQuad(problem.A, sol.EP), Xf)
    if !isnothing(sol.EK)
        YEK = map(SwT.EvalQuad(sol.EK), Xf)
    end
    YR0out = (map(x -> -F(x), Xf) for F in F_E0)
    YR1 = map(max_func(F_E1), Xf)
    YGV = compute_deriv_RHS(method, YP, sol, γ)
    #
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    plot_negativeset_list!(ax1, X1, X2, YR0out, "gray", 0.0)
    plot_negativeset!(ax1, X1, X2, YR1, "red", 0.1)
    ax1.plot(0.0, 0.0, marker = "+", ms = 15, c = "k", mew = 2.3)
    ax1.plot(xtraj1, xtraj2, c = "k", ls = "--")
    ax1.plot(x0..., marker = "*", ms = 15, c = "tab:orange")
    plot_negativeset!(ax1, X1, X2, YP .- sol.rmin, "tab:orange", 0.2)
    if !isnothing(sol.rmax)
        plot_negativeset!(ax1, X1, X2, -YP .+ sol.rmax, "tab:blue", 0.2)
    end
    #
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    plot_negativeset_list!(ax2, X1, X2, YR0out, "gray", 0.0)
    plot_negativeset!(ax2, X1, X2, YR1, "red", 0.1)
    ax2.plot(0.0, 0.0, marker = "+", ms = 15, c = "k", mew = 2.3)
    plot_negativeset!(ax2, X1, X2, YdP .+ YGV, "gold", 0.0, H = ("+",))
    if !isnothing(sol.EK)
        plot_negativeset!(ax2, X1, X2, YEK, "magenta", 0.0, H = ("\\\\",))
    end
    if leg
        LH = (
            matplotlib.patches.Patch(fc = "red", ec = "red", alpha = 0.5,
                label = L"$\mathcal{R}$"),
            matplotlib.lines.Line2D([0], [0], marker = "o", ls = "none",
                mew = 2, mfc = "none", mec = "gray", ms = 20,
                label = L"$\mathcal{F}_1$"),
            matplotlib.lines.Line2D([0], [0], marker = "*", ls = "none",
                c = "tab:orange", ms = 15, label = L"$x_0$"),
            matplotlib.patches.Patch(fc = "tab:orange", alpha = 0.5,
                ec = "tab:orange", label = L"$V(x)\leq r$"),
            matplotlib.patches.Patch(fc = "tab:blue", alpha = 0.5,
                ec = "tab:blue", label = L"$V(x)\geq 1$"),
            matplotlib.lines.Line2D([0], [0], ls = "--", c = "k",
                label = L"trajectory for $t\in[0,\tau]$"))
        ax1.legend(handles = LH, ncol = 2, fontsize = 25,
            loc = "lower center", bbox_to_anchor = (0.5, 1.01))

        LH = (
            matplotlib.patches.Patch(fc = "red", ec = "red", alpha = 0.5,
                label = L"$\mathcal{R}$"),
            matplotlib.lines.Line2D([0], [0], marker = "o", ls = "none",
                mew = 2, mfc = "none", mec = "gray", ms = 20,
                label = L"$\mathcal{F}_1$"),
            matplotlib.patches.Patch(fc = "none", ec = "magenta",
                hatch = "\\\\", label = L"$W(x)\leq0$"),
            matplotlib.patches.Patch(fc = "none", ec = "gold", hatch = "+",
                label = deriv_leg_label(method)))
        ax2.legend(handles = LH, ncol = 2, fontsize = 25,
            loc = "lower center", bbox_to_anchor = (0.5, 1.01))
    end
    return fig
end

function PLOT_escape_outside(problem, method, sol, x0, γ,
        xlims, ylims, np, Tmax, leg)
    #---------------------------------------------------------------------------
    fig = figure(figsize = (18.0, 8.5))
    ax1, ax2 = fig.subplots(1, 2,
        gridspec_kw = Dict("wspace" => 0.2),
        subplot_kw = Dict("aspect" => "equal"))
    xtraj1, xtraj2 = compute_trajectory(problem.A, x0, Tmax, 500)
    Xf, X1, X2 = build_grid(xlims, ylims, np)
    F_E0 = (SwT.EvalQuad(E) for E in problem.R0.Es)
    F_E1 = (SwT.EvalQuad(E) for E in problem.R1.Es)
    YP = map(SwT.EvalQuad(sol.EP), Xf)
    YdP = map(SwT.EvalDerivQuad(problem.A, sol.EP), Xf)
    if !isnothing(sol.λDP_)
        YλDPxR1 = map(sum_func(F_E1, sol.λDP_), Xf)
    end
    YR0out = (map(x -> -F(x), Xf) for F in F_E0)
    YR1 = map(max_func(F_E1), Xf)
    if !isnothing(sol.λP_)
        YλPxR1 = map(sum_func(F_E1, sol.λP_), Xf)
    end
    YGV = compute_deriv_RHS(method, YP, sol, γ)
    #
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    plot_negativeset_list!(ax1, X1, X2, YR0out, "gray", 0.0)
    plot_negativeset!(ax1, X1, X2, YR1, "red", 0.1)
    ax1.plot(0.0, 0.0, marker = "+", ms = 15, c = "k", mew = 2.3)
    ax1.plot(xtraj1, xtraj2, c = "k", ls = "--")
    ax1.plot(x0..., marker = "*", ms = 15, c = "tab:orange")
    plot_negativeset!(ax1, X1, X2, YP .- sol.rmin, "tab:orange", 0.2)
    if !isnothing(sol.rmax)
        plot_negativeset!(ax1, X1, X2, -YP .+ sol.rmax, "tab:blue", 0.2)
    end
    if !isnothing(sol.λP_)
        # plot_negativeset!(ax1, X1, X2, YλPxR1, "cyan", 0.0, H = ("//",))
    end
    #
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    plot_negativeset_list!(ax2, X1, X2, YR0out, "gray", 0.0)
    plot_negativeset!(ax2, X1, X2, YR1, "red", 0.1)
    ax2.plot(0.0, 0.0, marker = "+", ms = 15, c = "k", mew = 2.3)
    plot_negativeset!(ax2, X1, X2, YdP .+ YGV, "gold", 0.0, H = ("+",))
    if !isnothing(sol.λDP_)
        # plot_negativeset!(ax2, X1, X2, YλDPxR1, "cyan", 0.0, H = ("//",))
    end
    if leg
        LH = (
            matplotlib.patches.Patch(fc = "red", ec = "red", alpha = 0.5,
                label = L"$\mathcal{R}$"),
            matplotlib.lines.Line2D([0], [0], marker = "o", ls = "none",
                mew = 2, mfc = "none", mec = "gray", ms = 20,
                label = L"$\mathcal{F}_1$"),
            matplotlib.lines.Line2D([0], [0], marker = "*", ls = "none",
                c = "tab:orange", ms = 15, label = L"$x_0$"),
            matplotlib.patches.Patch(fc = "tab:orange", alpha = 0.5,
                ec = "tab:orange", label = L"$V(x)\leq r$"),
            matplotlib.patches.Patch(fc = "tab:blue", alpha = 0.5,
                ec = "tab:blue", label = L"$V(x)\geq 1$"),
            matplotlib.lines.Line2D([0], [0], ls = "--", c = "k",
                label = L"trajectory for $t\in[0,\tau]$"))
        ax1.legend(handles = LH, ncol = 2, fontsize = 25,
            loc = "lower center", bbox_to_anchor = (0.5, 1.01))

        LH = (
            matplotlib.patches.Patch(fc = "red", ec = "red", alpha = 0.5,
                label = L"$\mathcal{R}$"),
            matplotlib.lines.Line2D([0], [0], marker = "o", ls = "none",
                mew = 2, mfc = "none", mec = "gray", ms = 20,
                label = L"$\mathcal{F}_1$"),
            matplotlib.patches.Patch(fc = "none", ec = "gold", hatch = "+",
                label = deriv_leg_label(method)))
        ax2.legend(handles = LH, ncol = 2, fontsize = 25,
            loc = "lower center", bbox_to_anchor = (0.5, 1.01))
    end
    return fig
end
