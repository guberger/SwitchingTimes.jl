using LinearAlgebra
using Printf
using JuMP
using MosekTools
using PyPlot
using Roots

include(joinpath(@__DIR__, "../../src/SwitchingTimes.jl"))
SwT = SwitchingTimes

matplotlib.rc("font", size = 25)
matplotlib.rc("text", usetex = true)
matplotlib.rc("text.latex", preamble = "\\usepackage{amsmath,amssymb}")
_FC_(c, a) =  matplotlib.colors.colorConverter.to_rgba(c, alpha = a)

function PlotCrossingTime(problem, method, origin, x0, Tmin, Tmax, np)
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
    TimeMax = min(max(τ, Tmin), Tmax)
    @printf("TimeMax = %f\n", TimeMax)
    Dist_list = [SwT.EvalDistQuad(problem.A, x0, E) for E in problem.R1.Es]
    tplot = range(0.0, TimeMax, length = np)
    Dplot = [Dist.(t) for t in tplot, Dist in Dist_list]
    fig = figure(figsize = (18.0, 8.5))
    ax = fig.add_subplot()
    ax.plot(tplot, Dplot)
    for Dist in Dist_list
        roots = find_zeros(Dist, 0.0, TimeMax)
        println(Tuple(roots))
        ax.plot(roots, Dist.(roots), marker = ".", ms = 15, ls = "none")
    end
    DplotMax = maximum(Dplot, dims = 2)
    ax.plot(tplot, DplotMax, ls = "--", lw = 2.0)
    TCross = SwT.FindCrossingTime(problem, x0, (0.0, TimeMax), 1e-5)
    @printf("TCross = %f\n", TCross)
    if TCross < Inf
        ax.plot(TCross, 0.0, marker = "+", ms = 15, ls = "none", mew = 3.0, c = "k")
    end
    return fig
end
