module TestMain

using Test
using LinearAlgebra
using JuMP
using SDPA
@static if isdefined(Main, :TestLocal)
    include("../src/SwitchingTimes.jl")
else
    using SwitchingTimes
end
SwT = SwitchingTimes

sleep(0.1) # used for good printing
println("Started test")

@testset "2D models" begin
optim_solver = optimizer_with_attributes(SDPA.Optimizer)
Tmax = 1.0e2
Tmin = 1.0

## Inside
origin = SwT.origin_inside
b1 = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
c1p = [1.0, 1.0, 1.0, 1.0]
Q0_list = [0.5*[1.0 0.0; 0.0 1.0]]
xc0 = [-0.5, 0.5]
c1 = c1p + b1*xc0
E1s = [SwT.QuadMatrixAffine(b1[i, :]/2.0, -c1[i]) for i = 1:length(c1)]
E0s = [SwT.QuadMatrix(Q0, -Q0*xc0, -1.0 + xc0'*Q0*xc0) for Q0 in Q0_list]
R0 = SwT.ConvexSetHull(E0s)
R1 = SwT.ConvexSetCap(E1s)
x0 = xc0 + [1.0, 1.0]
## Jordan matrix
A = [-1.0 3.0; 0.0 -1.0]
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)
γ = SwT.StabilityMargin(A)/2

method = SwT.quad_bound_pre_sep

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 7.4375*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 296.79260459143285*1.0001

method = SwT.quad_bound_pre_all

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 7.4375*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 296.79260459143285*1.0001

method = SwT.quad_bound_optim

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= -9.178684780868824*0.9999
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 31.069422069715227*1.0001
gmin = SwT.CheckUpperBoundCrossingTime(sol, problem, method, origin, γ)
@test gmin >= -1e-5*norm(sol.EP)

TimeMax = min(max(τ, Tmin), Tmax)
TCross = SwT.FindCrossingTime(problem, x0, (0.0, TimeMax), 1e-5)
@test TCross <= eps(TCross)

## Rotation matrix
A = [-0.1 1.0; -1.0 -0.1]
problem.A = A
γ = SwT.StabilityMargin(A)/2

method = SwT.quad_bound_pre_sep

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 12.499999999999996*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 45.00002884249298*1.0001

method = SwT.quad_bound_pre_all

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 12.499999999999996*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 45.00002884249298*1.0001

method = SwT.quad_bound_optim

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= -38.98846114603356*0.9999
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 45.01109696593912*1.0001
gmin = SwT.CheckUpperBoundCrossingTime(sol, problem, method, origin, γ)
@test gmin >= -1e-5*norm(sol.EP)

TimeMax = min(max(τ, Tmin), Tmax)
TCross = SwT.FindCrossingTime(problem, x0, (0.0, TimeMax), 1e-5)
@test TCross <= eps(TCross)


## Outside
origin = SwT.origin_outside
b1 = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
c1p = [1.0, 1.0, 1.0, 1.0]
Q0_list = [0.5*[1.0 0.0; 0.0 1.0]]
xc0 = [-2.0, 2.0]
c1 = c1p + b1*xc0
E1s = [SwT.QuadMatrixAffine(b1[i, :]/2.0, -c1[i]) for i = 1:length(c1)]
E0s = [SwT.QuadMatrix(Q0, -Q0*xc0, -1.0 + xc0'*Q0*xc0) for Q0 in Q0_list]
R0 = SwT.ConvexSetHull(E0s)
R1 = SwT.ConvexSetCap(E1s)
x0 = xc0

## Jordan matrix
A = [-1.0 3.0; 0.0 -1.0]
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)
γ = SwT.StabilityMargin(A)/2

method = SwT.quad_bound_pre_all

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 7.0*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 9.847272565523655*1.0001

method = SwT.quad_bound_optim

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 0.6464465141426992*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 0.250000809390162*1.0001

method = SwT.log_bound_optim

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
# @test Int.(TSt) == [1]
# @test Int.(PSt) == [1]
# @test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 0.7795501176769903*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 0.2279656196822932*1.0001
gmin = SwT.CheckUpperBoundCrossingTime(sol, problem, method, origin, γ)
@test gmin >= -1e-2*norm(sol.EP)

TimeMax = min(max(τ, Tmin), Tmax)
TCross = SwT.FindCrossingTime(problem, x0, (0.0, TimeMax), 1e-5)
@test TCross <= 0.14136010689847434*1.0001

## Rotation matrix
A = [-0.1 1.0; -1.0 -0.1]
problem.A = A
γ = SwT.StabilityMargin(A)/2

method = SwT.quad_bound_pre_all

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 39.99999999999999*1.0001
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 15.000001148194476*1.0001

method = SwT.quad_bound_optim

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
@test Int.(TSt) == [1]
@test Int.(PSt) == [1]
@test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= -0.29033951752714415*0.9999
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 0.678466440366341*1.0001

method = SwT.log_bound_optim

sol, TSt, PSt, DSt = SwT.OptimUpperBoundCrossingTime(
    problem, method, origin, γ, optim_solver)
# @test Int.(TSt) == [1]
# @test Int.(PSt) == [1]
# @test Int.(DSt) == [1]
V0 = SwT.EvalQuad(sol.EP)(x0)
@test V0 <= 0.907989315359373*1.1
τ = SwT.UpperBoundCrossingTime(method, V0, sol.rmin, sol.EP, γ)
@test τ <= 0.9571862425417166*1.1
gmin = SwT.CheckUpperBoundCrossingTime(sol, problem, method, origin, γ)
@test gmin >= -1e-2*norm(sol.EP)

TimeMax = min(max(τ, Tmin), Tmax)
TCross = SwT.FindCrossingTime(problem, x0, (0.0, TimeMax), 1e-5)
@test TCross <= 0.40823414570309885*1.0001
end

end # module
