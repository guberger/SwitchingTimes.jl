module ExampleMain

include("macros.jl")
println("\nNew test")

method = SwT.quad_bound_pre_sep
method = SwT.quad_bound_pre_all
method = SwT.quad_bound_optim
method = SwT.log_bound_optim

d = 10
Tmax = 1.0e2
Tmin = 1.0
np = 500

A = SwT.GenerateMatrixDiagonal(d)
println(eigvals(A))

##
## Inside
origin = SwT.origin_inside

side_length = 2.0
xc = vcat(-side_length*0.25, zeros(d - 1))
b1 = [Matrix{Float64}(I, d, d); -Matrix{Float64}(I, d, d)]
c1p = ones(2*d)*side_length
c1 = c1p + b1*xc
E1s = [SwT.QuadMatrixAffine(b1[i, :]/2.0, -c1[i]) for i = 1:length(c1)]
R1 = SwT.ConvexSetCap(E1s)
Q0_list = [Matrix{Float64}(I, d, d)/side_length^2]
E0s = [SwT.QuadMatrix(Q0, -Q0*xc, -1.0 + xc'*Q0*xc) for Q0 in Q0_list]
R0 = SwT.ConvexSetHull(E0s)
x0 = -ones(d)*side_length*0.5 + xc
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)

PlotCrossingTime(problem, method, origin, x0, Tmin, Tmax, np)

##
## Outside
origin = SwT.origin_outside

side_length = 2.0
xc = vcat(-side_length*2.0, -side_length*2.0, zeros(d - 2))
b1 = [Matrix{Float64}(I, d, d); -Matrix{Float64}(I, d, d)]
c1p = ones(2*d)*side_length
c1 = c1p + b1*xc
E1s = [SwT.QuadMatrixAffine(b1[i, :]/2.0, -c1[i]) for i = 1:length(c1)]
R1 = SwT.ConvexSetCap(E1s)
R0 = SwT.ConvexSetHull([xc])
x0 = -ones(d)*side_length*0.5 + xc
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)

PlotCrossingTime(problem, method, origin, x0, Tmin, Tmax, np)

end # module
