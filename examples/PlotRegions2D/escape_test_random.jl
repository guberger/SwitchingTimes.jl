module ExampleMain

include("macros.jl")
println("\nNew test")

method = SwT.quad_bound_pre_sep
method1 = SwT.quad_bound_pre_all
method = SwT.quad_bound_optim
method2 = SwT.log_bound_optim

np = 300

##
## Inside
origin = SwT.origin_inside
xlims = [-2.9, 2.9]
ylims = [-2.9, 2.9]

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

A = SwT.GenerateMatrixDiagonal(2)
println(eigvals(A))
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)

fig = PlotRegions2D(problem, method1, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end
fig = PlotRegions2D(problem, method2, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end

##
## Outside
origin = SwT.origin_outside
xlims = [-4.0, 4.0]
ylims = [-4.0, 4.0]
ifig = 0

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

A = SwT.GenerateMatrixDiagonal(2)
println(eigvals(A))
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)

fig = PlotRegions2D(problem, method1, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end
fig = PlotRegions2D(problem, method2, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end

end # module