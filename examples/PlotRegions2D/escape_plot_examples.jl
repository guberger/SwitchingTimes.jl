module ExampleMain

include("macros.jl")
println("\nNew test")

method = SwT.quad_bound_pre_sep
# method = SwT.quad_bound_pre_all
method = SwT.quad_bound_optim
method = SwT.log_bound_optim

np = 300

##
## Inside
origin = SwT.origin_inside
xlims = [-2.7, 2.7]
ylims = [-2.7, 2.7]
ifig = 0

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

ifig += 1
fig = PlotRegions2D(problem, method, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end
fig.savefig(string("./figures/fig_PlotRegions_EscapeIn_2D_", ifig, ".png"),
    transparent = false, bbox_inches = "tight")

## Rotation matrix
A = [-0.1 1.0; -1.0 -0.1]
problem.A = A

ifig += 1
fig = PlotRegions2D(problem, method, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end
fig.savefig(string("./figures/fig_PlotRegions_EscapeIn_2D_", ifig, ".png"),
    transparent = false, bbox_inches = "tight")

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

## Jordan matrix
A = [-1.0 3.0; 0.0 -1.0]
problem = SwT.ProblemSpecifications(A, R0, R1, SwT.Escape_Obj)

ifig += 1
fig = PlotRegions2D(problem, method, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end
fig.savefig(string("./figures/fig_PlotRegions_EscapeOut_2D_", ifig, ".png"),
    transparent = false, bbox_inches = "tight")

## Rotation matrix
A = [-0.1 1.0; -1.0 -0.1]
problem.A = A

ifig += 1
fig = PlotRegions2D(problem, method, origin, x0, xlims, ylims, np)
for ax in fig.get_axes()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
end
fig.savefig(string("./figures/fig_PlotRegions_EscapeOut_2D_", ifig, ".png"),
    transparent = false, bbox_inches = "tight")

end # module