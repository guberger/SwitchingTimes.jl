origin = SwT.origin_inside
obj = SwT.Escape_Obj

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

Random.seed!(0)
method_list = (SwT.quad_bound_pre_all, SwT.quad_bound_optim, SwT.log_bound_optim)
# method_list = (SwT.quad_bound_pre_all, SwT.log_bound_optim) # Just for slides

κ_list, bounds_list, cputimes_list_inside = MakeRandomTestsUpperBoundsCompare(
    d, R0, R1, obj, method_list, origin, maxSamplesIn)
fig = PlotUpperBoundsCompare(κ_list, bounds_list, obj, method_list, 1,
    2:length(method_list), location = "top") # Used "bottom" for slides
fig.savefig(string("./figures/fig_compare_EscapeIn_", d, "D_", maxSamplesIn,
    "-", round(Int, time()*1000), ".png"),
    transparent = false, bbox_inches = "tight")
