origin = SwT.origin_outside
obj = SwT.Escape_Obj

side_length = 2.0
xc = vcat(-side_length*2.0, -side_length*2.0, zeros(d - 2))
b1 = [Matrix{Float64}(I, d, d); -Matrix{Float64}(I, d, d)]
c1p = ones(2*d)*side_length
c1 = c1p + b1*xc
E1s = [SwT.QuadMatrixAffine(b1[i, :]/2.0, -c1[i]) for i = 1:length(c1)]
R1 = SwT.ConvexSetCap(E1s)
R0 = SwT.ConvexSetHull([xc])
x0 = -ones(d)*side_length*0.5 + xc

Random.seed!(0)
method_list = (SwT.quad_bound_pre_all, SwT.quad_bound_optim, SwT.log_bound_optim)
# method_list = (SwT.quad_bound_pre_all, SwT.log_bound_optim) # Just for slides

κ_list, bounds_list, cputimes_list_outside = MakeRandomTestsUpperBoundsCompare(
    d, R0, R1, obj, method_list, origin, maxSamplesOut)
fig = PlotUpperBoundsCompare(κ_list, bounds_list, obj, method_list, 1,
    2:length(method_list), location = "bottom")
fig.savefig(string("./figures/fig_compare_EscapeOut_", d, "D_", maxSamplesOut,
    "-", round(Int, time()*1000), ".png"),
    transparent = false, bbox_inches = "tight")
