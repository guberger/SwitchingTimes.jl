module ExampleMain

include("macros.jl")
println("\nNew test")

d = 10
maxSamplesIn = 100
maxSamplesOut = 100

include("escape_compare_inside_10D_test.jl")
include("escape_compare_outside_10D_test.jl")

labels = ("Rabi (2020)", L"$G(V)=-1$", L"$G(V)=-2\gamma V$")
# labels = ("Rabi (2020)", L"$G(V)=-2\gamma V$") # Just for slides

if isdefined(ExampleMain, :cputimes_list_inside) &&
        isdefined(ExampleMain, :cputimes_list_outside)
    fig = PlotComputationTimes(cputimes_list_inside, cputimes_list_outside,
        labels)
    fig.savefig(string("./figures/fig_compare_times_", d, "D_", maxSamplesIn,
        "-", maxSamplesOut, "-", round(Int, time()*1000), ".png"),
        transparent = false, bbox_inches = "tight")
end

end # module
