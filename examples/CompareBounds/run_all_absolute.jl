module ExampleMain

include("macros.jl")
println("\nNew test")

d = 10
maxSamplesIn = 100
maxSamplesOut = 100

include("escape_absolute_inside_10D_test.jl")
include("escape_absolute_outside_10D_test.jl")

end # module
