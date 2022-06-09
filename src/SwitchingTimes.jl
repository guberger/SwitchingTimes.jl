module SwitchingTimes

@enum Objective Reach_Obj = 1 Escape_Obj = 2

# If E is a (d+1)x(d+1) symmetric matrix and x is a d-elements vector, we let
# V(E;x) = [x,1]'*P*[x,1] = x'*E[1:d,1:d]*x + 2*E[1:d,d+1]'*x + E[d+1,d+1].
# We say that E is convex if E[1:d,1:d] is positive semi-definite.
# We say that E is centered if E[1:d,d+1] = 0.
# We say that E is affine if E[1:d,1:d] = 0.
# We say that E is scalar if only E[d+1,d+1] is non-zero.
# The sublevel set {x : V(E;x)<r} is denoted by SL(E,r).

function QuadMatrix(Q::Matrix, b::Vector, c)
    return [Q b; b' c]
end

function QuadMatrixCentered(Q::Matrix, c::Float64)
    b = zeros(size(Q)[1])
    return [Q b; b' c]
end

function QuadMatrixAffine(b::Vector, c)
    d = length(b)
    return [zeros(d, d) b; b' c]
end

function QuadMatrixScalar(d::Int, c)
    return [zeros(d, d) zeros(d); zeros(d)' c]
end

function QuadMatrixZero(d::Int)
    return zeros(d + 1, d + 1)
end

# Computes the Lie derivative of V(P;x) with respect to the vector field A*x
function DerivQuadMatrix(A, E)
    d = size(A, 1)
    AA = [A zeros(d); zeros(d)' 0.0]
    return E*AA + AA'*E
end

function SumMatrix(Mat_list, λ_list)
    return sum(x -> x[1].*x[2], zip(λ_list, Mat_list))
end

_EvalQuad(Q, b, c, x) = x'*Q*x + 2*b'*x + c
EvalQuad(E, d) = let Q = E[1:d, 1:d], b = E[1:d, d+1], c = E[d+1, d+1]
    x -> _EvalQuad(Q, b, c, x)
end
EvalQuad(E) = EvalQuad(E, size(E, 1) - 1)
_EvalDerivQuad(Q, b, c, x, dx) = 2*dx'*Q*x + 2*b'*dx
EvalDerivQuad(A, E, d) = let A = A,
        Q = E[1:d, 1:d], b = E[1:d, d+1], c = E[d+1, d+1]
    x -> _EvalDerivQuad(Q, b, c, x, A*x)
end
EvalDerivQuad(A, E) = EvalDerivQuad(A, E, size(E, 1) - 1)

# Convex set in R^d as the intersection of ellipsoids.
# The ellipsoids are represented by SL(E,0) where E is convex.
struct ConvexSetCap
    Es::Vector{Matrix{Float64}}
end

# Convex set in R^d as the convex hull of ellipsoids.
# The ellipsoids are represented by SL(E,0) where E is convex.
struct ConvexSetHull
    Es::Vector{Matrix{Float64}}
    Ps::Vector{Vector{Float64}}
end
ConvexSetHull(Es::Vector{Matrix{Float64}}) = ConvexSetHull(Es, [])
ConvexSetHull(Ps::Vector{Vector{Float64}}) = ConvexSetHull([], Ps)

IsOriginInside(set::ConvexSetCap) = all(x -> x[end, end] < 0.0, set.Es)
IsOriginOutside(set::ConvexSetCap) = any(x -> x[end, end] > 0.0, set.Es)

mutable struct ProblemSpecifications
    A::Matrix{Float64}
    R0::ConvexSetHull
    R1::ConvexSetCap
    obj::Objective
end

include("optimize.jl")

function InnerRadius(EP)
    λmax = maximum(eigvals(Symmetric(EP[1:end-1, 1:end-1])))
    @assert λmax > 0.0
    return 1.0/λmax
end

function UpperBoundCrossingTime(method::OptimMethod, V0, Vmin, EP, γ)
    if Int(method) in (1, 2)
        return max((V0/Vmin-1)/InnerRadius(EP), 0.0)
    elseif Int(method) == 3
        return max(V0 - Vmin, 0.0)
    elseif Int(method) == 4
        return log(max(V0/Vmin, 1.0))/(2*γ)
    end
end

## Miscellaneous

function StabilityMargin(A)
    margin = -maximum(real.(eigvals(A)))
    @assert margin > 0.0
    return margin
end

function GenerateMatrixDiagonal(d)
    D = zeros(d, d)
    V = zeros(d, d)
    e = div(d, 2)
    r = d - 2*e
    for i = 1:e
        b = sqrt(rand())
        c = rand()
        Δ = Complex(b^2 - c)
        x1 = -b + sqrt(Δ)
        x2 = -b - sqrt(Δ)
        D[2*i-1:2*i, 2*i-1:2*i] = [real(x1) imag(x1); imag(x2) real(x2)]
        V[:, 2*i-1:2*i] = randn(d, 2)
    end
    if r > 0
        D[d, d] = -rand()
        V[:, d] = randn(d, 1)
    end
    return V*D/V
end

## Compute Crossing Times

using Roots

function FindCrossingTime(problem::ProblemSpecifications, x0, Tspan, δ)
    if Int(problem.obj) == 1
        error("Not implemented yet.")
    end

    if Int(problem.obj) == 2
        return EXACT_FindEscapeTime(problem.A, x0, problem.R1, Tspan, δ)
    end
end

EvalDistQuad(A, x0, E, d) = let A = A, x0 = x0,
        Q = E[1:d, 1:d], b = E[1:d, d+1], c = E[d+1, d+1]
    t -> _EvalQuad(Q, b, c, exp(A*t)*x0)
end
EvalDistQuad(A, x0, E) = EvalDistQuad(A, x0, E, length(x0))
# workaround for asking that at least one time in [t, t+δ] is outside R1
imminently_positive(t, f, δ) = f(t + δ/4) > 0.0 ||
    f(t + δ/2) > 0.0 ||
    f(t + δ) > 0.0

function EXACT_FindEscapeTime(A, x0, R1, Tspan, δ)
    Dist_list = [EvalDistQuad(A, x0, E) for E in R1.Es]
    if any(f -> imminently_positive(Tspan[1], f, δ), Dist_list)
        return Tspan[1]
    end
    TEscape = Inf
    for Dist in Dist_list
        roots = find_zeros(Dist, Tspan[1], Tspan[2])
        for t0 in roots
            if t0 <= TEscape &&
                    any(f -> imminently_positive(t0, f, δ), Dist_list)
                TEscape = t0
            end
        end
    end
    return TEscape
end

end # module
