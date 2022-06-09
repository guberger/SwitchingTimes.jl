using LinearAlgebra
using JuMP

@enum OptimMethod begin
    quad_bound_pre_sep = 1 # Each hyperplane separetely
    quad_bound_pre_all = 2 # Intersection of hyperplanes
    quad_bound_optim = 3
    log_bound_optim = 4
end
@enum OriginPosition origin_inside = 1 origin_outside = 2

function QuadMatrix(model::Model, d::Int)
    return @variable(model, [1:d+1, 1:d+1], Symmetric)
end

function QuadMatrixConvex(model::Model, d::Int)
    # Q = @variable(model, [1:d, 1:d], PSD)
    # b = @variable(model, [1:d])
    # c = @variable(model)
    return @variable(model, [1:d+1, 1:d+1], Symmetric)
end

function QuadMatrixConvexCentered(model::Model, d::Int)
    # Q = @variable(model, [1:d, 1:d], PSD)
    Q = @variable(model, [1:d, 1:d], Symmetric)
    b = zeros(d)
    c = @variable(model)
    return [Q b; b' c]
end

function SumMatrix(model::Model, Mat_list, λ_list)
    return @expression(model, sum(x -> x[1].*x[2], zip(λ_list, Mat_list)))
end

function solveLyapunovEquation(A::Matrix{T}, γ::T) where T
    EYE = Matrix{T}(I, size(A))
    Ashifted = A' + γ*EYE
    return lyap(Ashifted, EYE)
end

include("models.jl")

function OptimUpperBoundCrossingTime(problem::ProblemSpecifications,
        method::OptimMethod, origin::OriginPosition, γ, optim_solver;
        do_print = true)
    if Int(origin) == 1
        @assert IsOriginInside(problem.R1)
    else
        @assert IsOriginOutside(problem.R1)
    end

    model = Model(optim_solver)
    # set_time_limit_sec(model, maxTime)

    if Int(problem.obj) == 1
        error("Not implemented yet.")
    end

    if Int(problem.obj) == 2
        if Int(origin) == 1
            if Int(method) == 1
                sol_full = MODEL_escape_inside_quad_bound_pre(
                    model, problem.A, problem.R1)
            elseif Int(method) == 2
                sol_full = MODEL_escape_inside_quad_bound_pre(
                    model, problem.A, problem.R1)
            elseif Int(method) == 3
                sol_full = MODEL_escape_inside_quad_bound_optim(
                    model, problem.A, problem.R0, problem.R1)
            elseif Int(method) == 4
                sol_full = MODEL_escape_inside_log_bound_optim(
                    model, problem.A, problem.R0, problem.R1, γ)
            end
        end
        if Int(origin) == 2
            if Int(method) == 1
                sol_full = MODEL_escape_outside_quad_bound_pre_sep(
                    model, problem.A, problem.R1)
            elseif Int(method) == 2
                sol_full = MODEL_escape_outside_quad_bound_pre_all(
                    model, problem.A, problem.R1)
            elseif Int(method) == 3
                sol_full = MODEL_escape_outside_quad_bound_optim(
                    model, problem.A, problem.R0, problem.R1)
            elseif Int(method) == 4
                sol_full = MODEL_escape_outside_log_bound_optim(
                    model, problem.A, problem.R0, problem.R1, γ)
            end
        end
    end

    if do_print
        for (t, p, s) in zip(sol_full[2], sol_full[3], sol_full[4])
            println(sprint(Base.show, MIME"text/plain"(), t))
            println(sprint(Base.show, MIME"text/plain"(), p))
            println(sprint(Base.show, MIME"text/plain"(), s))
        end
    end

    return sol_full
end

function CheckUpperBoundCrossingTime(sol, problem::ProblemSpecifications,
        method::OptimMethod, origin::OriginPosition, γ)
    if Int(origin) == 1
        @assert IsOriginInside(problem.R1)
    else
        @assert IsOriginOutside(problem.R1)
    end

    if Int(problem.obj) == 1
        error("Not implemented yet.")
    end

    if Int(problem.obj) == 2
        if Int(origin) == 1
            if Int(method) == 3
                return VERIF_escape_inside_quad_bound_optim(
                    sol, problem.A, problem.R0, problem.R1)
            elseif Int(method) == 4
                return VERIF_escape_inside_log_bound_optim(
                    sol, problem.A, problem.R0, problem.R1, γ)
            else
                @warn("Not implemented yet.")
            end
        end
        if Int(origin) == 2
            if Int(method) == 3
                return VERIF_escape_outside_quad_bound_optim(
                    sol, problem.A, problem.R0, problem.R1)
            elseif Int(method) == 4
                return VERIF_escape_outside_log_bound_optim(
                    sol, problem.A, problem.R0, problem.R1, γ)
            else
                @warn("Not implemented yet.")
            end
        end
    end
end
