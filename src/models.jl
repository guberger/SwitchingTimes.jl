#
## INSIDE
function MODEL_escape_inside_quad_bound_pre(model, A, R1)
    d = size(A, 1)
    m = length(R1.Es)
    P = solveLyapunovEquation(A, 0.0)
    # The sublevel set SL(EP,rmin) must be inside R1
    νP_ = @variable(model, [1:m]) # coeffs of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    EP = QuadMatrixCentered(P, 0.0)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    for i = 1:m
        @constraint(model, Symmetric(EP - crmin + νP_[i].*R1.Es[i])
            in PSDCone())
    end
    @objective(model, Max, rmin)
    optimize!(model)
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP), EK = nothing,
        νK_ = nothing, μP_ = nothing, rmax = nothing,
        νP_ = value.(νP_), rmin = value(rmin))
    return (sol, [termination_status(model)],
        [primal_status(model)], [dual_status(model)])
end

function MODEL_escape_inside_quad_bound_optim(model, A, R0, R1)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = QuadMatrixConvex(model, d) # Lyapunov function VP
    # The Lie derivative of VP must be smaller than -1 everywhere except in an
    # invariant set not intersecting R1.
    # Asking that the above holds *in R1* except in an invariant set inside R1
    # would not bring anything useful: indeed, the set where this does not hold
    # is a conic containing the origin, so that the only compatible set is an
    # ellipsoid. This imply that -dEP is convex so that the set where the Lie
    # derivative is larger than one must be included inside the invariant set.
    EK = QuadMatrixConvexCentered(model, d) # SL(EK;0) is invariant
    νK_ = @variable(model, [1:m]) # coeffs of R1.Es
    # The sublevel set SL(EP,rmax) must enclose R0
    μP_ = @variable(model, [1:n], lower_bound = 0.0) # coeffs of R0.Es
    rmax = 1.0
    # The sublevel set SL(EP,rmin) must be inside R1
    νP_ = @variable(model, [1:m]) # coeffs of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    DEK = DerivQuadMatrix(A, EK)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = QuadMatrixScalar(d, 1.0)
    crmax = QuadMatrixScalar(d, rmax)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    @constraint(model, -Symmetric(DEK) in PSDCone())
    @constraint(model, -Symmetric(DEP + cDERIV + EK) in PSDCone())
    for i = 1:m
        @constraint(model, Symmetric(EK - νK_[i].*R1.Es[i]) in PSDCone())
        @constraint(model, Symmetric(EP - crmin + νP_[i].*R1.Es[i])
            in PSDCone())
    end
    for i = 1:n
        @constraint(model, -Symmetric(EP - crmax - μP_[i].*R0.Es[i])
            in PSDCone())
    end
    for x in R0.Ps
        y = vcat(x, 1.0)
        @constraint(model, y'*EP*y <= rmax)
    end
    @constraint(model, rmin <= rmax)
    @objective(model, Max, rmin - 0.0001*(tr(EP) + tr(EK)))
    optimize!(model)
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP), EK = value.(EK),
        νK_ = value.(νK_), μP_ = value.(μP_), rmax = rmax,
        νP_ = value.(νP_), rmin = value(rmin))
    return (sol, [termination_status(model)],
        [primal_status(model)], [dual_status(model)])
end

function MODEL_escape_inside_log_bound_optim(model, A, R0, R1, γ)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = QuadMatrixConvex(model, d) # Lyapunov function VP
    # The Lie derivative of VP must be smaller than -1 everywhere except in an
    # invariant set not intersecting R1.
    # Asking that the above holds *in R1* except in an invariant set inside R1
    # would not bring anything useful: indeed, the set where this does not hold
    # is a conic containing the origin, so that the only compatible set is an
    # ellipsoid. This imply that -dEP is convex so that the set where the Lie
    # derivative is larger than one must be included inside the invariant set.
    EK = QuadMatrixConvexCentered(model, d) # SL(EK;0) is invariant
    νK_ = @variable(model, [1:m]) # coeffs of R1.Es
    # The sublevel set SL(EP,rmax) must enclose R0
    μP_ = @variable(model, [1:n], lower_bound = 0.0) # coeffs of R0.Es
    rmax = 1.0
    # The sublevel set SL(EP,rmin) must be inside R1
    νP_ = @variable(model, [1:m]) # coeffs of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    DEK = DerivQuadMatrix(A, EK)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = 2*γ*EP
    crmax = QuadMatrixScalar(d, rmax)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    @constraint(model, -Symmetric(DEK) in PSDCone())
    @constraint(model, -Symmetric(DEP + cDERIV + EK) in PSDCone())
    for i = 1:m
        @constraint(model, Symmetric(EK - νK_[i].*R1.Es[i]) in PSDCone())
        @constraint(model, Symmetric(EP - crmin + νP_[i].*R1.Es[i])
            in PSDCone())
    end
    for i = 1:n
        @constraint(model, -Symmetric(EP - crmax - μP_[i].*R0.Es[i])
            in PSDCone())
    end
    for x in R0.Ps
        y = vcat(x, 1.0)
        @constraint(model, y'*EP*y <= rmax)
    end
    @constraint(model, rmin <= rmax)
    @objective(model, Max, rmin - 0.0001*(tr(EP) + tr(EK)))
    optimize!(model)
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP), EK = value.(EK),
        νK_ = value.(νK_), μP_ = value.(μP_), rmax = rmax,
        νP_ = value.(νP_), rmin = value(rmin))
    return (sol, [termination_status(model)],
        [primal_status(model)], [dual_status(model)])
end

function VERIF_escape_inside_quad_bound_optim(sol, A, R0, R1)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = sol.EP
    EK = sol.EK
    νK_ = sol.νK_
    μP_ = sol.μP_
    rmax = sol.rmax
    νP_ = sol.νP_
    rmin = sol.rmin
    #---------------------------------------------------------------------------
    DEK = DerivQuadMatrix(A, EK)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = QuadMatrixScalar(d, 1.0)
    crmax = QuadMatrixScalar(d, rmax)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    gmin = Inf
    gmin = min(gmin, minimum(eigvals(-Symmetric(DEK))))
    gmin = min(gmin, minimum(eigvals(-Symmetric(DEP + cDERIV + EK))))
    for i = 1:m
        gmin = min(gmin, minimum(eigvals(Symmetric(EK - νK_[i].*R1.Es[i]))))
        gmin = min(gmin,
            minimum(eigvals(Symmetric(EP - crmin + νP_[i].*R1.Es[i]))))
    end
    for i = 1:n
        gmin = min(gmin,
            minimum(eigvals(-Symmetric(EP - crmax - μP_[i].*R0.Es[i]))))
    end
    return gmin
end

function VERIF_escape_inside_log_bound_optim(sol, A, R0, R1, γ)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = sol.EP
    EK = sol.EK
    νK_ = sol.νK_
    μP_ = sol.μP_
    rmax = sol.rmax
    νP_ = sol.νP_
    rmin = sol.rmin
    #---------------------------------------------------------------------------
    DEK = DerivQuadMatrix(A, EK)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = 2*γ*EP
    crmax = QuadMatrixScalar(d, rmax)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    gmin = Inf
    gmin = min(gmin, minimum(eigvals(-Symmetric(DEK))))
    gmin = min(gmin, minimum(eigvals(-Symmetric(DEP + cDERIV + EK))))
    for i = 1:m
        gmin = min(gmin, minimum(eigvals(Symmetric(EK - νK_[i].*R1.Es[i]))))
        gmin = min(gmin,
            minimum(eigvals(Symmetric(EP - crmin + νP_[i].*R1.Es[i]))))
    end
    for i = 1:n
        gmin = min(gmin,
            minimum(eigvals(-Symmetric(EP - crmax - μP_[i].*R0.Es[i]))))
    end
    return gmin
end

#
## OUTSIDE
function MODEL_escape_outside_quad_bound_pre_sep(model, A, R1)
    d = size(A, 1)
    m = length(R1.Es)
    P = solveLyapunovEquation(A, 0.0)
    # The sublevel set SL(EP,rmin) must be outside R1
    λP_ = @variable(model, [1:m]) # sum of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    EP = QuadMatrixCentered(P, 0.0)
    λPxR1 = SumMatrix(model, R1.Es, λP_)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    @constraint(model, Symmetric(EP - crmin + λPxR1) in PSDCone())
    for i = 1:m
        fix(λP_[i], 0.0)
    end
    @objective(model, Max, rmin)
    #---------------------------------------------------------------------------
    rmin_max = -Inf
    TERM_STATUS = []
    PRIM_STATUS = []
    DUAL_STATUS = []
    λP_save = Vector{Float64}(undef, m)
    for i = 1:m
        unfix(λP_[i])
        @constraint(model, λP_[i] >= 0.0)
        optimize!(model)
        #---------------------------------------------------------------------------
        rmin_max = max(rmin_max, value(rmin))
        λP_save = value(λP_[i])
        push!(TERM_STATUS, termination_status(model))
        push!(PRIM_STATUS, primal_status(model))
        push!(DUAL_STATUS, dual_status(model))
        fix(λP_[i], 0.0, force = true)
    end
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP),  λDP_ = nothing,
        μP_ = nothing, rmax = nothing,
        λP_ = λP_save, rmin = rmin_max)
    return (sol, TERM_STATUS, PRIM_STATUS, DUAL_STATUS)
end

function MODEL_escape_outside_quad_bound_pre_all(model, A, R1)
    d = size(A, 1)
    m = length(R1.Es)
    P = solveLyapunovEquation(A, 0.0)
    # The sublevel set SL(EP,rmin) must be outside R1
    λP_ = @variable(model, [1:m], lower_bound = 0.0) # sum of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    EP = QuadMatrixCentered(P, 0.0)
    λPxR1 = SumMatrix(model, R1.Es, λP_)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    @constraint(model, Symmetric(EP - crmin + λPxR1) in PSDCone())
    @objective(model, Max, rmin)
    optimize!(model)
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP),  λDP_ = nothing,
        μP_ = nothing, rmax = nothing,
        λP_ = value.(λP_), rmin = value(rmin))
    return (sol, [termination_status(model)],
        [primal_status(model)], [dual_status(model)])
end

function MODEL_escape_outside_quad_bound_optim(model, A, R0, R1)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = QuadMatrixConvex(model, d) # Lyapunov function VP
    # The Lie derivative of VP must be smaller than -1 everywhere except in R1.
    λDP_ = @variable(model, [1:m], lower_bound = 0.0) # sum of R1.Es
    # The sublevel set SL(EP,rmax) must enclose R0
    μP_ = @variable(model, [1:n], lower_bound = 0.0) # coeffs of R0.Es
    rmax = 1.0
    # The sublevel set SL(EP,rmin) must be outside R1
    λP_ = @variable(model, [1:m], lower_bound = 0.0) # sum of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    λDPxR1 = SumMatrix(model, R1.Es, λDP_)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = QuadMatrixScalar(d, 1.0)
    crmax = QuadMatrixScalar(d, rmax)
    λPxR1 = SumMatrix(model, R1.Es, λP_)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    @constraint(model, -Symmetric(DEP + cDERIV - λDPxR1) in PSDCone())
    @constraint(model, Symmetric(EP - crmin + λPxR1) in PSDCone())
    for i = 1:n
        @constraint(model, -Symmetric(EP - crmax - μP_[i].*R0.Es[i])
            in PSDCone())
    end
    for x in R0.Ps
        y = vcat(x, 1.0)
        @constraint(model, y'*EP*y <= rmax)
    end
    @constraint(model, rmin <= rmax)
    # @objective(model, Max, rmin - 0.0001*(tr(EP) + sum(λP_) + sum(λDP_)))
    # @objective(model, Max, rmin - 0.0001*tr(EP))
    @objective(model, Max, rmin - 0.0001*sum(EP))
    optimize!(model)
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP),  λDP_ = value.(λDP_),
        μP_ = value.(μP_), rmax = rmax,
        λP_ = value.(λP_), rmin = value(rmin))
    return (sol, [termination_status(model)],
        [primal_status(model)], [dual_status(model)])
end

function MODEL_escape_outside_log_bound_optim(model, A, R0, R1, γ)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = QuadMatrixConvex(model, d) # Lyapunov function VP
    # The Lie derivative of VP must be smaller than -1 everywhere except in R1.
    λDP_ = @variable(model, [1:m], lower_bound = 0.0) # sum of R1.Es
    # The sublevel set SL(EP,rmax) must enclose R0
    μP_ = @variable(model, [1:n], lower_bound = 0.0) # coeffs of R0.Es
    rmax = 1.0
    # The sublevel set SL(EP,rmin) must be outside R1
    λP_ = @variable(model, [1:m], lower_bound = 0.0) # sum of R1.Es
    rmin = @variable(model)
    #---------------------------------------------------------------------------
    λDPxR1 = SumMatrix(model, R1.Es, λDP_)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = 2*γ*EP
    crmax = QuadMatrixScalar(d, rmax)
    λPxR1 = SumMatrix(model, R1.Es, λP_)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    @constraint(model, -Symmetric(DEP + cDERIV - λDPxR1) in PSDCone())
    @constraint(model, Symmetric(EP - crmin + λPxR1) in PSDCone())
    for i = 1:n
        @constraint(model, -Symmetric(EP - crmax - μP_[i].*R0.Es[i])
            in PSDCone())
    end
    for x in R0.Ps
        y = vcat(x, 1.0)
        @constraint(model, y'*EP*y <= rmax)
    end
    @constraint(model, rmin <= rmax)
    # @objective(model, Max, rmin - 0.0001*(tr(EP) + sum(λP_) + sum(λDP_)))
    # @objective(model, Max, rmin - 0.0001*tr(EP))
    @objective(model, Max, rmin - 0.0001*sum(EP))
    optimize!(model)
    #---------------------------------------------------------------------------
    sol = (EP = value.(EP),  λDP_ = value.(λDP_),
        μP_ = value.(μP_), rmax = rmax,
        λP_ = value.(λP_), rmin = value(rmin))
    return (sol, [termination_status(model)],
        [primal_status(model)], [dual_status(model)])
end

function VERIF_escape_outside_quad_bound_optim(sol, A, R0, R1)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = sol.EP
    λDP_ = sol.λDP_
    μP_ = sol.μP_
    rmax = sol.rmax
    λP_ = sol.λP_
    rmin = sol.rmin
    #---------------------------------------------------------------------------
    λDPxR1 = SumMatrix(R1.Es, λDP_)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = QuadMatrixScalar(d, 1.0)
    crmax = QuadMatrixScalar(d, rmax)
    λPxR1 = SumMatrix(R1.Es, λP_)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    gmin = Inf
    gmin = min(gmin, minimum(eigvals(-Symmetric(DEP + cDERIV - λDPxR1))))
    gmin = min(gmin, minimum(eigvals(Symmetric(EP - crmin + λPxR1))))
    for i = 1:n
        gmin = min(gmin,
            minimum(eigvals(-Symmetric(EP - crmax - μP_[i].*R0.Es[i]))))
    end
    return gmin
end

function VERIF_escape_outside_log_bound_optim(sol, A, R0, R1, γ)
    d = size(A, 1)
    n = length(R0.Es)
    m = length(R1.Es)
    EP = sol.EP
    λDP_ = sol.λDP_
    μP_ = sol.μP_
    rmax = sol.rmax
    λP_ = sol.λP_
    rmin = sol.rmin
    #---------------------------------------------------------------------------
    λDPxR1 = SumMatrix(R1.Es, λDP_)
    DEP = DerivQuadMatrix(A, EP)
    cDERIV = 2*γ*EP
    crmax = QuadMatrixScalar(d, rmax)
    λPxR1 = SumMatrix(R1.Es, λP_)
    crmin = QuadMatrixScalar(d, rmin)
    #---------------------------------------------------------------------------
    gmin = Inf
    gmin = min(gmin, minimum(eigvals(-Symmetric(DEP + cDERIV - λDPxR1))))
    gmin = min(gmin, minimum(eigvals(Symmetric(EP - crmin + λPxR1))))
    for i = 1:n
        gmin = min(gmin,
            minimum(eigvals(-Symmetric(EP - crmax - μP_[i].*R0.Es[i]))))
    end
    return gmin
end
