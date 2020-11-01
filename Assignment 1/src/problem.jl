using Random
using Plots
using LinearAlgebra


include("functions.jl")
"""
    problem_data()

Returns the Q, q, a, and b matrix/vectors that defines the problem in Hand-In 1.

"""
function problem_data()
	mt = MersenneTwister(123)

	n = 20

	Qv = randn(mt,n,n)
	Q = Qv'*Qv
	q = randn(mt,n)

	a = -rand(mt,n)
	b = rand(mt,n)

	return Q,q,a,b
end


"""
=========================== TASK 6 =============================
1 ) range of different step-size => the best choice
2 ) different initial points  	 => this affects the final solution
3 ) x* ∈ S ? and why?
===========================================================
"""

"""
The proximal gradient method:
		x^{k+1} = prox_{λh}(x^{k} - λ∇ϕ(x^{k}))

h = ι_{S}
ϕ = f

=>
		prox_{λh} = prox_box
		∇ϕ        = grad_quad
"""
function prox_h_k(x, Q, q, a, b, γ)
	y = x - γ .* grad_quad(x, Q, q)
	return prox_box(y, a, b, γ)
end

function x_n(n, x0, Q, q, a, b, γ)
	xn = x0
	distance = zeros(n)
	for i = 1:n
		x = xn
		xn = prox_h_k(xn, Q, q, a, b, γ)
		# v[i] is the norm of the step-length/residual
		distance[i] = sqrt((xn-x)'*(xn - x))
	end
	return distance, xn
end

"""
Returning a range of step-sizes

γ = 2/L is the k th step-size
2*l + 1 = number of step-sizes

"""
function step_sizes(Q, l)
	L = maximum(eigvals(Q))
	γ = 2/L
	step = γ/l
	steps = zeros(2*l-1)
	for i = 1:2*l-1
		steps[i] = step * i
	end
	return steps
end

"""
1) range of step-sizes
We plot the norm of the step-length/residual in order to visualize the best step-size.

We take for example (2k - 1 = 11) different step_sizes (k is the index of the upper bound)
"""
function range_step_sizes(l, n)
	Q,q,a,b = problem_data()
	x0 = randn(20)
	plot(title = "Norm of the step-length / residual  : || x_{k+1} - x_k ||", xlabel = "k : number of iterations",ylabel = "The norm (log scale)", bg = :lightgrey, grid="on", legend=:bottomleft)
	i = 1
	for γ in step_sizes(Q, l)
		distance, _ = x_n(n, x0, Q, q, a, b, γ)
		if i<l
			display(plot!(1:n, distance, c=:green, label=string("\\gamma ", i, "< 2/L"),lw = 3, yaxis=:log))
		elseif i==l
			display(plot!(1:n, distance, c=:red, label=string("\\gamma ", i, "= 2/L"), lw = 3, yaxis=:log))
		else
			display(plot!(1:n, distance, c=:blue, label=string("\\gamma ", i, "> 2/L"), lw = 1, yaxis=:log))
		end
		i +=1
	end
	savefig("convergence_norm_step_size.png")
end
"""
Result : As we can see the upper bound γ < 2/ L seems reasonable.
We choose the 2/L as a step-size for the rest of the exercise
since it gives as the best result.
"""

"""
2) Different initial points:
"""
function initial_points(k, n, ϵ)
	Q,q,a,b = problem_data()
	γ = step_sizes(Q, 1)[1] - ϵ # γ= 2/L - ϵ < 2/L
	plot(title = "The norm of || x* - xf || ")
	x0 = randn(20)
	# xf is the first solution
	_, xf = x_n(n, x0, Q, q, a, b, γ)
	distance = zeros(k)
	labels = String[]
	for i=1:k
		x0 = randn(20)
		_, x = x_n(n, x0, Q, q, a, b, γ)
		# the distance between the new solution and the first one
		distance[i] = sqrt((x-xf)'*(x - xf))
		push!(labels, string("d",i))
	end
	display(bar!(labels,distance, label=false, alpha=0.4, yaxis=:log))
	savefig("initial_points.png")
end


"""
======================  Task 7  ===================================

Now we will solve the dual problem using the proximal gradient method

		min_μ f*(μ) + g*(μ)

We have f* a differentiable function and g* is proximable
"""

"""
The proximal gradient method:
		x^{k+1} = prox_{λh}(x^{k} + λ∇ϕ(-x^{k}))

h = ι*_{S}
ϕ = f*

"""
function prox_conj_h_k(μ, Q, q, a, b, γ)
	y = μ + γ .* grad_quadconj(-1 .* μ, Q, q)
	return prox_boxconj(y, a, b, γ)
end


function μ_n(n, μ0, Q, q, a, b, γ)
	μn = μ0
	v = zeros(n)
	for i = 1:n
		μ = μn
		μn = prox_conj_h_k(μn, Q, q, a, b, γ)
		v[i] = sqrt((μn - μ)'*(μn - μ))
	end
	return v, μn
end

"""
1 ) Different step-sizes
"""


function range_dual_step_size(l, n)
	Q,q,a,b = problem_data()
	μ0 = randn(20)
	plot(title = "The step-length / residual  : || \\mu_{k+1} - \\mu_k ||", xlabel = "k : number of iterations",ylabel = "The norm (log scale)", bg = :lightgrey, grid="on", legend=:topright)
	i = 1
	for γ in step_sizes(inv(Q), l)
		distance, _ = μ_n(n, μ0, Q, q, a, b, γ)
		if i<l
			display(plot!(1:n, distance, c=:green, label=string("\\gamma ", i),lw = 2, yaxis=:log))
		elseif i==l
			display(plot!(1:n, distance, c=:red, label=string("\\gamma ", i), lw = 2, yaxis=:log))
			savefig("norm_dual_less.png")
			plot(title = "The step-length / residual  : || \\mu_{k+1} - \\mu_k ||", xlabel = "k : number of iterations",ylabel = "The norm (log scale)", bg = :lightgrey, grid="on", legend=:bottomright)
		else
			display(plot!(1:n, distance, c=:blue, label=string("\\gamma ", i), lw = 2, yaxis=:log))
		end
		i +=1
	end
	savefig("norm_dual_great.png")
end

"""
Compare the solutions from the primal and the one extracted from the dual.
"""

function dual_primal(k, n, ϵ)
	Q,q,a,b = problem_data()
	γp = step_sizes(Q, 1)[1] - ϵ
	γd = step_sizes(inv(Q), 1)[1] - ϵ
	display(plot(title = "The distance between solutions from dual and primal"))
	distance = zeros(k)
	labels = String[]
	for i=1:k
		μ0 = randn(20)
		_, μ = μ_n(n, μ0, Q, q, a, b, γd)
		_, x = x_n(n, μ0, Q, q, a, b, γp)
		xd = dual2primal(μ,Q,q,a,b)
		distance[i] = sqrt((xd - x)'*(xd - x))
		push!(labels, string("d", i))
	end
	display(bar!(labels,distance, label=false, alpha=0.4, yaxis=:log))
	savefig("primal and dual solutions")

end

function iterates(n, ϵ)
	Q,q,a,b = problem_data()
	γd = step_sizes(inv(Q), 1)[1] - ϵ
	γp = step_sizes(inv(Q), 1)[1] - ϵ
	display(plot(title = "f over iterations"))
	μn = randn(20)
	xp = randn(20)
	f = zeros(n)
	g = zeros(n)
	f_s = zeros(n)
	for i = 1:n
		μn = prox_conj_h_k(μn, Q, q, a, b, γd)
		xd = dual2primal(μn,Q,q,a,b)
		xp = prox_h_k(xp, Q, q, a, b, γp)
		f[i] = quad(xd,Q,q)
		g[i] = quad(xp,Q,q)
		f_s[i] = f[i] + (all(a .<= xd .<= b) ? 0.0 : 50)
	end
	display(plot!(1:n, f, label = "f(x^k) : extracted iterates", lw=2))
	display(plot!(1:n, g, label = "f(x^k) : primal iterates", lw=2))
	display(plot!(1:n, f_s, label = "f(x^k) + i_S(x^k) : extracted iterates", lw=2))
	savefig("f_10.png")
end
