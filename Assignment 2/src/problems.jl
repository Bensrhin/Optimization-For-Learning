using Random
using LinearAlgebra
using Plots
using ProximalOperators
"""
	problem(i)

Returns objective of problems 1,2, and 3.
"""
problem(i::Int) = problem(Val{i}())
problem(::Val{1}) = Sum(NormL2(0.5),SqrNormL2(0.5))
problem(::Val{2}) = CubeNormL2(1.0/3.0)
problem(::Val{3}) = SqrNormL2()

"""
	parameters()

Returns objective parameters of the problems 4 and 5.
"""
function parameters()
	mt = MersenneTwister(123)
	mu = 13
	L = 598
	n = 17
	N = 323

	evals = rand(mt,n)
	evals = (evals .- minimum(evals))./(maximum(evals) - minimum(evals)).*(L-mu) .+ mu

	f = svd([randn(mt,N,n-1) ones(N)])
	A = f.U*diagm(sqrt.(evals))*f.Vt
	ai = [A[i,:] for i in 1:size(A,1)]
	l = ones(N)
	return ai,A,l
end

"""
	prescale(A,r)

Returns a prescaling matrix H to be used in Task 9. The argument 0 <= r <= 1
determine the computational effort expended to reduce the condition number, r =
1 gives the best but most expensive prescaling and r = 0 the worst but cheapest.
"""
function prescale(A,r)
	mt = MersenneTwister(159)
	sel = max(size(A,2), round(Int,r*size(A,1)))
	fact = svd(A[1:sel,:])
	H = fact.V*diagm((1-r) .+ r ./ fact.S)
end

"""
TASK 3
"""

"""
iterates1(γ, x, n):   Problem (1)
	Plots the distances between the iterates and the solution 0
	γ:step-size     x: initial point       n: number iterations
"""
function iterates1(γ, x, n)
	f = problem(1)
	distance = zeros(n)
	for i = 1:n
		# (sub) gradient of f at x
		df, _ = gradient(f, x)
		distance[i] = sqrt(x[1]'*x[1])
		x = x - γ .* df
	end

	display(plot!(0:n-1, distance, xlabel = "k : number of iterations",ylabel = "The distance",label = string("\\gamma = ", γ), lw = 3))
end

"""
solve1(k, x):      Problem (1)
	Plots the distances between the iterates and the solution 0
	for each step size among [0.3 0.9 1.8 2.1]
	x: initial point       k: number iterations
"""
function solve1(k, x)
	step_size = [0.3 0.9 1.8 2.1]
	sizes = length(step_size)
	plot(title = "The distance to the solution  : || x_k ||")
	for i = 1:sizes
		γ = step_size[i]
		iterates1(γ, x, k)
	end
	savefig("iterates1.png")
end

"""
step_size1(k, x):
	Find the best step size for each non zero and non negative intial point x.
	And plots the distances of the iterates over n iterations.
"""

function step_size1(k, x)
	step_size = 2*x[1]/(x[1] + 1)
	plot(title = "The distance to the solution  : || x_k ||")
	γ = step_size
	iterates1(γ, x, k)
	savefig("step_size1.1.png")
end



"""
TASK 4
"""

"""
iterates2(γ, x, n):   Problem (2)
	Plots the distances between the iterates and the solution 0
	γ:step-size     x: initial point       n: number iterations
"""
function iterates2(γ, x, n)
	f = problem(2)
	distance = zeros(n)
	x0=x
	for i = 1:n
		# gradient of f at x
		df, _ = gradient(f, x)
		x = x - γ .* df
		distance[i] = sqrt(x'*x)
	end
	display(plot!(1:n, distance, xlabel = "k : number of iterations",ylabel = "The function values",label = string("x = ", x0[1]), lw = 2, yaxis=:log))
end

"""
solve2(k, γ):      Problem (2)
	Plots the distances between the iterates and the solution 0
	if the boolean conv is:
		using the initial points [0.3 0.9 1.8]
	else:
		using the initial points [2.1 3.0 5.0]
	γ: step-size       k: number iterations     conv:boolean
"""
function solve2(k, γ, conv)
	points = conv ? [0.3 0.9 1.8] : [2.1 3.0 5.0]
	sizes = length(points)
	plot(title = "The distance to the solution  : || x_k ||")
	for i = 1:sizes
		x = [points[i]]
		iterates2(γ, x, k)
	end
	conv ? savefig("iterates2.png") : savefig("dive2.png")
end

"""
TASK 5
"""

"""
iterates3(γ, x, n):   Problem (3)
	Plots the distances between the iterates and the solution 0
	γ:step-size     x: initial point       n: number iterations
"""
function iterates3(γ, x, n)
	f = problem(3)
	distance = zeros(n)
	x0=x
	for i = 1:n
		# gradient of f at x
		df, _ = gradient(f, x)
		distance[i] = sqrt(x'*x)
		x = x - γ .* df
	end
	display(plot!(0:n-1, distance, xlabel = "k : number of iterations",ylabel = "The function values",label = string("x= ", x0[1]), lw = 2))
end
"""
solve3(k, x):      Problem (3)
	Plots the distances between the iterates and the solution 0
	for each initial point among [0.2 2.0 20.0]
	γ:step-size       k: number iterations
"""
# Only use k must be less than 200 since we reach 0 and (log(0) raise an exception)
function solve3(k, γ)
	points = [0.2 2.0 20.0]
	sizes = length(points)
	plot(title = string("The distance to the solution for \\gamma =", γ), legend=:topright, yaxis=:log)
	for i = 1:sizes
		x = [points[i]]
		iterates3(γ, x, k)
	end
	savefig(string("iterates3_", γ,".png"))
end

"""
trying a range of step-sizes in order to find the best choice of γ.
search3(x, n):
	Only plots the iterates that converge to the solution over n iterations
	for the initial point x
"""
# Only use k must be less than 200 since we reach 0 and (log(0) raise an exception)
function search3(x, n)
	f = problem(3)
	plot(title = string("The distance to the solution for x =", x[1]), legend=:bottomleft, yaxis=:log)
	for γ in 0.1:0.1:3
		distance = zeros(n)
		x0=x
		for i = 1:n
			# gradient of f at x
			df, _ = gradient(f, x0)
			distance[i] = sqrt(x0'*x0)
			x0 = x0 - γ .* df
		end
		if 0<distance[n] < 10^(-10)
			println(x0[1])
			display(plot!(0:n-1, distance, xlabel = "k : number of iterations",ylabel = "The function values",label = string("\\gamma = ", γ), lw = 2))
		end
	end
	savefig("search3.png")
end


"""
TASK 7
problem (2) and (3) are proximable
"""

"""
prox_iter(γ, x, n, prob):
	plots the norm of the iterates
	- over n iterations
	- starting with the initial point x
	- using the step-size γ
	- prob=2 or 3
"""
function prox_iter(γ, x, n, prob)
	f = problem(prob)
	distance = zeros(n)
	x0=x
	for i = 1:n
		distance[i] = sqrt(x'*x)
		x, _ = prox(f, x, γ)
	end
	display(plot!(0:n-1, distance, xlabel = "k : number of iterations",ylabel = "The norm ||x_k||",label = string("\\gamma = ", γ), lw = 2))
end

"""
prox_solve(k, x, prob):
	Plots the distances between the iterates and the solution 0
	for each step-size among [0.3 3.0 30.0 300.0 3000.0]
	x:initial point       k: number iterations
	prob=2 or 3
"""
function prox_solve(k, x, prob)
	step = [0.3 3.0 30.0 300.0 3000.0]
	plot(title = string("The distance to the solution for x=", x[1]), legend=:topright, yaxis=:log)
	for γ in step
		prox_iter(γ, x, k, prob)
	end
	savefig(string("prox_", prob,"_prob.png"))
end

"""
TASK 8
Solve (3) and (4) with gradient descent
"""

"""
grad_prob4(a, l, x):
 	return the gradient of the problem (4)
"""
function grad_prob4(a, l, x)
	α = l.*a
	n = length(α)
	grad = zeros(length(α[1]))
	for i = 1:n
		grad +=  .- α[i] ./ (2 * (1 + exp(x'*α[i])))
	end
	return grad
end
"""
solve_prob4(k):
	Plots the norm of the gradient over k iterations in the case of Problem (4)
"""
function solve_prob4(k)
	a, A, l = parameters()
	L = 1/8 *maximum(eigvals(A'*A))
	γ = 1.2/L
	x = zeros(length(a[1]))
	plot(title = string("The convergence of the gradient to zero for x=", x[1]), legend=:topright)
	distance = zeros(k)
	for j = 1:k
		grad = grad_prob4(a, l, x)
		x = x - γ .* grad
		distance[j] = sqrt(grad'*grad)
	end
	display(plot!(1:k, distance, xlabel = "k : number of iterations",ylabel = "The log of the distance",label = "\\gamma = 1.2\\/L", lw = 2, c=:green, yaxis=:log))
	savefig(string("grad_", 4,"_prob.png"))
end

"""
Problem (5)
solve_prob5(k):
	Plots the norm of the gradient over k iterations in the case of Problem (5)
"""
function solve_prob5(k)
	a, A, l = parameters()
	L = maximum(eigvals(A'*A))
	γ = 1.2/L
	x = zeros(length(a[1]))
	plot(title = string("The convergence of the gradient to zero for x=", x[1]), legend=:topright)
	distance = zeros(k)
	for j = 1:k
		grad = A'*(A*x - l)
		x = x - γ .* grad
		distance[j] = log(sqrt(grad'*grad))
		distance[j] = sqrt(grad'*grad)
	end
	display(plot!(1:k, distance, xlabel = "k : number of iterations",ylabel = "The log of the distance",label = "\\gamma = 1.2\\/L", lw = 2, c=:brown, yaxis=:log))
	savefig(string("grad_", 5,"_prob.png"))
end

"""
TASK 9
scaled_prob5(k):
	- Plots the norm of the gradient over k iterations in the case of:
	 * The scaled Problem (5)
	- For each parameter r:
	 * Calculate the condition number.
	 * Prints its the value in the labels.
"""

function scaled_prob5(k)
	a, A, l = parameters()
	params = [0 0.2 0.4 0.6 0.8 1.0]
	plot(title = string("The convergence of the gradient to zero for x=", 0), legend=:topright)
	for r in params
		H = prescale(A, r)
		Â = A*H
		λ = eigvals(Â'*Â)
		κ = maximum(λ)/minimum(λ)
		γ = 1.2/maximum(λ)
		distance = zeros(k)
		x = zeros(length(a[1]))
		for j = 1:k
			grad = Â'*(Â*x - l)
			x = x - γ .* grad
			distance[j] = sqrt(grad'*grad)
		end
		display(plot!(1:k, distance, xlabel = "k : number of iterations",ylabel = "The log of the distance",label = string("r = ", r, " , \\kappa = ", round(κ,digits=3)), lw = 2, yaxis=:log))
	end
	savefig(string("prescaled_", 5,"_prob.png"))
end
