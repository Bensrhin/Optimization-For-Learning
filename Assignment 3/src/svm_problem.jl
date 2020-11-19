using Random
using LinearAlgebra
using ProximalOperators
using Plots
#########################################################
#########################################################
#########################################################

function svm_problem()
	lambda = 0.001
	sigma = 0.5
	K(q,p) = exp(-0.5/(sigma^2)*norm(q-p)^2)
	x,y = svm_data()

	N = length(y)

	Q = [y[i]*K(x[i], x[j]) *y[j]/lambda for i = 1:N, j = 1:N]
	q = zeros(N)
	f = Quadratic(Q,q)
	hstar = Conjugate(HingeLoss(ones(N),1/N))
	hstar_i = Conjugate(HingeLoss([1.0], 1/N))

	nu0 = zeros(N)

	return f,hstar,nu0,Q,q,hstar_i
end

# Helper function for generating problem data
function svm_data()
	mt = MersenneTwister(0b11011110)
	N = 500

	n = 4

	xmax = 1
	xmin = -1
	x = [(xmax - xmin)*rand(mt,n) .+ xmin for _ in 1:N]

	nrefs = 40
	mtref = MersenneTwister(0xC0FFEE)
	refs = [(xmax - xmin)*rand(mtref,n) .+ xmin for _ in 1:nrefs]
	refy = [i > nrefs/2 ? 1 : -1 for i = 1:nrefs]

	f(x) = sum(i-> exp(-0.5/(.3^2)*norm(x - refs[i])^2)*refy[i], 1:nrefs)

	y = [sign(f(xi)) for xi in x]
	return x,y
end

"""
Returns the step size of the algorithms PG and APG:
"""
function step_size(Q)
	L = maximum(eigvals(Q))
	return 1/L
end

"""
The proximal gradient method
	ν_{k+1} = prox_{γ h*}( ν_k - γ ∇f(ν_k) )
and returns the solution ν*
"""

function solution()
	f,hstar,nu0,Q,q,hstar_i = svm_problem()
	γ = step_size(Q)
	ν = nu0
	ν0 = nu0
	stop = 1
	k = -1
	while stop > 10^(-14)
		df, _ = gradient(f, ν)
		ν, _ = prox(hstar, ν - γ * df, γ)
		stop = sqrt((ν - ν0)'*(ν - ν0))
		ν0 = ν
		k += 1
	end
	νs = ν
	return νs
end

νs = solution()

function PG(k)
	f,hstar,nu0,Q,q,hstar_i = svm_problem()
	γ = step_size(Q)
	distance = zeros(k)
	ν = nu0
	plot(title = "The norm  ||v_k - v*||", xlabel = "k : number of iterations",ylabel = "The norm (log scale)", legend=:topright, yaxis=:log)
	for i=1:k
		df, _ = gradient(f, ν)
		ν, _ = prox(hstar, ν - γ * df, γ)
		distance[i] = sqrt((ν - νs)'*(ν - νs))
	end
	display(plot!(0:k-1, distance, label = string("PG"), lw = 3))
	savefig("PG.png")
end


"""
The accelerated proximal gradient method:

	y_k = ν_k + β_k( ν_k - ν_{k-1})
	ν_{k+1} = prox_{γ h*}( y_k - γ ∇f(u_k) )
"""
function APG_3(k)
	f,hstar,nu0,Q,q,hstar_i = svm_problem()
	γ = step_size(Q)
	"""
	APG with (3)
	"""
	#plot(title = "The norm  ||v_k - v*||", xlabel = "k : number of iterations",ylabel = "The norm (log scale)", legend=:topright, yaxis=:log)
	distance = zeros(k)
	ν = nu0
	ν_1 = nu0
	t = 1
	for i=1:k
		tk = t
		t = 0.5 * (1 + sqrt(1 + 4* t^2))
		β = (tk - 1) / t
		y = ν + β*(ν - ν_1)
		ν_1 = ν
		df, _ = gradient(f, y)
		ν, _ = prox(hstar, y - γ * df, γ)
		distance[i] = sqrt((ν - νs)'*(ν - νs))
	end
	display(plot!(0:k-1, distance, label = string("APG (3)"), lw = 2))
end

function APG_4(k)
	f,hstar,nu0,Q,q,hstar_i = svm_problem()
	γ = step_size(Q)
	"""
	APG with (4)
	"""
	ν = nu0
	ν_1 = nu0
	distance = zeros(k)
	μ = minimum(eigvals(Q))
	β = (1 - sqrt(μ*γ)) / (1 + sqrt(μ*γ))
	for i=1:k
		y = ν + β*(ν - ν_1)
		ν_1 = ν
		df, _ = gradient(f, y)
		ν, _ = prox(hstar, y - γ * df, γ)
		distance[i] = sqrt((ν - νs)'*(ν - νs))
	end
	display(plot!(0:k-1, distance, label = string("APG (4)"), lw = 2))
	savefig("APG.png")
end


"""
The coordinate gradient descent method:

"""
function CG(n)
	f,hstar,nu0,Q,q,hstar_i = svm_problem()
	N = size(Q, 2)
	"""
	Uniform choice
	"""
	γ = step_size(Q)
	#plot(title = "The norm  ||v_k - v*||", xlabel = "k : number of iterations (axis scaled with 1/N )",ylabel = "The norm (log scale)", legend=:bottomright, yaxis=:log)
	distance = zeros(n)
	ν = nu0
	for k=1:(n*N)
		i = rand(1:N)
		arr, _ = prox(hstar_i, [ν[i] - γ * Q[i,:]'*ν], γ)
		ν[i] = arr[1]
		if k%N == 0
			distance[k÷N] = sqrt((ν - νs)'*(ν - νs))
		end
	end
	display(plot!(1:n, distance, label = string("CG"), lw = 3))
end

function CG_wise(n)
	f,hstar,nu0,Q,q,hstar_i = svm_problem()
	N = size(Q,2)
	"""
	The coordinate-wise step-size
	"""
	distance = zeros(n)
	ν = nu0
	for k=1:(n*N)
		i = rand(1:N)
		γ = 1/Q[i,i]
		arr, _ = prox(hstar_i, [ν[i] - γ * Q[:,i]'*ν], γ)
		ν[i] = arr[1]
		if k%N==0
			distance[k÷N] = sqrt((ν - νs)'*(ν - νs))
		end
	end
	display(plot!(1:n, distance, label = string("CG wise"), lw = 2))
	savefig("CG.png")
end
