using Plots
using LinearAlgebra
using Random
using Statistics

include("flux_wrapper.jl")

const MT = MersenneTwister


#########################################################
#########################################################
#########################################################

struct GD{P,T}
	prob::P
	gamma::T
	k::Base.RefValue{Int} # The iteration counter is a refernce so it can be updated
end

GD(prob::NNProblem, gamma=1.0) = GD(prob, gamma, Ref(0))

function run!(
		alg::GD, npasses=1, status_callback=()->nothing, status_interval=1)

	prob, gamma = alg.prob, alg.gamma # Extract from struct
	k = alg.k[] # This is how we get the value of a reference

	callback = throttle(status_callback, status_interval) # Only call the status function every <status_interval> seconds

	# Allocate arrays for the iterates and gradients
	x,g = grad(prob,1)
	x = similar(x)
	g = similar(g)

	for _ in 1:npasses
		# Sum up full gradient and get iterate
		x_1, g_1 = grad(prob,1)
		g .= g_1./nfunc(prob)
		x .= x_1
		for i in 2:nfunc(prob)
			_, g_i = grad(prob,i)
			g .+= g_i./nfunc(prob)
		end
		x .= x .- gamma.*g # Take gradient step
		update!(prob,x) # Update the problem with the new iterate
		k += 1
		callback() # Show result of optimization
	end

	alg.k[] = k # Update iteration counter of struct
	status_callback() # Show final result
	nothing # return nothing
end

#########################################################
#########################################################
#########################################################

#+++
# Add stochastic gradient implementation here
function SG(γ, β, n)
	nnp, status_callback = problem()
	alg = GD(nnp, γ)
	prob, γ = alg.prob, alg.gamma # Extract from struct

	p = alg.k[] # This is how we get the value of a reference
	callback = throttle(status_callback, 1) # Only call the status function every <status_interval> seconds

	_, _, get_true_solution = get_poly_model()
	ys = get_true_solution(prob.x,prob.y)
	distance = zeros(n)
	meanloss = zeros(n)
	N = nfunc(prob)
	ν, _ = grad(prob, 1)
	for k=1:(n*N)
		γk = γ / (1 + β * k)
		i = rand(1:N)
		_, df = grad(prob, i)
		ν .= ν .- γk .* (df)
		println(length(ν))
		update!(prob,ν)

		if k%N == 0
			p += 1
			callback() # Show result of optimization
			meanloss[k÷N] = mean(d -> ((x,y) = d; prob.loss(prob.model(x),y)), zip(prob.x,prob.y))
			# OBS! Remove this check when using a neural network model
			distance[k÷N] = norm(ys - get_params(prob))

		end
	end

	alg.k[] = p # Update iteration counter of struct
	status_callback() # Show final result
	nothing # return nothing


	println("Mean loss: ", meanloss[n])
	plot(title = "The mean loss", xlabel = "k : number of iterations", legend=:topright, yaxis=:log)
	display(plot!(0:n-1, meanloss, label = string("\\gamma=",γ, ", \\beta=", β), lw = 3, c=:red))
	savefig(string("loss.png"))

	# OBS! Remove this check when using a neural network model
	println("Distance to solution: ",distance[n])
	plot(title = "The norm  ||v_k - v*||", xlabel = "k : number of iterations",ylabel = "The norm (log scale)", legend=:topright, yaxis=:log)
	display(plot!(0:n-1, distance, label = string("\\gamma=",γ, ", \\beta=", β), lw = 3))
	savefig(string("sol.png"))

end


#+++

#########################################################
#########################################################
#########################################################
### Polynomial Fitting

function get_poly_model()
	### Define polynomial model
	order = 2
	# order = 5
	l1 = x -> [x[1]^p for p in 1:order] # First layer is a polynomial feature map
	l2 = Dense(order, 1, identity)
	model = Chain(l1,l2) # Put the layers together
	loss = (yhat,y) -> norm(yhat - y)^2 # Least square loss

	### Calculate true solution
	function get_true_solution(xs,ys)
		A = mapreduce(transpose,vcat,[[l1(x);1.0] for x in xs])
		b = getindex.(ys,1)
		p_solution = (A'*A)\A'*b
		return p_solution
	end
	return model, loss, get_true_solution
end

function get_nn_model()
	### Define neural network model
	sigma = x -> leakyrelu(x,0.2)
	l1 = Dense(1, 30, sigma)
	li = [Dense(30 , 30, sigma) for _ in 1:4]
	ln = Dense(30, 1, identity)
	model = Chain(l1,li...,ln)
	loss = (yhat,y) -> norm(yhat - y)^2
	return model, loss
end

# Set up problem and create a callback for ploting status
function problem()
	### This is the function we want to approximate, a 5th order polynomial
	order_sol = 5
	coeff = randn(MT(111),order_sol+1)
	fsol(x) = [dot([x[1]^p for p in 0:order_sol], coeff)]

	### Define data
	x_range = range(-2,2,length=100)
	num2array= x -> [x]
	xs = num2array.(rand(MT(222),2000).*(x_range[end]-x_range[1]).+x_range[1])
	ys = fsol.(xs)

	# Get model
	model, loss, get_true_solution = get_poly_model()
	p_solution = get_true_solution(xs,ys)

	# model, loss = get_nn_model()

	### Define callback function for displaying training info
	function status_callback()
		meanloss = mean(d -> ((x,y) = d; loss(model(x),y)), zip(xs,ys))
		println("Mean loss: ", meanloss)

		# OBS! Remove this check when using a neural network model
		println("Distance to solution: ", norm(p_solution - get_params(nnp)))

		# Plot over range
		plot(x_range, [fsol(x)[1] for x in x_range], c=:blue, label="")
		plot!(x_range, [model([x])[1] for x in x_range], c=:red, label="", show=true)
	end

	nnp = NNProblem(model, loss, xs, ys)
	return return nnp, status_callback
end

### Create algorithm and run for 100 number of passes over the data
nnp, status_callback = problem()
gd = GD(nnp, 0.1) # Distance to solution should be decreasing
# @time run!(gd, 200, status_callback, 1)

function task5()
	nnp, status_callback = problem()
	gd = GD(nnp, 0.009)
	@time run!(gd, 100, status_callback, 1)
end

function task6()
	nnp, status_callback = problem()
	gd = GD(nnp, 0.02)
	@time run!(gd, 100, status_callback, 1)
end
