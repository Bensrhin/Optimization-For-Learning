# Need Flux v0.10.4 (or at least v0.10 or newer)
using Flux: gradient, params, throttle, Chain, Dense, leakyrelu, relu

#########################################################
#########################################################
#########################################################

struct NNProblem{M,L,X,Y,T<:AbstractFloat,P,NF,NP}
	model::M
	loss::L
	x::X
	y::Y
	opt_var::Vector{T}
	grad::Vector{T}
	ad_params::P

	function NNProblem(model::M,loss::L,x::X,y::Y) where {M,L,X,Y}
		ad_params = params(model)
		P = typeof(ad_params)

		NP = sum(p -> length(p), ad_params)
		T = mapreduce(eltype, promote_type, ad_params)

		opt_var = Vector{T}(undef,NP)
		grad = Vector{T}(undef,NP)

		NF = length(x)

		new{M,L,X,Y,T,P,NF,NP}(model, loss, x, y, opt_var, grad, ad_params)
	end
end

nfunc(::NNProblem{M,L,X,Y,T,P,NF,NP}) where {M,L,X,Y,T,P,NF,NP} = NF
nparams(::NNProblem{M,L,X,Y,T,P,NF,NP}) where {M,L,X,Y,T,P,NF,NP} = NP
Base.eltype(::NNProblem{M,L,X,Y,T,P,NF,NP}) where {M,L,X,Y,T,P,NF,NP} = T

function grad(p::NNProblem, i)
	g = gradient(p.ad_params) do
		y_hat = p.model(p.x[i])
		p.loss(y_hat, p.y[i])
	end

	foreach_linrange(p.ad_params) do param, range
		p.opt_var[range] = vec(param)
		p.grad[range] = vec(g[param])
	end
	return p.opt_var, p.grad
end

function update!(p::NNProblem, v)
	foreach_linrange(p.ad_params) do param, range
		vec(param) .= v[range]
	end
end

function get_params(p::NNProblem)
	foreach_linrange(p.ad_params) do param, range
		p.opt_var[range] = vec(param)
	end
	return p.opt_var
end

# Helper for operating on automatic differentiation parameters
function foreach_linrange(f,ad)
	i = 1
	for p in ad
		p_len = length(p)
		f(p,i:(i+p_len-1))
		i += p_len
	end
end
