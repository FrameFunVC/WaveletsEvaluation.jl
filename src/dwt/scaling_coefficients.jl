# scaling_coefficients.jl
import ..Util: ndyadicscales, isdyadic

function scaling_coefficients(f::Function, w::DWT.DiscreteWavelet, L::Int, fembedding, a::Real=0, b::Real=1; side::Side=Dul(), options...)
  T = promote_type(eltype(w), eltype(a), eltype(b))
  a = T(a); b = T(b)
  flt = filter(side, Scl(), w)
  (b-a)*scaling_coefficients(x->f((b-a)*x+a), flt, L::Int, fembedding; options...)
end

# Function on the interval (0 1) to scaling coefficients
function scaling_coefficients{T}(f::Function, s::CompactSequence{T}, L::Int, fembedding; options...)
  x = linspace(T(0), T(1), 1<<L + 1)[1:end-1]
  fcoefs = map(f, x)
  @assert eltype(fcoefs)==T
  filter = _scalingcoefficient_filter(s)
  fembedding == nothing && (fembedding = FunctionEmbedding(k -> f(k*T(2)^(-T(L)))))
  scaling_coefficients(fcoefs, filter, fembedding; options...)
end

scaling_coefficients{T}(f::AbstractArray, w::DiscreteWavelet{T}, bnd::PeriodicBoundary; options...) =
    scaling_coefficients(f, _scalingcoefficient_filter(filter(Dul(), Scl(), w)), PeriodicEmbedding(); options...)

scaling_coefficients!{T}(c::AbstractArray, f::AbstractArray, w::DiscreteWavelet{T}, bnd::PeriodicBoundary; options...) =
    scaling_coefficients!(c, f, _scalingcoefficient_filter(filter(Dul(), Scl(), w)), PeriodicEmbedding(); options...)

# Function evaluations on a dyadic grid to scaling coefficients
function scaling_coefficients{T}(f::AbstractArray, filter::CompactSequence{T}, fembedding; n::Int=length(f), options...)
  @assert isdyadic(f)
  c = Array(T,n)
  scaling_coefficients!(c, f, filter, fembedding; options...)
  c
end

function scaling_coefficients!{T}(c, f, filter::CompactSequence{T}, fembedding; offset::Int=0, options...)
  # TODO write a convolution function
  # convolution between low pass filter and function values gives approximation of scaling coefficients
  for j in offset:offset+length(c)-1
    ci = zero(T)
    for l in firstindex(filter):lastindex(filter)
      ci += filter[l]*fembedding[f, j-l]
    end
    c[j+1-offset] = T(1)/T(sqrt(length(f)))*ci
  end
end

_scalingcoefficient_filter(f::CompactSequence) =
    reverse(CompactSequence(cascade_algorithm(f, 0), f.offset))


function scaling_coefficients_to_dyadic_grid{T}(scaling_coefficients::AbstractArray, w::DWT.DiscreteWavelet{T}, bnd::WaveletBoundary, d=ndyadicscales(scaling_coefficients); grid=false, options...)
  function_evals = zeros(T,DWT.scaling_coefficients_to_dyadic_grid_length(w,d))
  scratch = zeros(DWT.scaling_coefficients_to_dyadic_grid_scratch_length(w,d))
  scratch2 = zeros(DWT.scaling_coefficients_to_dyadic_grid_scratch2_length(w,d))

  scaling_coefficients_to_dyadic_grid!(function_evals, scaling_coefficients, w, bnd, scratch, scratch2; options...)
  grid ?
    (return function_evals, linspace(T(0), T(1), length(function_evals)+1)[1:end-1]) :
    (return function_evals)
end

# Scaling coefficients to function evaluations on dyadic grid (assumes periodic extension)
function scaling_coefficients_to_dyadic_grid!{T}(function_evals::AbstractArray, scaling_coefficients::AbstractArray, w::DWT.DiscreteWavelet{T}, ::DWT.PeriodicBoundary, scratch=nothing, scratch2=nothing; grid=false, options...)
  d = Int(log2(length(function_evals)))
  @assert length(function_evals) == length(scratch)
  @assert DWT.scaling_coefficients_to_dyadic_grid_scratch2_length(w, d) == length(scratch2)
  @assert isdyadic(scaling_coefficients)
  j = ndyadicscales(scaling_coefficients)
  function_evals[:] = T(0)
  for (c_i,c) in enumerate(scaling_coefficients)
    k = c_i - 1
    DWT.evaluate_periodic_in_dyadic_points!(scratch, DWT.Prl(), DWT.Scl(), w, j, k, d, scratch2, nothing)
    scale!(scratch, c)
    for i in 1:length(function_evals)
      function_evals[i] += scratch[i]
    end
    # function_evals[:] += c*DWT.evaluate_periodic_in_dyadic_points(Prl(), Scl(), w, j, k, d)
  end
  nothing
end

scaling_coefficients_to_dyadic_grid_length(w::DWT.DiscreteWavelet, d::Int) = 1<<d

scaling_coefficients_to_dyadic_grid_scratch_length(w::DWT.DiscreteWavelet, d::Int) =
    scaling_coefficients_to_dyadic_grid_length(w::DWT.DiscreteWavelet, d::Int)
scaling_coefficients_to_dyadic_grid_scratch2_length(w::DWT.DiscreteWavelet, d::Int) =
    DWT.evaluate_periodic_in_dyadic_points_scratch_length(DWT.Prl(), DWT.Scl(), w, d, 0, d)

function DWT.support(side::Side, n::Int, i::Int, l::Int, w::DiscreteWavelet)
  kind, j, k = wavelet_index(n,i,l)
  support(side, kind, w, j, k)
end

"""
  The index ([:scaling/:wavelet], j, k) in the (scaling+wavelet) sequence for coefficient i in a sequence of length n after l dwt synthesis_lowpassfilter

  For example, the indices of a sequence with 4 elements after
  0 dwt steps
    (Scl(), 2, 0),    (Scl(), 2, 1),    (Scl(), 2, 2),     (Scl(), 2, 3)
  1 dwt step
    (Scl(), 1, 0),    (Scl(), 1, 1),    (Wvl(), 1, 0),     (Wvl(), 1, 1)
  2 dwt steps
    (Scl(), 0, 0),    (Wvl(), 0, 0),    (Wvl(), 1, 0),     (Wvl(), 1, 1)
"""
function wavelet_index(n::Int, i::Int, l::Int)
  if i > n/(1<<l)
    j = level(n,i)
    k = mod(i-1,1<<j)
    Wvl(), j, k
  else
    Scl(), Int(log2(n))-l, i-1
  end
end

coefficient_index(kind::Scl, j::Int, k::Int)::Int = k+1
coefficient_index(kind::Wvl, j::Int, k::Int)::Int = 1<<j+k+1

function level(n::Int, i::Int)
  (i == 1 || i == 2) && (return 0)
  for l in 1:round(Int,log2(n))
    if i <= (1<<(l+1))
      return l
    end
  end
end
