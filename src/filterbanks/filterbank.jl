# filterbank.jl

module Filterbanks
using InfiniteVectors, ..Embeddings
using InfiniteVectors: _firstindex, _lastindex
import Base: transpose, eltype, getindex, adjoint
export FilterPair, FilterMatrix, PolyphaseMatrix, Filterbank

export polyphasematrix, modulationmatrix, polyphase_analysis!, polyphase_synthesis!,
    lowpassfilter, highpassfilter, primal_lowpassfilter, primal_highpassfilter,
    dual_lowpassfilter, dual_highpassfilter


"A pair of two filters."
struct FilterPair{F1 <: InfiniteVector,F2 <: InfiniteVector}
    f1  ::  F1
    f2  ::  F2
end

eltype(::Type{FilterPair{F1,F2}}) where {F1,F2} = eltype(F1)

lowpassfilter(fp::FilterPair) = fp.f1

highpassfilter(fp::FilterPair) = fp.f2


function getindex(fp::FilterPair, k::Int)
    if k == 1
        fp.f1
    elseif k == 2
        fp.f2
    else
        throw(BoundsError())
    end
end

adjoint(fb::FilterPair) = FilterPair(fb.f1', fb.f2')

"A 2x2 matrix of filters."
struct FilterMatrix{F11,F12,F21,F22}
    a11 ::  F11
    a12 ::  F12
    a21 ::  F21
    a22 ::  F22
end

PolyphaseMatrix = FilterMatrix

eltype(::Type{FilterMatrix{F11,F12,F21,F22}}) where {F11,F12,F21,F22} = eltype(F11)

transpose(m::FilterMatrix) = FilterMatrix(m.a11, m.a21, m.a12, m.a22)
adjoint(m::FilterMatrix) = FilterMatrix(m.a11', m.a21', m.a12', m.a22')


# Done removed: call(m::FilterMatrix, z) = [ztransform(m.a11, z) ztransform(m.a12, z); ztransform(m.a21, z) ztransform(m.a22, z)]
(m::FilterMatrix)(z) = [ztransform(m.a11, z) ztransform(m.a12, z); ztransform(m.a21, z) ztransform(m.a22, z)]

polyphasematrix(fp::FilterPair) = FilterMatrix(evenpart(fp[1]), evenpart(fp[2]), oddpart(fp[1]), oddpart(fp[2]))

modulationmatrix(fp::FilterPair) = FilterMatrix(fp[1], alternating(fp[1]), fp[2], alternating(fp[2]))



"A Filterbank groups several objects related to a two-phase filterbank."
struct Filterbank{P1 <: FilterMatrix, P2 <: FilterMatrix, FP1 <: FilterPair, FP2 <: FilterPair}
    "The primal filter pair, to be used on the synthesis side."
    primal_pair     ::  FP1
    "The dual filter pair, to be used on the analysis side."
    dual_pair       ::  FP2
    "The polyphase matrix for the synthesis side."
    pm_synthesis    ::  P2
    "The polyphase matrix for the analysis side."
    pm_analysis     ::  P1
end

Filterbank(lowpass::InfiniteVector) = Filterbank(FilterPair(lowpass, alternating_flip(lowpass)))

Filterbank(primal_lowpass::InfiniteVector, dual_lowpass::InfiniteVector) =
    Filterbank(FilterPair(primal_lowpass, alternating_flip(dual_lowpass)),
        FilterPair(dual_lowpass, alternating_flip(primal_lowpass)))

# If no dual pair is given, assume orthogonality.
# TODO: verify orthogonality
Filterbank(primal_pair::FilterPair) = Filterbank(primal_pair, primal_pair)

Filterbank(primal_pair::FilterPair, dual_pair::FilterPair) =
    Filterbank(primal_pair, dual_pair, polyphasematrix(primal_pair), polyphasematrix(dual_pair)')

eltype(::Type{Filterbank{P1,P2,FP1,FP2}}) where {P1,P2,FP1,FP2} = eltype(P1)

primal_lowpassfilter(fb::Filterbank) = lowpassfilter(fb.primal_pair)
primal_highpassfilter(fb::Filterbank) = highpassfilter(fb.primal_pair)

dual_lowpassfilter(fb::Filterbank) = lowpassfilter(fb.dual_pair)
dual_highpassfilter(fb::Filterbank) = highpassfilter(fb.dual_pair)

# "Return a temporary array with elements of the given type and of length L. Memory is allocated only
# on the first call of this function and reused afterwards."
# @generated temparray{T,L}(::Type{T}, ::Type{Val{L}}) = quote
#     a = $(Array(T,L-1))
#     return a
# end

# Split the (finite) signal x embedded in (infinite) embedding into a low pass signal y1
# and a high pass signal y2. The filters are given by the polyphasematrix f.
function polyphase_analysis!(y1, y2, x, f::PolyphaseMatrix, embedding)
    Heven = f.a11
    Hodd = f.a12
    Geven = f.a21
    Godd = f.a22

    T = eltype(y1)
    # Determine for which range the convolution of the filters with x results
    # in evaluations of x that not occur outside of the embedding.
    # Remark (TODO?): this could be even more efficient.
    L = length(x)
    lower_i = min(length(y1)-1, maximum(map(_lastindex,(Heven,Geven,Godd,Hodd))))
    (L%2 == 0) ?
      upper_i = L>>1 - 1 + minimum(map(_firstindex,(Heven,Geven,Godd,Hodd))) :
      upper_i = L>>1 - L%2 + minimum(map(_firstindex,(Heven,Geven,Godd,Hodd)))
    upper_i = max(0, upper_i)
    # Lower boundary region
    @inbounds for i in 0:lower_i
        y1i = zero(T)
        y2i = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            y1i += Heven[l] * embedding[x, 2*i-2*l]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            y2i += Geven[l] * embedding[x, 2*i-2*l]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            y1i += Hodd[l] * embedding[x, 2*i-2*l+1]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            y2i += Godd[l] * embedding[x, 2*i-2*l+1]
        end
        y1[i+1] = y1i
        y2[i+1] = y2i
    end

    # Middle region
    @inbounds for i in lower_i+1:upper_i-1
      # for l in minimum(map(_firstindex,(Heven,Geven,Hodd,Godd))):maximum(map(_lastindex,(Heven,Geven,Hodd,Godd)))
      #   @assert (0<=2(i-l)<L) && (0<=2(i-l)+1<L)
      # end
      # @assert 0<=i<min(length(y1),length(y2))
        y1i = zero(T)
        y2i = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            y1i += Heven[l] * x[2*i-2*l+1]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            y2i += Geven[l] * x[2*i-2*l+1]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            y1i += Hodd[l] * x[2*i-2*l+2]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            y2i += Godd[l] * x[2*i-2*l+2]
        end
        y1[i+1] = y1i
        y2[i+1] = y2i
    end

    # Upper boundary region
    @inbounds for i in upper_i:length(y1)-1
        y1i = zero(T)
        y2i = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            y1i += Heven[l] * embedding[x, 2*i-2*l]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            y2i += Geven[l] * embedding[x, 2*i-2*l]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            y1i += Hodd[l] * embedding[x, 2*i-2*l+1]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            y2i += Godd[l] * embedding[x, 2*i-2*l+1]
        end
        y1[i+1] = y1i
        if i+1 <= length(y2)
            y2[i+1] = y2i
        end
    end
end

function polyphase_synthesis!(x, y1, y2, f::PolyphaseMatrix, embedding)
    Heven = f.a11
    Geven = f.a12
    Hodd = f.a21
    Godd = f.a22

    T = eltype(x)
    l1 = length(y1)
    l2 = length(y2)

    for j in 0:length(x)>>1 - 1
        xj_e = zero(T)
        xj_o = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            xj_e += Heven[l] * embedding[y1,j-l, l1]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            xj_e += Geven[l] * embedding[y2,j-l, l2]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            xj_o += Hodd[l] * embedding[y1,j-l, l1]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            xj_o += Godd[l] * embedding[y2,j-l, l2]
        end
        x[2*j+1] = xj_e
        x[2*j+2] = xj_o
    end
end

function polyphase_analysis!(y::AbstractVector{T}, l1::Int, l2::Int, x::AbstractVector{T}, L::Int, f::Filterbanks.PolyphaseMatrix, embedding) where T
    Heven = f.a11
    Hodd = f.a12
    Geven = f.a21
    Godd = f.a22

    # Determine for which range the convolution of the filters with x results
    # in evaluations of x that not occur outside of the embedding.
    # Remark (TODO?): this could be even more efficient.


    lower_i = min(l1-1, maximum(map(_lastindex,(Heven,Geven,Godd,Hodd))))
    (L%2 == 0) ?
      upper_i = L>>1 - 1 + minimum(map(_firstindex,(Heven,Geven,Godd,Hodd))) :
      upper_i = L>>1 - L%2 + minimum(map(_firstindex,(Heven,Geven,Godd,Hodd)))
    upper_i = max(0, upper_i)

    # Lower boundary region
    @inbounds for i in 0:lower_i
        y1i = zero(T)
        y2i = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            y1i += Heven[l] * embedding[x, 2*i-2*l, L]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            y2i += Geven[l] * embedding[x, 2*i-2*l, L]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            y1i += Hodd[l] * embedding[x, 2*i-2*l+1, L]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            y2i += Godd[l] * embedding[x, 2*i-2*l+1, L]
        end
        y[i+1] = y1i
        y[l1+i+1] = y2i
    end

    # Middle region
    @inbounds for i in lower_i+1:upper_i-1
        y1i = zero(T)
        y2i = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            y1i += Heven[l] * x[2*i-2*l+1]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            y2i += Geven[l] * x[2*i-2*l+1]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            y1i += Hodd[l] * x[2*i-2*l+2]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            y2i += Godd[l] * x[2*i-2*l+2]
        end
        y[i+1] = y1i
        y[l1+i+1] = y2i
    end

    # Upper boundary region
    @inbounds for i in upper_i:l1-1
        y1i = zero(T)
        y2i = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            y1i += Heven[l] * embedding[x, 2*i-2*l, L]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            y2i += Geven[l] * embedding[x, 2*i-2*l, L]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            y1i += Hodd[l] * embedding[x, 2*i-2*l+1, L]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            y2i += Godd[l] * embedding[x, 2*i-2*l+1, L]
        end
        y[i+1] = y1i
        if i+1 <= l2
            y[l1+i+1] = y2i
        end
    end
end

function polyphase_synthesis!(x, L::Int, y, l1::Int, l2::Int, f::PolyphaseMatrix, embedding)
    Heven = f.a11
    Geven = f.a12
    Hodd = f.a21
    Godd = f.a22

    T = eltype(x)
    l = L>>1 - 1

    @inbounds for j in 0:l
        xj_e = zero(T)
        xj_o = zero(T)
        for l = _firstindex(Heven):_lastindex(Heven)
            xj_e += Heven[l] * embedding[y,j-l, l1]
        end
        for l = _firstindex(Geven):_lastindex(Geven)
            xj_e += Geven[l] * embedding[y,j-l, l2, l1]
        end
        for l = _firstindex(Hodd):_lastindex(Hodd)
            xj_o += Hodd[l] * embedding[y,j-l, l1]
        end
        for l = _firstindex(Godd):_lastindex(Godd)
            xj_o += Godd[l] * embedding[y,j-l, l2, l1]
        end
        x[(j<<1)+1] = xj_e
        x[(j<<1)+2] = xj_o
    end
end

end # module
