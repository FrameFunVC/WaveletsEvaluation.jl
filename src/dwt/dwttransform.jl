
SFilterBank(s::Side, w::DiscreteWavelet, scaled=true) =
    scaled ?
        ((s==Primal) ? Filterbank(w) : DualFilterbank(w)) :
        ((s==Primal) ? UnscaledFilterbank(w) : UnscaledDualFilterbank(w))

InverseSFilterBank(s::Side, w::DiscreteWavelet, scaled=true) =
    scaled ?
        ((s==Primal) ? Filterbank(w) : DualFilterbank(w)) :
        ((s==Primal) ? InverseUnscaledFilterbank(w) : InverseUnscaledDualFilterbank(w))

InverseUnscaledFilterbank(w::DiscreteWavelet{T}) where T =
    Filterbank( FilterPair((sqrt(T(2)))*filter(Prl(), Scl(), w), (T(1)/sqrt(T(2)))*filter(Prl(), Wvl(), w)),
                FilterPair((sqrt(T(2)))*filter(Dul(), Scl(), w), (T(1)/sqrt(T(2)))*filter(Dul(), Wvl(), w)) )
InverseUnscaledDualFilterbank(w::DiscreteWavelet{T}) where T =
    Filterbank( FilterPair((T(1)/sqrt(T(2)))*filter(Dul(), Scl(), w), (sqrt(T(2)))*filter(Dul(), Wvl(), w)),
                FilterPair((T(1)/sqrt(T(2)))*filter(Prl(), Scl(), w), (sqrt(T(2)))*filter(Prl(), Wvl(), w)) )
UnscaledFilterbank(w::DiscreteWavelet{T}) where T =
    Filterbank( FilterPair((T(1)/sqrt(T(2)))*filter(Prl(), Scl(), w), (sqrt(T(2)))*filter(Prl(), Wvl(), w)),
                FilterPair((T(1)/sqrt(T(2)))*filter(Dul(), Scl(), w), (sqrt(T(2)))*filter(Dul(), Wvl(), w)) )
UnscaledDualFilterbank(w::DiscreteWavelet{T}) where T =
    Filterbank( FilterPair((sqrt(T(2)))*filter(Dul(), Scl(), w), (T(1)/sqrt(T(2)))*filter(Dul(), Wvl(), w)),
                FilterPair((sqrt(T(2)))*filter(Prl(), Scl(), w), (T(1)/sqrt(T(2)))*filter(Prl(), Wvl(), w)) )


export dwt, idwt, full_dwt, full_idwt
"Perform `L` steps of the wavelet transform"
function dwt!
end

"Perform `L` steps of the inverse wavelet transform"
function idwt!
end

dwt_size(x, fb::Filterbank, bnd::WaveletBoundary) = length(x)

dwt(m::Matrix, w::DiscreteWavelet, bnd::WaveletBoundary, d = 2, L::Int=maxtransformlevels(size(m, 3-d));scaled=true) =
    dwt(m, Primal, w, bnd, d, L;scaled=scaled)

dwt(m::Matrix{T}, s::Side, w::DiscreteWavelet{T}, bnd::WaveletBoundary, d = 2, L::Int=maxtransformlevels(size(m, 3-d));scaled=true) where {T}=
    dwt!(copy(m), s, w, bnd, d, L;scaled=scaled)

function dwt!(m::Matrix{T}, s::Side, w::DiscreteWavelet{T}, bnd::WaveletBoundary, d = 2, L::Int=maxtransformlevels(size(m, 3-d));scaled=true) where {T}
    if d ==2
        ld, ls = size(m)
        u = Array{T}(ls)
        v = Array{T}(ls)
        t = Array{T}(ls)
        fb = SFilterBank(s, w, scaled)
        os = 1
        @inbounds for i in 1:ld
            # @time u .= m[:,i]
            # @time dwt!(v, u, fb, bnd, L, t)
            # m[:,i] .= v
            copyto!(u, 1, m, os, ls)
            dwt!(v, u, fb, bnd, L, t)
            copyto!(m, os, v, 1, ls)
            os += ls
        end
    end
    m
end

function idwt!(m::Matrix{T}, s::Side, w::DiscreteWavelet{T}, bnd::WaveletBoundary, d = 2, L::Int=maxtransformlevels(size(m, 3-d));scaled=true) where {T}
    if d ==2
        ld, ls = size(m)
        u = Array{T}(ls)
        v = Array{T}(ls)
        t = Array{T}(ls)
        fb = InverseSFilterBank(s, w, scaled)
        os = 1
        @inbounds for i in 1:ld
            copyto!(u, 1, m, os, ls)
            idwt!(v, u, fb, bnd, L, t)
            copyto!(m, os, v, 1, ls)
            os += ls
        end
    end
    m
end

dwt(x::AbstractVector, w::DiscreteWavelet, bnd::WaveletBoundary, L::Int=maxtransformlevels(x);scaled=true) =
    dwt(x, Primal, w, bnd, L;scaled=scaled)

idwt(x::AbstractVector, w::DiscreteWavelet, bnd::WaveletBoundary, L::Int=maxtransformlevels(x);scaled=true) =
    idwt(x, Primal, w, bnd, L;scaled=scaled)

dwt(x::AbstractVector, s::Side, w::DiscreteWavelet, bnd::WaveletBoundary, L::Int=maxtransformlevels(x);scaled=true) =
    dwt(x, SFilterBank(s, w, scaled), bnd, L)


idwt(x, s::Side, w::DiscreteWavelet, bnd::WaveletBoundary, L::Int=maxtransformlevels(x);scaled=true) =
    idwt(x, InverseSFilterBank(s, w, scaled), bnd, L)

function dwt!(y, x, fb::Filterbank, bnd::WaveletBoundary, L::Int=maxtransformlevels(x), s=similar(y))
    lx = length(x)
    @assert 2^L <= lx
    copyto!(y,x) #copyto!
    @inbounds for l in 1:L
        lx2 = lx >> 1
        l1, l2 = dwtstep_size(lx, fb, bnd)
        DWT.dwtstep!(s, l1, l2, y, lx, fb, bnd)
        copyto!(y, 1, s, 1, lx)
        lx = lx2
    end
end

function dwt(x, fb::Filterbank, bnd::WaveletBoundary,  L::Int=maxtransformlevels(x))
    @assert isdyadic(x)
    T = promote_type(eltype(x), eltype(fb))
    y = zeros(T, dwt_size(x, fb, bnd))
    dwt!(y, x, fb, bnd, L)
    y
end

function idwt!(y, x, fb::Filterbank, bnd::WaveletBoundary, L::Int=maxtransformlevels(x),s=similar(y))
    lx = length(x)
    @assert 2^L <= lx
    copyto!(y,x)
    lx = lx >> L
    @inbounds for l in L:-1:1
        lx2 = lx << 1
        # t = DWT.idwtstep(y[1:lx], y[lx+1:lx2], fb, bnd)
        DWT.idwtstep!(s, lx2, y, lx, lx, fb, bnd)
        copyto!(y, 1, s, 1, lx2)
        lx = lx2
    end
end

function idwt(x, fb::Filterbank, bnd::WaveletBoundary, L::Int=maxtransformlevels(x))
    @assert isdyadic(x)
    T = promote_type(eltype(x), eltype(fb))
    y = zeros(T, dwt_size(x, fb, bnd))
    idwt!(y, x, fb, bnd, L)
    y
end

full_dwt(x, w::DiscreteWavelet{T}, bnd::WaveletBoundary; scaled=true)  where {T} =
    full_dwt(x, Primal, w, bnd; scaled=scaled)

full_idwt(x, w::DiscreteWavelet{T}, bnd::WaveletBoundary; scaled=true)  where {T} =
    full_idwt(x, Primal, w, bnd; scaled=scaled)

function full_dwt(x, s::Side, w::DiscreteWavelet{T}, bnd::WaveletBoundary; scaled=true) where {T}
    coefs = scaling_coefficients(x, s, w, bnd; scaled=scaled)
    dwt(coefs, s, w, bnd; scaled=scaled)
end

function full_idwt(x, s::Side, w::DiscreteWavelet{T}, bnd::WaveletBoundary; scaled=true) where {T}
    scalingcoefs = idwt(x, s, w, bnd; scaled=scaled)
    scaling_coefficients_to_dyadic_grid(scalingcoefs, s, w, bnd; scaled=scaled)
end
