# recursion.jl

"""
The scaling function of a wavelet with filtercoeficients h evaluated in diadic points

`x[k] = 2^(-L)k for k=0..K`

where K is the length of h. Thus L=0, gives the evaluation of the
scaling function in the points [0,1,...,K], and L=1, the points [0,.5,1,...,K].
"""
function recursion_algorithm(side::Side, kind::Kind, w::DiscreteWavelet{T}, L=0; options...) where {T}
    # Wrapper for allocating memory and using recursion_algorithm!

    f = zeros(T, recursion_length(side, kind, w, L))
    recursion_algorithm!(f, side, kind, w, L; options...)
    f
end

function recursion_algorithm(s::CompactInfiniteVector{T}, L; options...) where {T}
    # Wrapper for allocating memory and using recursion_algorithm!
    f = zeros(T, recursion_length(s, L))
    recursion_algorithm!(f, s, L; options...)
    f
end

# Convenience function: convert wavelet to filter
recursion_algorithm!(f::AbstractArray{T,1}, side::Side, kind::Kind, w::DiscreteWavelet{T}, L=0; options...)  where {T} =
    recursion_algorithm!(f::AbstractArray, filter(side, kind, w), L; options...)

# Convenience function: convert compact InfiniteVector to array with filter coefficients
recursion_algorithm!(f::AbstractArray{T,1}, s::CompactInfiniteVector{T}, L; options...)  where {T} = recursion_algorithm!(f::AbstractArray, subvector(s), L; options...)

function recursion_algorithm!(f::AbstractArray{T,1}, h::AbstractArray{T,1}, L; tol = sqrt(eps(T)), options...) where {T}
    @assert L >= 0
    sqrt2 = sqrt(T(2))
    @assert sum(h)≈sqrt2
    N = length(h)
    # find ϕ(0), .. , ϕ(N-1) by solving a eigenvalue problem
    # The eigenvector with eigenvalue equal to 1/√2 is the vector containting ϕ(0), .. , ϕ(N-1)
    # see http://archive.cnx.org/contents/d279ef3c-661f-4e14-bba8-70a4eb5c0bcf@7/computing-the-scaling-function-the-recursion-algorithm for more mathematics

    # Create matrix from filter coefficients
    H = DWT._get_H(h)
    # Find eigenvector eigv
    E = eigen(H)

    # Select eigenvector with eigenvalue equal to 1/√2
    index = findall(abs.(E.values .- 1 ./sqrt2) .< tol)
    @assert length(index) > 0
    i = index[1]
    V = E.vectors[:,i]
    @assert norm(imag(V)) < tol
    eigv = real(V)
    (abs(sum(eigv)) < tol*100) && (warn("Recursion algorithm is not convergent"))

    # Normalize the eigenvector to have sum equal to 1
    eigv /= sum(eigv)

    # Initialize loop
    eigv_length = N
    f[1:1<<L:end] = eigv
    # # Find intermediate values ϕ(1/2), .. ,ϕ(N-1 -1/2)
    K = 2
    for m in 1:L
        for n in 1+1<<(L-m):1<<(L-m+1):length(f)
            t = T(0)
            for i in max(1,ceil(Int, (2n-1-length(f))/(1<<L)+1)):min(N, 1+(n-1)>>(L-1))
                t += h[i].*f[2*n-1-(1<<L)*(i-1)]
            end
            f[n] = sqrt2*t
        end
    end
    nothing
end

"Expected length of the output array of the recursion_algorithm. "
recursion_length(side::Side, kind::Kind, w::DiscreteWavelet{T}, L::Int)  where {T} =
    recursion_length(support_length(side, kind, w), L)
recursion_length(f::CompactInfiniteVector, L::Int) = recursion_length(sublength(f)-1, L)
recursion_length(H::Int, L::Int) = (L >= 0) ? (1<<L)*H+1 : 1

"The equispaced grid where recursion_algorithm evaluates. "
dyadicpointsofrecursion(side::Side, kind::Kind, w::DiscreteWavelet{T}, j::Int, k::Int, d::Int)  where {T} =
    T(2)^(-j).*(k .+ dyadicpointsofrecursion(side, kind, w, d-j))
function dyadicpointsofrecursion(side::Side, kind::Kind, w::DiscreteWavelet{T}, L::Int) where {T}
    s = support(side, kind, w)
    H = support_length(side, kind, w)
    if L >= 0
        LinRange(T(s[1]), T(s[2]), (1<<L)*H+1)
    else
        # include zero, therefore, do some rounding
        (1<<-L)*(cld(s[1],1<<-L):fld(s[2],1<<-L))
    end
end

# Put filter coeffients in matrix H as on page 66 of wavelets2014.pdf (Adhemar, Huybrechs)
function _get_H(h)
    N = length(h)
    T = eltype(h)
    M = zeros(T,N,N)
    for l in 1:2:N
        hh = h[l]
        for m in 1:ceil(Int,N/2)
            M[l>>1+m,1+(m-1)*2] = hh
        end
    end
    for l in 2:2:N
        hh = h[l]
        for m in 1:N>>1
            M[l>>1+m,2+(m-1)*2] = hh
        end
    end
    M
end
